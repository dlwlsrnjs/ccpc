"""
위상 행렬 이미지를 입력으로 받아 태스크를 분류하는 CNN 모델 (XAI 지원)

입력: connectivity matrix 이미지 (주파수 대역별, absCPCC/imCPCC별)
출력: 태스크 분류

Subject 독립적 학습 (train/val/test split을 subject 기준으로)
XAI 지원: Grad-CAM, Feature Visualization 등
"""

import os
import re
import glob
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    import torchvision.models as models
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("Warning: torchvision not available. Using basic transforms.")
    # 기본 transforms 정의
    class Compose:
        def __init__(self, transforms_list):
            self.transforms_list = transforms_list
        def __call__(self, img):
            for t in self.transforms_list:
                img = t(img)
            return img
    
    class Resize:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            return img.resize(self.size, Image.BILINEAR)
    
    class ToTensor:
        def __call__(self, img):
            img_array = np.array(img).astype(np.float32) / 255.0
            if len(img_array.shape) == 3:
                img_array = img_array.transpose(2, 0, 1)
            return torch.from_numpy(img_array)
    
    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
        def __call__(self, img_tensor):
            return (img_tensor - self.mean) / self.std
    
    class RandomHorizontalFlip:
        def __call__(self, img):
            import random
            if random.random() > 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            return img
    
    class RandomRotation:
        def __init__(self, degrees):
            self.degrees = degrees
        def __call__(self, img):
            import random
            angle = random.uniform(-self.degrees, self.degrees)
            return img.rotate(angle)
    
    # transforms 모듈 생성
    class TransformsModule:
        Compose = Compose
        Resize = Resize
        ToTensor = ToTensor
        Normalize = Normalize
        RandomHorizontalFlip = RandomHorizontalFlip
        RandomRotation = RandomRotation
    
    transforms = TransformsModule()
    models = None  # models는 None으로 설정

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 실행
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None


# ======================================
# 1. 데이터셋 클래스 (기존과 동일)
# ======================================

class ConnectivityImageDataset(Dataset):
    """
    위상 행렬 이미지를 로드하는 데이터셋
    멀티뷰 학습 지원: 여러 주파수 대역 이미지를 채널로 결합
    """
    
    def __init__(
        self,
        image_dirs: List[str],
        subjects: List[str],
        tasks: List[str],
        image_type: str = 'connectivity_matrix',
        value_type: str = 'absCPCC',
        freq_band: Optional[str] = None,
        crop_top: int = 100,
        crop_bottom: int = 100,
        crop_left: int = 100,
        crop_right: int = 100,
        transform: Optional[transforms.Compose] = None,
        multiview: bool = False,  # 멀티뷰 학습 여부
        use_both_cpcc: bool = False,  # absCPCC와 imCPCC 모두 사용 (멀티뷰 모드에서만)
    ):
        self.image_dirs = image_dirs
        self.subjects = subjects
        self.tasks = tasks
        self.image_type = image_type
        self.value_type = value_type
        self.freq_band = freq_band
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.transform = transform
        self.multiview = multiview
        self.use_both_cpcc = use_both_cpcc
        
        self.unique_tasks = sorted(list(set(tasks)))
        self.task_to_idx = {task: idx for idx, task in enumerate(self.unique_tasks)}
        self.idx_to_task = {idx: task for task, idx in self.task_to_idx.items()}
        
        self.freq_bands = ['alpha', 'delta', 'gamma', 'high_beta', 'low_beta', 'theta']
        
        self.samples = []
        
        for img_dir, subject, task in zip(image_dirs, subjects, tasks):
            if multiview:
                # 멀티뷰: 모든 주파수 대역 이미지 경로 저장
                image_paths = []
                
                if use_both_cpcc:
                    # absCPCC와 imCPCC 모두 사용
                    # 순서: 각 주파수 대역별로 absCPCC, imCPCC 순서로
                    for band in self.freq_bands:
                        # absCPCC
                        img_path_abs = os.path.join(
                            img_dir,
                            f'{image_type}_absCPCC_{band}.png'
                        )
                        # imCPCC
                        img_path_im = os.path.join(
                            img_dir,
                            f'{image_type}_imCPCC_{band}.png'
                        )
                        if os.path.exists(img_path_abs) and os.path.exists(img_path_im):
                            image_paths.append(img_path_abs)
                            image_paths.append(img_path_im)
                        else:
                            break  # 하나라도 없으면 이 폴더는 건너뜀
                    
                    # 모든 주파수 대역의 absCPCC와 imCPCC가 있어야만 샘플로 추가
                    # 6주파수 × 2타입 = 12개 이미지
                    if len(image_paths) == len(self.freq_bands) * 2:
                        self.samples.append({
                            'image_paths': image_paths,  # 12개 이미지 경로
                            'subject': subject,
                            'task': task,
                            'freq_bands': self.freq_bands,
                            'value_types': ['absCPCC', 'imCPCC'] * len(self.freq_bands),
                        })
                else:
                    # 기존 방식: value_type에 지정된 타입만 사용
                    for band in self.freq_bands:
                        img_path = os.path.join(
                            img_dir,
                            f'{image_type}_{value_type}_{band}.png'
                        )
                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                    
                    # 모든 주파수 대역 이미지가 있어야만 샘플로 추가
                    if len(image_paths) == len(self.freq_bands):
                        self.samples.append({
                            'image_paths': image_paths,  # 6개 이미지 경로
                            'subject': subject,
                            'task': task,
                            'freq_bands': self.freq_bands,
                        })
            elif freq_band is not None:
                # 단일 주파수 대역
                img_path = os.path.join(
                    img_dir,
                    f'{image_type}_{value_type}_{freq_band}.png'
                )
                if os.path.exists(img_path):
                    self.samples.append({
                        'image_path': img_path,
                        'subject': subject,
                        'task': task,
                        'freq_band': freq_band,
                    })
            else:
                # 기존 방식: 각 주파수 대역별로 별도 샘플
                for band in self.freq_bands:
                    img_path = os.path.join(
                        img_dir,
                        f'{image_type}_{value_type}_{band}.png'
                    )
                    if os.path.exists(img_path):
                        self.samples.append({
                            'image_path': img_path,
                            'subject': subject,
                            'task': task,
                            'freq_band': band,
                        })
        
        print(f"총 {len(self.samples)}개의 샘플 로드 완료")
        print(f"태스크 개수: {len(self.unique_tasks)}")
        if len(tasks) > 0:
            print(f"태스크 분포: {dict(Counter(tasks))}")
        if multiview:
            if use_both_cpcc:
                print(f"멀티뷰 모드: 각 샘플은 {len(self.freq_bands)}개 주파수 대역 × 2타입(abs+im) = {len(self.freq_bands) * 2}개 이미지 포함")
            else:
                print(f"멀티뷰 모드: 각 샘플은 {len(self.freq_bands)}개 주파수 대역 이미지 포함")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.multiview:
            # 멀티뷰: 여러 주파수 대역 이미지를 채널로 결합
            images = []
            for img_path in sample['image_paths']:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                h, w = img_array.shape[:2]
                img_cropped = img_array[
                    self.crop_top:h-self.crop_bottom,
                    self.crop_left:w-self.crop_right
                ]
                
                img_pil = Image.fromarray(img_cropped)
                
                if self.transform:
                    img_pil = self.transform(img_pil)
                
                images.append(img_pil)
            
            # 이미지를 채널로 결합
            # use_both_cpcc=True: 12개 이미지 → (36, H, W)
            # use_both_cpcc=False: 6개 이미지 → (18, H, W)
            multiview_image = torch.cat(images, dim=0)  # 채널 차원으로 결합
            
            task_idx = self.task_to_idx[sample['task']]
            
            return {
                'image': multiview_image,
                'label': task_idx,
                'task': sample['task'],
                'subject': sample['subject'],
                'freq_bands': sample['freq_bands'],
                'image_paths': sample['image_paths'],  # Grad-CAM을 위해 경로 추가
            }
        else:
            # 단일 이미지 (기존 방식)
            img = Image.open(sample['image_path']).convert('RGB')
            img_array = np.array(img)
            
            h, w = img_array.shape[:2]
            img_cropped = img_array[
                self.crop_top:h-self.crop_bottom,
                self.crop_left:w-self.crop_right
            ]
            
            img_pil = Image.fromarray(img_cropped)
            
            if self.transform:
                img_pil = self.transform(img_pil)
            
            task_idx = self.task_to_idx[sample['task']]
            
            return {
                'image': img_pil,
                'label': task_idx,
                'task': sample['task'],
                'subject': sample['subject'],
                'freq_band': sample.get('freq_band', None),
                'image_path': sample.get('image_path', None),
            }


# ======================================
# 2. XAI 지원 CNN 모델
# ======================================

class XAIClassificationModel(nn.Module):
    """
    XAI를 지원하는 분류 모델
    - Pre-trained ResNet 기반 (Grad-CAM에 적합)
    - Feature map 접근 가능
    - 멀티뷰 학습 지원 (여러 주파수 대역 이미지 입력)
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet18',  # 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
        pretrained: bool = True,
        dropout: float = 0.5,
        input_channels: int = 3,  # 멀티뷰: 18 (3채널 × 6주파수), 단일: 3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.input_channels = input_channels
        
        # Backbone 선택
        if backbone.startswith('resnet'):
            if backbone == 'resnet18':
                backbone_model = models.resnet18(pretrained=pretrained)
                self.feature_dim = 512
                self.last_conv_name = 'layer4'
            elif backbone == 'resnet34':
                backbone_model = models.resnet34(pretrained=pretrained)
                self.feature_dim = 512
                self.last_conv_name = 'layer4'
            elif backbone == 'resnet50':
                backbone_model = models.resnet50(pretrained=pretrained)
                self.feature_dim = 2048
                self.last_conv_name = 'layer4'
            else:
                raise ValueError(f"Unsupported ResNet: {backbone}")
            
            # 첫 번째 Conv layer를 input_channels에 맞게 수정
            # 마지막 FC layer 제거
            features_list = list(backbone_model.children())[:-2]
            
            # 첫 번째 Conv layer를 input_channels에 맞게 수정
            if input_channels != 3:
                first_conv = features_list[0]
                # 새로운 Conv layer 생성 (input_channels 입력)
                new_first_conv = nn.Conv2d(
                    input_channels,
                    first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )
                # Pre-trained weights를 복사 (3채널 → input_channels로 확장)
                if pretrained and input_channels > 3:
                    # 3채널 weights를 input_channels로 복제/평균
                    with torch.no_grad():
                        if input_channels % 3 == 0:
                            # 3채널을 반복하여 복제 (예: 18채널 = 3채널 × 6)
                            repeat_factor = input_channels // 3
                            new_first_conv.weight.data = first_conv.weight.data.repeat(1, repeat_factor, 1, 1) / repeat_factor
                        else:
                            # 평균으로 채움
                            new_first_conv.weight.data = first_conv.weight.data.repeat(1, input_channels // 3 + 1, 1, 1)[:, :input_channels, :, :]
                            new_first_conv.weight.data = new_first_conv.weight.data / (input_channels / 3)
                features_list[0] = new_first_conv
            
            self.features = nn.Sequential(*features_list)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        
        elif backbone.startswith('efficientnet'):
            try:
                import timm
                self.features = timm.create_model(backbone, pretrained=pretrained, features_only=True)
                # EfficientNet의 경우 feature extraction이 다름
                raise NotImplementedError("EfficientNet는 추후 구현 예정")
            except ImportError:
                raise ImportError("EfficientNet 사용을 위해 'timm' 패키지가 필요합니다.")
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Grad-CAM을 위한 hook
        self.gradients = None
        self.activations = None
        
        # 마지막 convolutional layer에 hook 등록
        self._register_hooks()
    
    def _register_hooks(self):
        """Grad-CAM을 위한 hook 등록"""
        # 기존 hook 제거
        self._remove_hooks()
        
        if self.backbone_name.startswith('resnet'):
            # ResNet의 마지막 conv layer 찾기
            target_module = None
            for name, module in self.features.named_modules():
                if name == self.last_conv_name:
                    target_module = module
                    break
            
            if target_module is None:
                # layer4를 직접 찾기
                if hasattr(self.features, 'layer4'):
                    target_module = self.features.layer4
                else:
                    # 마지막 Sequential module 사용
                    target_module = list(self.features.children())[-1]
            
            # Hook 함수 정의 (클로저 문제 해결)
            def forward_hook(module, input, output):
                self.activations = output.detach()
            
            def backward_hook(module, grad_input, grad_output):
                if grad_output is not None and len(grad_output) > 0:
                    self.gradients = grad_output[0].detach()
            
            # Hook 등록
            self._forward_handle = target_module.register_forward_hook(forward_hook)
            self._backward_handle = target_module.register_full_backward_hook(backward_hook)
    
    def _remove_hooks(self):
        """기존 hook 제거"""
        if hasattr(self, '_forward_handle'):
            self._forward_handle.remove()
        if hasattr(self, '_backward_handle'):
            self._backward_handle.remove()
    
    def forward(self, x):
        x = self.features(x)
        x_pooled = self.avgpool(x)
        x_flattened = torch.flatten(x_pooled, 1)
        x = self.classifier(x_flattened)
        return x
    
    def get_feature_maps(self, x):
        """Feature map 추출 (XAI용)"""
        return self.features(x)


# ======================================
# 3. Grad-CAM 구현
# ======================================

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) 구현
    모델이 예측을 할 때 어떤 영역에 주목했는지 시각화
    """
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook 등록
        self._register_hooks()
    
    def _register_hooks(self):
        """Gradient와 activation을 추출하기 위한 hook"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # 모델의 마지막 conv layer에 hook 등록
        if hasattr(self.model, 'features'):
            if isinstance(self.model.features, nn.Sequential):
                # ResNet의 경우 마지막 layer 찾기
                last_layer = None
                for module in reversed(list(self.model.features.modules())):
                    if isinstance(module, (nn.Conv2d, nn.Sequential)) and module != self.model.features:
                        last_layer = module
                        break
                
                if last_layer is None:
                    # 마지막 Sequential module (layer4) 사용
                    last_layer = list(self.model.features.children())[-1]
                
                last_layer.register_forward_hook(forward_hook)
                last_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None, use_guided_grad=False):
        """
        Grad-CAM 생성
        
        Args:
            input_tensor: 입력 이미지 텐서 (1, C, H, W)
            target_class: 시각화할 클래스 (None이면 예측된 클래스)
            use_guided_grad: Guided Grad-CAM 사용 여부
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
            prediction: 예측 결과
        """
        self.model.eval()
        input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Gradient와 activation 가져오기
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            raise ValueError("Gradients or activations not captured. Check hook registration.")
        
        # Gradient의 평균 계산 (GAP)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU로 음수 제거
        
        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam_np = cam.cpu().numpy()
        if HAS_CV2:
            if len(cam_np.shape) == 2:
                cam_resized = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))
            else:
                cam_resized = cv2.resize(cam_np[0], (input_tensor.shape[3], input_tensor.shape[2]))
        else:
            # cv2가 없으면 numpy로 resize
            from scipy.ndimage import zoom
            if len(cam_np.shape) == 2:
                zoom_factors = (input_tensor.shape[2] / cam_np.shape[0], input_tensor.shape[3] / cam_np.shape[1])
                cam_resized = zoom(cam_np, zoom_factors, order=1)
            else:
                zoom_factors = (1, input_tensor.shape[2] / cam_np.shape[1], input_tensor.shape[3] / cam_np.shape[2])
                cam_resized = zoom(cam_np[0], zoom_factors[1:], order=1)
        
        return cam_resized, output.argmax(dim=1).item(), output.softmax(dim=1)[0].cpu().numpy()
    
    def visualize(self, input_tensor, original_image, target_class=None, save_path=None):
        """
        Grad-CAM 시각화
        
        Args:
            input_tensor: 입력 텐서
            original_image: 원본 이미지 (numpy array, H, W, C)
            target_class: 타겟 클래스
            save_path: 저장 경로
        
        Returns:
            visualization: 시각화된 이미지
        """
        if not HAS_CV2 or not HAS_MATPLOTLIB:
            raise ImportError("Grad-CAM visualization requires cv2 and matplotlib. Install with: pip install opencv-python matplotlib")
        
        cam, pred_class, probs = self.generate_cam(input_tensor, target_class)
        
        # CAM을 heatmap으로 변환
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[:, :, ::-1]  # BGR to RGB
        
        # 원본 이미지와 overlay
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
        
        # 원본 이미지 크기에 맞춤
        if original_image.shape[:2] != cam.shape:
            original_image = cv2.resize(original_image, (cam.shape[1], cam.shape[0]))
        
        if len(original_image.shape) == 2:
            original_image = np.stack([original_image] * 3, axis=-1)
        
        # Overlay
        visualization = 0.6 * original_image + 0.4 * heatmap
        visualization = np.clip(visualization, 0, 1)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(visualization)
        axes[2].set_title(f'Overlay (Pred: {pred_class}, Prob: {probs[pred_class]:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        
        plt.close()
        
        return visualization


# ======================================
# 4. 유틸리티 함수
# ======================================

def parse_subject_task_from_dirname(dirname: str) -> Tuple[str, str]:
    """디렉토리명에서 subject와 task 추출"""
    match = re.match(r'sub-([^_]+)_(.+?)_connectivity', dirname)
    if not match:
        raise ValueError(f"Unexpected directory name format: {dirname}")
    return match.group(1), match.group(2)


def split_subjects_independently(
    subjects: List[str],
    tasks: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    participants_file: Optional[str] = None,  # participants.tsv 파일 경로
    stratify_by_age: bool = False,  # 나이 기반 stratified split
) -> Tuple[List[int], List[int], List[int]]:
    """
    Subject 독립적으로 train/val/test split
    나이 기반 stratified split 지원 (validation/test에 나이 분포 고르게)
    """
    unique_subjects = sorted(list(set(subjects)))
    
    # 변수 초기화
    train_subjects = []
    val_subjects = []
    test_subjects = []
    
    if stratify_by_age and participants_file and os.path.exists(participants_file):
        # 나이 기반 stratified split
        try:
            import pandas as pd
            participants_df = pd.read_csv(participants_file, sep='\t')
            
            # Subject ID 매핑 (participants.tsv의 participant_id 형식 확인)
            # 예: 'sub-NDARAA396TWZ' 또는 'NDARAA396TWZ'
            def normalize_subject_id(subj_id):
                if subj_id.startswith('sub-'):
                    return subj_id[4:]  # 'sub-' 제거
                return subj_id
            
            # 나이 정보 수집 및 연령대 그룹화
            subject_ages = {}
            for subj in unique_subjects:
                subj_normalized = normalize_subject_id(subj)
                # participant_id 컬럼 확인
                if 'participant_id' in participants_df.columns:
                    participant_col = 'participant_id'
                elif 'subject_id' in participants_df.columns:
                    participant_col = 'subject_id'
                else:
                    # 첫 번째 컬럼 사용
                    participant_col = participants_df.columns[0]
                
                # participant_id도 정규화해서 비교
                participants_df['normalized_id'] = participants_df[participant_col].apply(
                    lambda x: normalize_subject_id(str(x)) if pd.notna(x) else None
                )
                
                subject_row = participants_df[participants_df['normalized_id'] == subj_normalized]
                if len(subject_row) > 0 and 'age' in subject_row.columns:
                    age = subject_row['age'].values[0]
                    if pd.notna(age):
                        # 연령대 그룹화 (논문: 5-9세, 10-14세, 15-21세)
                        if age < 10:
                            age_group = '5-9'
                        elif age < 15:
                            age_group = '10-14'
                        else:
                            age_group = '15-21'
                        subject_ages[subj] = age_group
                    else:
                        subject_ages[subj] = 'unknown'
                else:
                    subject_ages[subj] = 'unknown'
            
            # 연령대별로 subject 그룹화
            from collections import defaultdict
            age_groups = defaultdict(list)
            for subj in unique_subjects:
                age_group = subject_ages.get(subj, 'unknown')
                age_groups[age_group].append(subj)
            
            print(f"연령대별 Subject 분포:")
            for age_group, subj_list in sorted(age_groups.items()):
                print(f"  - {age_group}세: {len(subj_list)}명")
            
            # 각 연령대별로 train/val/test 분할
            train_subjects = []
            val_subjects = []
            test_subjects = []
            
            for age_group, subj_list in age_groups.items():
                if len(subj_list) < 3:
                    # 너무 적으면 모두 train에
                    train_subjects.extend(subj_list)
                    continue
                
                # 연령대별로 분할
                age_train, age_temp = train_test_split(
                    subj_list,
                    test_size=(1 - train_ratio),
                    random_state=random_state,
                )
                
                age_val_size = val_ratio / (val_ratio + test_ratio)
                age_val, age_test = train_test_split(
                    age_temp,
                    test_size=(1 - age_val_size),
                    random_state=random_state,
                )
                
                train_subjects.extend(age_train)
                val_subjects.extend(age_val)
                test_subjects.extend(age_test)
            
            print(f"\n나이 기반 Stratified Split 결과:")
            print(f"  Train: {len(train_subjects)}명")
            print(f"  Val: {len(val_subjects)}명")
            print(f"  Test: {len(test_subjects)}명")
            
            # 연령대별 분포 확인
            def count_by_age_group(subj_list):
                counts = defaultdict(int)
                for subj in subj_list:
                    counts[subject_ages.get(subj, 'unknown')] += 1
                return counts
            
            train_age_dist = count_by_age_group(train_subjects)
            val_age_dist = count_by_age_group(val_subjects)
            test_age_dist = count_by_age_group(test_subjects)
            
            print(f"\nTrain 연령대 분포: {dict(train_age_dist)}")
            print(f"Val 연령대 분포: {dict(val_age_dist)}")
            print(f"Test 연령대 분포: {dict(test_age_dist)}")
            
            # 연령대별 비율 출력 (나이 분포 확인)
            total_train = len(train_subjects)
            total_val = len(val_subjects)
            total_test = len(test_subjects)
            
            if total_train > 0:
                print(f"\nTrain 연령대 비율:")
                for age_group in ['5-9', '10-14', '15-21']:
                    count = train_age_dist.get(age_group, 0)
                    ratio = count / total_train * 100 if total_train > 0 else 0
                    print(f"  {age_group}세: {count}명 ({ratio:.1f}%)")
            
            if total_val > 0:
                print(f"\nVal 연령대 비율:")
                for age_group in ['5-9', '10-14', '15-21']:
                    count = val_age_dist.get(age_group, 0)
                    ratio = count / total_val * 100 if total_val > 0 else 0
                    print(f"  {age_group}세: {count}명 ({ratio:.1f}%)")
            
            if total_test > 0:
                print(f"\nTest 연령대 비율:")
                for age_group in ['5-9', '10-14', '15-21']:
                    count = test_age_dist.get(age_group, 0)
                    ratio = count / total_test * 100 if total_test > 0 else 0
                    print(f"  {age_group}세: {count}명 ({ratio:.1f}%)")
            
        except Exception as e:
            print(f"⚠ 나이 기반 split 실패, 기본 split 사용: {e}")
            # 기본 split으로 fallback
            stratify_by_age = False
    
    if not stratify_by_age or len(train_subjects) == 0:
        # 기본 split (나이 고려 안 함)
        train_subjects, temp_subjects = train_test_split(
            unique_subjects,
            test_size=(1 - train_ratio),
            random_state=random_state,
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            test_size=(1 - val_size),
            random_state=random_state,
        )
    
    train_subjects_set = set(train_subjects)
    val_subjects_set = set(val_subjects)
    test_subjects_set = set(test_subjects)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx, subject in enumerate(subjects):
        if subject in train_subjects_set:
            train_indices.append(idx)
        elif subject in val_subjects_set:
            val_indices.append(idx)
        elif subject in test_subjects_set:
            test_indices.append(idx)
    
    print(f"\nTrain subjects: {len(train_subjects)} ({len(train_indices)} samples)")
    print(f"Val subjects: {len(val_subjects)} ({len(val_indices)} samples)")
    print(f"Test subjects: {len(test_subjects)} ({len(test_indices)} samples)")
    
    return train_indices, val_indices, test_indices


# ======================================
# 5. XAI 분석 함수
# ======================================

def analyze_with_gradcam(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    num_samples: int = 10,
    save_dir: str = './xai_results/gradcam',
):
    """
    Grad-CAM을 사용하여 모델의 예측을 분석
    
    Args:
        model: 학습된 모델
        dataloader: 분석할 데이터 로더
        device: 디바이스
        num_samples: 분석할 샘플 수
        save_dir: 결과 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    gradcam = GradCAM(model)
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break
        
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        tasks = batch['task']
        image_paths = batch['image_path']
        
        for i in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            # 단일 이미지 처리
            img_tensor = images[i:i+1]
            label = labels[i].item()
            task = tasks[i]
            img_path = image_paths[i]
            
            # 원본 이미지 로드 (transform 전)
            original_img = Image.open(img_path).convert('RGB')
            original_img_array = np.array(original_img)
            
            # Grad-CAM 생성 및 시각화
            save_path = os.path.join(
                save_dir,
                f"gradcam_{sample_count:04d}_{task}_label{label}.png"
            )
            
            try:
                gradcam.visualize(
                    img_tensor,
                    original_img_array,
                    target_class=label,
                    save_path=save_path
                )
                sample_count += 1
                print(f"Processed {sample_count}/{num_samples}: {task}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue


# ======================================
# 6. 메인 학습 함수 (XAI 지원)
# ======================================

def train_model(
    image_base_dir: str = '/home/work/skku/startkit-main/images/connectivity',
    image_type: str = 'connectivity_matrix',
    value_type: str = 'absCPCC',
    freq_band: Optional[str] = None,
    crop_top: int = 100,
    crop_bottom: int = 100,
    crop_left: int = 100,
    crop_right: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_epochs: int = 50,
    device: str = 'cuda',
    save_dir: str = './checkpoints/task_classification_images_xai',
    backbone: str = 'resnet18',
    pretrained: bool = True,
    run_xai: bool = False,  # 학습 후 XAI 분석 실행 여부
    folder_list_txt: Optional[str] = None,  # 완료된 폴더 리스트 txt 파일 경로
    multiview: bool = False,  # 멀티뷰 학습 여부
    use_both_cpcc: bool = False,  # absCPCC와 imCPCC 모두 사용 (멀티뷰 모드에서만)
    use_data_augmentation: bool = False,  # 데이터 증강 사용 여부
    aug_strength: str = 'medium',  # 'light', 'medium', 'strong'
    # 논문 실험용 추가 파라미터
    weight_decay: float = 0.0,  # 논문: 1e-5
    lr_scheduler_patience: int = 5,  # 논문: 5
    lr_scheduler_factor: float = 0.5,  # 논문: 0.5
    early_stopping_patience: int = None,  # 논문: 15 (None이면 사용 안 함)
    use_weighted_loss: bool = False,  # 가중치 교차 엔트로피 손실 사용
    use_metadata: bool = False,  # 메타데이터 통합 사용
    return_metrics: bool = False,  # 평가 지표 반환 (실험용)
    # Advanced Augmentation
    use_mixup: bool = False,  # MixUp 사용
    use_cutmix: bool = False,  # CutMix 사용
    mixup_alpha: float = 0.2,  # MixUp alpha
    cutmix_alpha: float = 1.0,  # CutMix alpha
    use_autoaugment: bool = False,  # AutoAugment 사용
    use_randaugment: bool = False,  # RandAugment 사용
    use_gridmask: bool = False,  # GridMask 사용
    use_freq_masking: bool = False,  # 주파수 대역 masking 사용
    # Optimizer & Scheduler
    optimizer_type: str = 'adamw',  # 'adam', 'adamw', 'lion', 'adafactor'
    scheduler_type: str = 'plateau',  # 'plateau', 'cosine_warm_restarts', 'onecycle'
    use_amp: bool = True,  # Mixed Precision Training
    num_workers: Optional[int] = None,  # DataLoader num_workers (None이면 GPU당 12개 자동 설정)
):
    """모델 학습 (XAI 지원)"""
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA를 사용할 수 없어 CPU를 사용합니다.")
    device = torch.device(device)
    
    # 이미지 디렉토리 수집
    if folder_list_txt and os.path.exists(folder_list_txt):
        # txt 파일에서 완료된 폴더 리스트 읽기
        print(f"완료된 폴더 리스트 사용: {folder_list_txt}")
        with open(folder_list_txt, 'r', encoding='utf-8') as f:
            completed_folders = [line.strip() for line in f if line.strip()]
        print(f"  ✓ {len(completed_folders)}개 완료된 폴더 로드")
        
        all_dirs = completed_folders
    else:
        # 기존 방식: 모든 디렉토리 사용
        all_dirs = [d for d in os.listdir(image_base_dir) 
                    if os.path.isdir(os.path.join(image_base_dir, d))]
        if folder_list_txt:
            print(f"⚠ 경고: {folder_list_txt} 파일을 찾을 수 없습니다. 모든 디렉토리를 사용합니다.")
    
    subjects = []
    tasks = []
    image_dirs = []
    
    for dirname in all_dirs:
        try:
            subject, task = parse_subject_task_from_dirname(dirname)
            img_dir = os.path.join(image_base_dir, dirname)
            
            if freq_band is not None:
                img_path = os.path.join(img_dir, f'{image_type}_{value_type}_{freq_band}.png')
            else:
                img_path = os.path.join(img_dir, f'{image_type}_{value_type}_alpha.png')
            
            if os.path.exists(img_path):
                subjects.append(subject)
                tasks.append(task)
                image_dirs.append(img_dir)
        except ValueError:
            continue
    
    print(f"총 {len(image_dirs)}개의 디렉토리 발견")
    
    # Subject 독립적 split (나이 기반 stratified split)
    # participants.tsv 파일 경로 찾기
    participants_file = None
    if image_base_dir:
        # 여러 가능한 경로 확인
        possible_paths = [
            '/home/work/skku/startkit-main/hbn_eeg_releases/participants.tsv',  # 우선 확인
            os.path.join(os.path.dirname(image_base_dir), '..', '..', 'participants.tsv'),
            os.path.join(os.path.dirname(image_base_dir), '..', 'participants.tsv'),
            '/home/work/skku/startkit-main/data/processed/connectivity/multiband/participants.tsv',
            '/home/work/skku/startkit-main/data/raw/participants.tsv',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                participants_file = path
                print(f"✓ Participants 파일 발견: {participants_file}")
                break
    
    if not participants_file:
        print("⚠ Participants 파일을 찾을 수 없습니다. 기본 split을 사용합니다.")
    
    train_indices, val_indices, test_indices = split_subjects_independently(
        subjects, tasks, 
        random_state=42,
        participants_file=participants_file,
        stratify_by_age=True,  # 나이 기반 stratified split 활성화
    )
    
    # Transform 정의
    if use_data_augmentation:
        # 데이터 증강 사용
        if aug_strength == 'light':
            # 가벼운 증강: 기본 + 약간의 변환
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif aug_strength == 'medium':
            # 중간 증강: 회전, 색상 조정, 기하 변환
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif aug_strength == 'strong':
            # 강한 증강: 모든 변환 + RandomErasing
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ])
        else:
            raise ValueError(f"Unknown augmentation strength: {aug_strength}. Choose from 'light', 'medium', 'strong'")
    else:
        # 기본 증강 (기존 방식)
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 데이터셋 생성
    train_dataset_base = ConnectivityImageDataset(
        image_dirs=image_dirs,
        subjects=subjects,
        tasks=tasks,
        image_type=image_type,
        value_type=value_type,
        freq_band=freq_band,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        transform=transform_train,
        multiview=multiview,
        use_both_cpcc=use_both_cpcc,
    )
    
    val_dataset_base = ConnectivityImageDataset(
        image_dirs=image_dirs,
        subjects=subjects,
        tasks=tasks,
        image_type=image_type,
        value_type=value_type,
        freq_band=freq_band,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        transform=transform_val,
        multiview=multiview,
        use_both_cpcc=use_both_cpcc,
    )
    
    test_dataset_base = ConnectivityImageDataset(
        image_dirs=image_dirs,
        subjects=subjects,
        tasks=tasks,
        image_type=image_type,
        value_type=value_type,
        freq_band=freq_band,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        transform=transform_val,
        multiview=multiview,
        use_both_cpcc=use_both_cpcc,
    )
    
    # Subject 기준으로 샘플 필터링
    train_subjects_set = set([subjects[i] for i in train_indices])
    val_subjects_set = set([subjects[i] for i in val_indices])
    test_subjects_set = set([subjects[i] for i in test_indices])
    
    train_sample_indices = []
    val_sample_indices = []
    test_sample_indices = []
    
    for sample_idx, sample in enumerate(train_dataset_base.samples):
        sample_subject = sample['subject']
        if sample_subject in train_subjects_set:
            train_sample_indices.append(sample_idx)
        elif sample_subject in val_subjects_set:
            val_sample_indices.append(sample_idx)
        elif sample_subject in test_subjects_set:
            test_sample_indices.append(sample_idx)
    
    train_dataset = torch.utils.data.Subset(train_dataset_base, train_sample_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_base, val_sample_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_base, test_sample_indices)
    
    # DataLoader
    # num_workers 설정 (기본값: 4개로 설정)
    if num_workers is None:
        # 기본값 4개 워커
        default_num_workers = 4
    else:
        default_num_workers = num_workers
    print(f"DataLoader num_workers: {default_num_workers}")
    # Shared memory 오류 방지를 위해 pin_memory=False, persistent_workers=False, prefetch_factor=1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=default_num_workers, pin_memory=False, persistent_workers=False,
                              prefetch_factor=1 if default_num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=default_num_workers, pin_memory=False, persistent_workers=False,
                            prefetch_factor=1 if default_num_workers > 0 else None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=default_num_workers, pin_memory=False, persistent_workers=False,
                             prefetch_factor=1 if default_num_workers > 0 else None)
    
    # 모델 생성
    num_classes = len(train_dataset_base.unique_tasks)
    
    # 입력 채널 계산
    if multiview:
        if use_both_cpcc:
            input_channels = 36  # 6주파수 × 2타입(abs+im) × 3채널 = 36채널
        else:
            input_channels = 18  # 6주파수 × 3채널 = 18채널
    else:
        input_channels = 3  # 단일 이미지
    
    model = XAIClassificationModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        input_channels=input_channels,
    ).to(device)
    
    print(f"모델 구조: {backbone}")
    print(f"클래스 수: {num_classes}")
    if multiview:
        if use_both_cpcc:
            print(f"입력 채널: {input_channels} (멀티뷰: 6주파수 × 2타입(abs+im) × 3RGB)")
        else:
            print(f"입력 채널: {input_channels} (멀티뷰: 6주파수 × 3RGB)")
    else:
        print(f"입력 채널: {input_channels} (단일 이미지)")
    if use_data_augmentation:
        print(f"데이터 증강: 활성화 (강도: {aug_strength})")
    else:
        print(f"데이터 증강: 비활성화 (기본 증강만 사용)")
    
    # Loss와 Optimizer
    # 가중치 교차 엔트로피 손실 (클래스 불균형 고려)
    if use_weighted_loss:
        # 클래스별 역빈도 가중치 계산
        task_counts = Counter([sample['task'] for sample in train_dataset_base.samples])
        total_samples = len(train_dataset_base.samples)
        num_classes = len(train_dataset_base.unique_tasks)
        
        class_weights = []
        for task in train_dataset_base.unique_tasks:
            count = task_counts.get(task, 1)
            weight = total_samples / (num_classes * count)  # 역빈도
            class_weights.append(weight)
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"가중치 교차 엔트로피 손실 사용 (클래스 가중치: {class_weights.cpu().numpy()})")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer (최신 optimizer 지원)
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay if weight_decay > 0 else 0.01
        )
    elif optimizer_type == 'lion':
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay if weight_decay > 0 else 0.01
            )
        except ImportError:
            print("⚠ Lion optimizer not available, using AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay if weight_decay > 0 else 0.01
            )
    elif optimizer_type == 'adafactor':
        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                model.parameters(),
                lr=learning_rate,
                scale_parameter=False,
                relative_step=False
            )
        except ImportError:
            print("⚠ Adafactor optimizer not available, using AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay if weight_decay > 0 else 0.01
            )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay if weight_decay > 0 else 0.0
        )
    
    # Scheduler (최신 scheduler 지원)
    if scheduler_type == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, epochs=num_epochs,
            steps_per_epoch=len(train_loader), pct_start=0.3
        )
    else:
        # 기본: ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience
        )
    
    # 학습 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed Precision Training: 활성화")
    
    # MixUp / CutMix 설정
    if use_mixup:
        print(f"MixUp: 활성화 (alpha={mixup_alpha})")
    if use_cutmix:
        print(f"CutMix: 활성화 (alpha={cutmix_alpha})")
    
    def mixup_data(x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        y_a, y_b = y, y[index]
        
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam
    
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    # 학습 루프
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # MixUp / CutMix 적용
            if use_mixup and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            elif use_cutmix and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            
            optimizer.zero_grad()
            
            # Mixed Precision Training
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    if use_mixup or use_cutmix:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # OneCycle scheduler는 step per batch
            if scheduler_type == 'onecycle':
                scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        # Scheduler step (OneCycle은 이미 step됨)
        if scheduler_type != 'onecycle':
            if scheduler_type == 'cosine_warm_restarts':
                scheduler.step()
            else:
                scheduler.step(val_loss_avg)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}')
        
        # Best model 저장 (validation loss 기준)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss_avg,
                'num_classes': num_classes,
                'backbone': backbone,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f'  ✓ Best model saved (Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f})')
        else:
            epochs_without_improvement += 1
        
        # Early stopping (논문: patience=15)
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f'\nEarly stopping at epoch {epoch+1} (patience={early_stopping_patience})')
            break
    
    # Test 평가
    print("\n" + "="*70)
    print("Test 평가")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # XAI 분석 실행
    if run_xai:
        print("\n" + "="*70)
        print("XAI 분석 시작 (Grad-CAM)")
        print("="*70)
        
        xai_save_dir = os.path.join(save_dir, 'gradcam_analysis')
        analyze_with_gradcam(
            model=model,
            dataloader=test_loader,
            device=device,
            num_samples=20,
            save_dir=xai_save_dir,
        )
    
    # 실험용 메트릭 반환
    if return_metrics:
        return {
            'model': model,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'confusion_matrix': test_confusion,
            'best_val_acc': best_val_acc,
        }
    
    return model, test_acc


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_type', type=str, default='connectivity_matrix',
                       choices=['connectivity_matrix', 'connectivity_heatmap'])
    parser.add_argument('--value_type', type=str, default='absCPCC',
                       choices=['absCPCC', 'imCPCC'])
    parser.add_argument('--freq_band', type=str, default=None,
                       choices=[None, 'alpha', 'delta', 'gamma', 'high_beta', 'low_beta', 'theta'])
    parser.add_argument('--crop_top', type=int, default=100)
    parser.add_argument('--crop_bottom', type=int, default=100)
    parser.add_argument('--crop_left', type=int, default=100)
    parser.add_argument('--crop_right', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--run_xai', action='store_true', 
                       help='학습 후 Grad-CAM 분석 실행')
    parser.add_argument('--folder_list_txt', type=str, default=None,
                       help='완료된 이미지 폴더 리스트 txt 파일 경로')
    parser.add_argument('--multiview', action='store_true',
                       help='멀티뷰 학습: 6개 주파수 대역 이미지를 채널로 결합')
    parser.add_argument('--use_both_cpcc', action='store_true',
                       help='absCPCC와 imCPCC 모두 사용 (멀티뷰 모드에서만 유효)')
    
    # 데이터 증강 옵션
    parser.add_argument('--use_data_augmentation', action='store_true',
                       help='데이터 증강 사용 (ColorJitter, RandomAffine 등)')
    parser.add_argument('--aug_strength', type=str, default='medium',
                       choices=['light', 'medium', 'strong'],
                       help='데이터 증강 강도: light(가벼운 증강), medium(중간), strong(강한 증강)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader num_workers (기본값: 4)')
    
    args = parser.parse_args()
    
    train_model(
        image_type=args.image_type,
        value_type=args.value_type,
        freq_band=args.freq_band,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        crop_left=args.crop_left,
        crop_right=args.crop_right,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        backbone=args.backbone,
        pretrained=args.pretrained,
        run_xai=args.run_xai,
        folder_list_txt=args.folder_list_txt,
        multiview=args.multiview,
        use_both_cpcc=args.use_both_cpcc,
        use_data_augmentation=args.use_data_augmentation,
        aug_strength=args.aug_strength,
        num_workers=args.num_workers,
    )

