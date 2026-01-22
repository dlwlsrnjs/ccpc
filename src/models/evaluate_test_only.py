#!/usr/bin/env python3
"""
Script to load model from checkpoint and evaluate on test set only
"""

import os
import sys
import torch
import argparse
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.task_classification_from_images_xai import (
    ConnectivityImageDataset,
    XAIClassificationModel,
    split_subjects_independently,
    parse_subject_task_from_dirname,
)


def load_checkpoint_info(checkpoint_path):
    """Extract model configuration information from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 구조 정보
    state_dict = checkpoint.get('model_state_dict', {})
    
    # 입력 채널 수 추정 (첫 번째 레이어의 입력 채널)
    if 'features.0.weight' in state_dict:
        input_channels = state_dict['features.0.weight'].shape[1]
    else:
        input_channels = 3  # 기본값
    
    # 저장된 정보
    num_classes = checkpoint.get('num_classes', 10)
    backbone = checkpoint.get('backbone', 'resnet18')
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    
    # 입력 채널로 multiview 설정 추정
    if input_channels == 36:
        multiview = True
        use_both_cpcc = True  # 6 bands * 2 types * 3 RGB = 36
        value_type = 'absCPCC'  # both 모드
    elif input_channels == 18:
        multiview = True
        use_both_cpcc = False  # 6 bands * 3 RGB = 18
        value_type = 'imCPCC'  # imCPCC만 사용
    else:
        multiview = False
        use_both_cpcc = False
        value_type = 'absCPCC'
    
    return {
        'input_channels': input_channels,
        'num_classes': num_classes,
        'backbone': backbone,
        'epoch': epoch,
        'val_acc': val_acc,
        'multiview': multiview,
        'use_both_cpcc': use_both_cpcc,
        'value_type': value_type,
        'state_dict': state_dict,
    }


def create_test_dataset(
    image_base_dir,
    folder_list_txt=None,
    value_type='absCPCC',
    multiview=True,
    use_both_cpcc=False,
    crop_top=100,
    crop_bottom=100,
    crop_left=100,
    crop_right=100,
    participants_file=None,
):
    """Create test dataset"""
    print("\n테스트 데이터셋 생성 중...")
    
    # 폴더 리스트 로드
    if folder_list_txt and os.path.exists(folder_list_txt):
        with open(folder_list_txt, 'r') as f:
            all_dirs = [line.strip() for line in f if line.strip()]
    else:
        # 모든 디렉토리 검색
        all_dirs = []
        for item in os.listdir(image_base_dir):
            item_path = os.path.join(image_base_dir, item)
            if os.path.isdir(item_path):
                all_dirs.append(item)
    
    print(f"  총 {len(all_dirs)}개 디렉토리")
    
    # Subject와 Task 추출
    subjects = []
    tasks = []
    image_dirs = []
    
    for dirname in all_dirs:
        try:
            subject_id, task_name = parse_subject_task_from_dirname(dirname)
            subjects.append(subject_id)
            tasks.append(task_name)
            image_dirs.append(os.path.join(image_base_dir, dirname))
        except:
            continue
    
    print(f"  총 {len(image_dirs)}개 샘플 로드 완료")
    
    # Subject 독립적 split
    train_indices, val_indices, test_indices = split_subjects_independently(
        subjects=subjects,
        tasks=tasks,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        participants_file=participants_file,
        stratify_by_age=True if participants_file else False,
    )
    
    print(f"  Test subjects: {len(set([subjects[i] for i in test_indices]))}명")
    print(f"  Test samples: {len(test_indices)}개")
    
    # 테스트 데이터만 사용
    test_dirs = [image_dirs[i] for i in test_indices]
    test_subjects = [subjects[i] for i in test_indices]
    test_tasks = [tasks[i] for i in test_indices]
    
    # 기본 transform (PIL Image를 Tensor로 변환, 크기 통일)
    from torchvision import transforms as T
    basic_transform = T.Compose([
        T.Resize((224, 224)),  # 모든 이미지를 같은 크기로
        T.ToTensor(),  # PIL Image -> Tensor
    ])
    
    # 데이터셋 생성
    dataset = ConnectivityImageDataset(
        image_dirs=test_dirs,
        subjects=test_subjects,
        tasks=test_tasks,
        image_type='connectivity_matrix',
        value_type=value_type,
        freq_band=None,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        transform=basic_transform,
        multiview=multiview,
        use_both_cpcc=use_both_cpcc,
    )
    
    print("✓ 테스트 데이터셋 생성 완료\n")
    
    return dataset, test_subjects, test_tasks


def evaluate_test(
    checkpoint_path,
    image_base_dir,
    folder_list_txt=None,
    device='cuda',
    batch_size=32,
    crop_top=100,
    crop_bottom=100,
    crop_left=100,
    crop_right=100,
    participants_file=None,
):
    """Load model from checkpoint and evaluate on test set"""
    
    print("="*80)
    print("테스트 평가 시작")
    print("="*80)
    
    # 1. 체크포인트 정보 로드
    print(f"\n체크포인트 로드: {checkpoint_path}")
    checkpoint_info = load_checkpoint_info(checkpoint_path)
    
    print(f"\n모델 설정:")
    print(f"  Backbone: {checkpoint_info['backbone']}")
    print(f"  입력 채널: {checkpoint_info['input_channels']}")
    print(f"  클래스 수: {checkpoint_info['num_classes']}")
    print(f"  Multiview: {checkpoint_info['multiview']}")
    print(f"  Use Both CPCC: {checkpoint_info['use_both_cpcc']}")
    print(f"  Value Type: {checkpoint_info['value_type']}")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Val Accuracy: {checkpoint_info['val_acc']:.4f}")
    
    # 2. 테스트 데이터셋 생성
    test_dataset, test_subjects, test_tasks = create_test_dataset(
        image_base_dir=image_base_dir,
        folder_list_txt=folder_list_txt,
        value_type=checkpoint_info['value_type'],
        multiview=checkpoint_info['multiview'],
        use_both_cpcc=checkpoint_info['use_both_cpcc'],
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        participants_file=participants_file,
    )
    
    # 3. 모델 생성 및 체크포인트 로드
    print("\n모델 생성 중...")
    model = XAIClassificationModel(
        backbone=checkpoint_info['backbone'],
        num_classes=checkpoint_info['num_classes'],
        pretrained=False,  # 체크포인트에서 로드하므로 False
        input_channels=checkpoint_info['input_channels'],
    ).to(device)
    
    # 체크포인트 로드
    print("체크포인트에서 가중치 로드 중...")
    model.load_state_dict(checkpoint_info['state_dict'], strict=True)
    model.eval()
    print("✓ 모델 로드 완료\n")
    
    # 4. DataLoader 생성
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 테스트만 실행하므로 0
        pin_memory=False,
    )
    
    # 5. 테스트 평가
    print("="*80)
    print("테스트 평가 실행")
    print("="*80)
    
    all_predictions = []
    all_labels = []
    all_tasks = []
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test 평가'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            tasks = batch['task']
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_tasks.extend(tasks)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    # 6. 결과 계산 및 출력
    test_acc = test_correct / test_total
    
    # Per-class metrics
    test_f1 = f1_score(all_labels, all_predictions, average='macro')
    test_precision = precision_score(all_labels, all_predictions, average='macro')
    test_recall = recall_score(all_labels, all_predictions, average='macro')
    test_confusion = confusion_matrix(all_labels, all_predictions)
    
    print(f"\n{'='*80}")
    print("테스트 평가 결과")
    print("="*80)
    print(f"전체 정확도: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"F1-Score (macro): {test_f1:.4f}")
    print(f"Precision (macro): {test_precision:.4f}")
    print(f"Recall (macro): {test_recall:.4f}")
    print(f"총 테스트 샘플: {test_total}개")
    
    # 태스크별 정확도
    task_to_label = {task: idx for idx, task in enumerate(sorted(set(test_tasks)))}
    task_correct = Counter()
    task_total = Counter()
    
    for i, task in enumerate(all_tasks):
        label = all_labels[i]
        pred = all_predictions[i]
        task_total[task] += 1
        if label == pred:
            task_correct[task] += 1
    
    print(f"\n태스크별 정확도:")
    for task in sorted(task_total.keys()):
        correct = task_correct[task]
        total = task_total[task]
        acc = (correct / total * 100) if total > 0 else 0
        print(f"  {task:30s}: {correct:3d}/{total:3d} = {acc:5.1f}%")
    
    print(f"\n{'='*80}")
    
    return {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': test_confusion,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='체크포인트에서 모델을 로드하여 테스트 평가만 실행')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='체크포인트 파일 경로')
    parser.add_argument('--image_base_dir', type=str,
                       default='/home/work/skku/startkit-main/images/connectivity',
                       help='이미지 기본 디렉토리')
    parser.add_argument('--folder_list_txt', type=str,
                       default='/home/work/skku/startkit-main/images/connectivity/completed_image_folders.txt',
                       help='완료된 이미지 폴더 리스트 txt 파일 경로')
    parser.add_argument('--participants_file', type=str,
                       default='/home/work/skku/startkit-main/hbn_eeg_releases/participants.tsv',
                       help='Participants 파일 경로')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--crop_top', type=int, default=100,
                       help='상단 크롭 크기')
    parser.add_argument('--crop_bottom', type=int, default=100,
                       help='하단 크롭 크기')
    parser.add_argument('--crop_left', type=int, default=100,
                       help='좌측 크롭 크기')
    parser.add_argument('--crop_right', type=int, default=100,
                       help='우측 크롭 크기')
    
    args = parser.parse_args()
    
    results = evaluate_test(
        checkpoint_path=args.checkpoint,
        image_base_dir=args.image_base_dir,
        folder_list_txt=args.folder_list_txt,
        device=args.device,
        batch_size=args.batch_size,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        crop_left=args.crop_left,
        crop_right=args.crop_right,
        participants_file=args.participants_file if os.path.exists(args.participants_file) else None,
    )
    
    print("\n✓ 테스트 평가 완료")
