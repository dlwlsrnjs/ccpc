#!/usr/bin/env python3
"""
위상 행렬 이미지로 태스크 분류 모델 학습 스크립트 (XAI 지원)

사용 예시:
    # 기본 학습
    python train_task_classification_images_xai.py

    # 학습 후 Grad-CAM 분석 실행
    python train_task_classification_images_xai.py --run_xai

    # ResNet50 사용
    python train_task_classification_images_xai.py --backbone resnet50

    # 특정 주파수 대역만 사용
    python train_task_classification_images_xai.py --freq_band alpha
"""

import argparse
from task_classification_from_images_xai import train_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='위상 행렬 이미지로 태스크 분류 모델 학습 (XAI 지원)')
    
    parser.add_argument('--image_type', type=str, default='connectivity_matrix',
                       choices=['connectivity_matrix', 'connectivity_heatmap'],
                       help='사용할 이미지 타입')
    parser.add_argument('--value_type', type=str, default='absCPCC',
                       choices=['absCPCC', 'imCPCC'],
                       help='absCPCC 또는 imCPCC')
    parser.add_argument('--freq_band', type=str, default=None,
                       choices=[None, 'alpha', 'delta', 'gamma', 'high_beta', 'low_beta', 'theta'],
                       help='특정 주파수 대역만 사용 (None이면 모든 주파수 대역)')
    
    # Crop 파라미터
    parser.add_argument('--crop_top', type=int, default=100, 
                       help='상단 텍스트 영역 제거 크기 (픽셀)')
    parser.add_argument('--crop_bottom', type=int, default=100, 
                       help='하단 텍스트 영역 제거 크기 (픽셀)')
    parser.add_argument('--crop_left', type=int, default=100, 
                       help='좌측 텍스트 영역 제거 크기 (픽셀)')
    parser.add_argument('--crop_right', type=int, default=100, 
                       help='우측 텍스트 영역 제거 크기 (픽셀)')
    
    # 모델 파라미터
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='사용할 backbone 모델 (XAI에 적합)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Pre-trained weights 사용')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                       help='Pre-trained weights 사용 안 함')
    
    # 학습 파라미터
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    
    # XAI 파라미터
    parser.add_argument('--run_xai', action='store_true',
                       help='학습 후 Grad-CAM 분석 자동 실행')
    
    # 폴더 리스트 파일
    parser.add_argument('--folder_list_txt', type=str, 
                       default='/home/work/skku/startkit-main/images/connectivity/completed_image_folders.txt',
                       help='완료된 이미지 폴더 리스트 txt 파일 경로 (None이면 모든 폴더 사용)')
    
    # 멀티뷰 학습
    parser.add_argument('--multiview', action='store_true',
                       help='멀티뷰 학습 활성화: 6개 주파수 대역 이미지를 채널로 결합하여 입력')
    parser.add_argument('--use_both_cpcc', action='store_true',
                       help='absCPCC와 imCPCC 모두 사용 (멀티뷰 모드에서만 유효, 36채널 입력)')
    
    # 데이터 증강 옵션
    parser.add_argument('--use_data_augmentation', action='store_true',
                       help='데이터 증강 사용 (ColorJitter, RandomAffine, RandomErasing 등)')
    parser.add_argument('--aug_strength', type=str, default='medium',
                       choices=['light', 'medium', 'strong'],
                       help='데이터 증강 강도: light(가벼운 증강), medium(중간), strong(강한 증강)')
    
    # Advanced Augmentation
    parser.add_argument('--use_mixup', action='store_true', help='MixUp 사용')
    parser.add_argument('--use_cutmix', action='store_true', help='CutMix 사용')
    parser.add_argument('--use_autoaugment', action='store_true', help='AutoAugment 사용')
    parser.add_argument('--use_randaugment', action='store_true', help='RandAugment 사용')
    parser.add_argument('--use_gridmask', action='store_true', help='GridMask 사용')
    parser.add_argument('--use_freq_masking', action='store_true', help='주파수 대역 masking 사용')
    
    # Optimizer & Scheduler
    parser.add_argument('--optimizer_type', type=str, default='adamw',
                       choices=['adam', 'adamw', 'lion', 'adafactor'],
                       help='Optimizer 타입')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                       choices=['plateau', 'cosine_warm_restarts', 'onecycle'],
                       help='Learning rate scheduler 타입')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Mixed Precision Training 사용')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader num_workers (기본값: 4, shared memory 고려)')
    
    # 저장 경로
    parser.add_argument('--save_dir', type=str, 
                       default='./checkpoints/task_classification_images_xai_advanced')
    
    args = parser.parse_args()
    
    print("="*70)
    print("위상 행렬 이미지 태스크 분류 모델 학습 (XAI 지원)")
    print("="*70)
    print(f"이미지 타입: {args.image_type}")
    print(f"값 타입: {args.value_type}")
    print(f"주파수 대역: {args.freq_band if args.freq_band else '모든 주파수 대역'}")
    print(f"Backbone: {args.backbone} (pretrained: {args.pretrained})")
    print(f"Crop 설정: top={args.crop_top}, bottom={args.crop_bottom}, left={args.crop_left}, right={args.crop_right}")
    print(f"멀티뷰 학습: {'활성화' if args.multiview else '비활성화 (단일 이미지)'}")
    if args.multiview:
        if args.use_both_cpcc:
            print(f"  → 6주파수 × 2타입(abs+im) = 12개 이미지 결합 (36채널)")
        else:
            print(f"  → 6주파수 대역 결합 (18채널)")
    print(f"데이터 증강: {'활성화' if args.use_data_augmentation else '비활성화'}")
    if args.use_data_augmentation:
        print(f"  → 증강 강도: {args.aug_strength}")
    print(f"XAI 분석: {'실행' if args.run_xai else '미실행'}")
    print("="*70)
    
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
        save_dir=args.save_dir,
        backbone=args.backbone,
        pretrained=args.pretrained,
        run_xai=args.run_xai,
        folder_list_txt=args.folder_list_txt,
        multiview=args.multiview,
        use_both_cpcc=args.use_both_cpcc,
        use_data_augmentation=args.use_data_augmentation,
        aug_strength=args.aug_strength,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        use_autoaugment=args.use_autoaugment,
        use_randaugment=args.use_randaugment,
        use_gridmask=args.use_gridmask,
        use_freq_masking=args.use_freq_masking,
        optimizer_type=args.optimizer_type,
        scheduler_type=args.scheduler_type,
        use_amp=args.use_amp,
        num_workers=args.num_workers,
    )

