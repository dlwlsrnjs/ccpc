#!/usr/bin/env python3
"""
Script to generate connectivity images from CSV files (Improved version)
Optimized for CNN training connectivity matrix image generation

Key improvements:
- Filter NaN values and invalid freq_band
- Enhanced error handling
- Generate only images needed for CNN training (remove unnecessary visualizations)
- Improved memory efficiency
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행 가능하도록
import matplotlib.pyplot as plt
import warnings
import glob
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50

# 데이터 경로 설정
DATA_DIR = '/home/work/skku/startkit-main/data/processed/connectivity/multiband/data'
OUTPUT_BASE_DIR = '/home/work/skku/startkit-main/images/connectivity'

# 유효한 주파수 대역 (논문에서 사용)
VALID_FREQ_BANDS = ['delta', 'theta', 'alpha', 'low_beta', 'high_beta', 'gamma']


def create_connectivity_matrix_from_df(df, freq_band=None, value_col='absCPCC'):
    """
    Create connectivity matrix from dataframe (absCPCC or imCPCC)
    
    Args:
        df: connectivity dataframe
        freq_band: frequency band (None for all)
        value_col: column to use ('absCPCC' or 'imCPCC')
    
    Returns:
        numpy.ndarray: connectivity matrix (n_channels x n_channels)
    """
    if len(df) == 0:
        return None
    
    # 주파수 필터링
    if freq_band is not None:
        df = df[df['freq_band'] == freq_band]
        if len(df) == 0:
            return None
    
    # 채널 수 (NaN 체크 추가)
    try:
        n_channels = int(df['num_channels'].iloc[0])
        if n_channels <= 0 or np.isnan(n_channels):
            return None
    except (ValueError, TypeError):
        return None
    
    # 빈 matrix 생성
    conn_matrix = np.zeros((n_channels, n_channels))
    count_matrix = np.zeros((n_channels, n_channels))  # 평균 계산용
    
    # Matrix 채우기 (모든 window의 평균)
    for _, row in df.iterrows():
        try:
            i = int(row['channel_i_idx'])
            j = int(row['channel_j_idx'])
            value = float(row[value_col])
            
            # 인덱스 범위 체크
            if i < 0 or i >= n_channels or j < 0 or j >= n_channels:
                continue
            
            # NaN 체크
            if np.isnan(value):
                continue
                
            conn_matrix[i, j] += value
            conn_matrix[j, i] += value
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1
        except (ValueError, TypeError, IndexError):
            continue
    
    # 평균 계산
    conn_matrix = np.divide(conn_matrix, count_matrix,
                           out=np.zeros_like(conn_matrix),
                           where=count_matrix!=0)
    
    # 대각선을 1.0으로 설정 (absCPCC) 또는 0.0으로 설정 (imCPCC)
    if value_col == 'absCPCC':
        np.fill_diagonal(conn_matrix, 1.0)
    elif value_col == 'imCPCC':
        np.fill_diagonal(conn_matrix, 0.0)
    
    return conn_matrix


def save_connectivity_matrix_image(conn_matrix, output_path, title=None, value_col='absCPCC'):
    """
    Connectivity matrix를 이미지로 저장 (CNN 학습용 최적화)
    
    Args:
        conn_matrix: connectivity matrix
        output_path: 저장 경로
        title: 이미지 제목
        value_col: 값 타입
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 색상맵 및 범위 설정
        vmax = conn_matrix.max()
        vmin = 0 if value_col == 'absCPCC' else conn_matrix.min()
        
        im = ax.imshow(conn_matrix, cmap='RdYlBu_r', aspect='auto', 
                      vmin=vmin, vmax=vmax, interpolation='nearest')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Channel Index', fontsize=12)
        ax.set_ylabel('Channel Index', fontsize=12)
        
        cbar_label = '|CPCC|' if value_col == 'absCPCC' else '|Im(CPCC)|'
        plt.colorbar(im, ax=ax, label=cbar_label)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    경고: 이미지 저장 실패 ({output_path}): {str(e)}")
        plt.close('all')


def validate_and_clean_dataframe(df):
    """
    데이터프레임 검증 및 정제
    
    Args:
        df: 원본 데이터프레임
    
    Returns:
        tuple: (cleaned_df, is_valid, error_message)
    """
    # 필수 컬럼 확인
    required_cols = ['absCPCC', 'freq_band', 'window_idx', 
                     'channel_i_idx', 'channel_j_idx', 'num_channels']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None, False, f"필요한 컬럼이 없습니다: {missing_cols}"
    
    # 데이터 확인
    if len(df) == 0:
        return None, False, "데이터가 비어있습니다"
    
    # NaN 값 제거
    df_clean = df.dropna(subset=required_cols)
    
    if len(df_clean) == 0:
        return None, False, "NaN 제거 후 데이터가 비어있습니다"
    
    # freq_band 필터링: 유효한 주파수 대역만 남김
    df_clean = df_clean[df_clean['freq_band'].isin(VALID_FREQ_BANDS)]
    
    if len(df_clean) == 0:
        return None, False, f"유효한 주파수 대역이 없습니다 (유효: {VALID_FREQ_BANDS})"
    
    # imCPCC 컬럼 확인
    has_imCPCC = 'imCPCC' in df.columns and not df['imCPCC'].isna().all()
    if has_imCPCC:
        df_clean = df_clean.dropna(subset=['imCPCC'])
    
    return df_clean, True, None


def process_single_csv(csv_path, essential_only=False):
    """
    단일 CSV 파일을 처리하여 connectivity 이미지를 생성
    
    Args:
        csv_path: CSV 파일 경로
        essential_only: True이면 CNN 학습에 필수적인 이미지만 생성
    """
    csv_name = Path(csv_path).stem
    output_dir = os.path.join(OUTPUT_BASE_DIR, csv_name)
    
    try:
        # CSV 파일 읽기 (에러 핸들링 개선)
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
        except Exception as e:
            print(f"\n✗ {csv_name}: CSV 파일 읽기 실패 - {str(e)}")
            return False
        
        # 데이터 검증 및 정제
        df_clean, is_valid, error_msg = validate_and_clean_dataframe(df)
        
        if not is_valid:
            print(f"\n✗ {csv_name}: {error_msg}")
            return False
        
        # 주파수 대역과 imCPCC 여부 확인
        freq_bands = sorted(df_clean['freq_band'].unique())
        has_imCPCC = 'imCPCC' in df_clean.columns and not df_clean['imCPCC'].isna().all()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 필수 이미지만 생성 (CNN 학습용)
        images_generated = 0
        
        for freq_band in freq_bands:
            # absCPCC connectivity matrix (필수)
            conn_matrix_abs = create_connectivity_matrix_from_df(
                df_clean, freq_band=freq_band, value_col='absCPCC'
            )
            
            if conn_matrix_abs is not None:
                output_path = os.path.join(
                    output_dir, 
                    f'connectivity_matrix_absCPCC_{freq_band}.png'
                )
                save_connectivity_matrix_image(
                    conn_matrix_abs, output_path,
                    title=f'{csv_name} - absCPCC - {freq_band}',
                    value_col='absCPCC'
                )
                images_generated += 1
            
            # imCPCC connectivity matrix (있는 경우)
            if has_imCPCC:
                conn_matrix_im = create_connectivity_matrix_from_df(
                    df_clean, freq_band=freq_band, value_col='imCPCC'
                )
                
                if conn_matrix_im is not None:
                    output_path = os.path.join(
                        output_dir,
                        f'connectivity_matrix_imCPCC_{freq_band}.png'
                    )
                    save_connectivity_matrix_image(
                        conn_matrix_im, output_path,
                        title=f'{csv_name} - imCPCC - {freq_band}',
                        value_col='imCPCC'
                    )
                    images_generated += 1
        
        if images_generated > 0:
            print(f"✓ {csv_name}: {images_generated}개 이미지 생성 완료")
            return True
        else:
            print(f"✗ {csv_name}: 이미지 생성 실패")
            return False
        
    except Exception as e:
        print(f"\n✗ {csv_name}: 처리 중 에러 발생 - {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 메모리 정리
        plt.close('all')


def check_images_exist(csv_path, output_base_dir, freq_bands=None, has_imCPCC=False):
    """
    해당 CSV 파일의 이미지가 이미 생성되었는지 확인
    
    Args:
        csv_path: CSV 파일 경로
        output_base_dir: 출력 베이스 디렉토리
        freq_bands: 주파수 대역 리스트 (None이면 기본값 사용)
        has_imCPCC: imCPCC 이미지 포함 여부
    
    Returns:
        bool: 이미지가 완전히 생성되었으면 True
    """
    csv_name = Path(csv_path).stem
    output_dir = os.path.join(output_base_dir, csv_name)
    
    if not os.path.exists(output_dir):
        return False
    
    # 기본 주파수 대역 사용
    if freq_bands is None:
        freq_bands = VALID_FREQ_BANDS
    
    # 필수 이미지 개수 계산
    expected_count = len(freq_bands)  # absCPCC
    if has_imCPCC:
        expected_count += len(freq_bands)  # imCPCC
    
    # 실제 생성된 이미지 확인
    existing_images = list(Path(output_dir).glob('connectivity_matrix_*.png'))
    
    # 최소한 필요한 개수만큼 있으면 완료로 간주
    return len(existing_images) >= expected_count


def main():
    """메인 함수"""
    print("=" * 80)
    print("Connectivity 이미지 생성 스크립트 (CNN 학습 최적화 버전)")
    print("=" * 80)
    print(f"데이터 디렉토리: {DATA_DIR}")
    print(f"출력 디렉토리: {OUTPUT_BASE_DIR}")
    print(f"유효한 주파수 대역: {VALID_FREQ_BANDS}")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 모든 CSV 파일 찾기
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_connectivity.csv')))
    print(f"\n발견된 CSV 파일 수: {len(csv_files)}")
    
    # 빈 파일 및 이미 처리된 파일 필터링
    valid_csv_files = []
    skipped_empty = 0
    skipped_complete = 0
    
    print("\n파일 검사 중...")
    for csv_file in tqdm(csv_files, desc="파일 검사"):
        # 빈 파일 체크
        if os.path.getsize(csv_file) == 0:
            skipped_empty += 1
            continue
        
        # 이미 완료된 파일 체크
        if check_images_exist(csv_file, OUTPUT_BASE_DIR):
            skipped_complete += 1
            continue
        
        valid_csv_files.append(csv_file)
    
    print(f"\n파일 필터링 결과:")
    print(f"  - 빈 파일: {skipped_empty}개")
    print(f"  - 이미 처리 완료: {skipped_complete}개")
    print(f"  - 처리 필요: {len(valid_csv_files)}개")
    print("=" * 80)
    
    if len(valid_csv_files) == 0:
        print("\n✅ 모든 이미지가 이미 생성되었습니다!")
        return
    
    # 병렬 처리 설정
    num_workers = min(40, cpu_count())
    print(f"\n병렬 처리: {num_workers}개 worker 사용")
    print("=" * 80)
    
    # 병렬 처리로 각 CSV 파일 처리
    success_count = 0
    fail_count = 0
    
    with Pool(processes=num_workers) as pool:
        results = []
        for csv_file in valid_csv_files:
            result = pool.apply_async(process_single_csv, (csv_file, True))
            results.append((csv_file, result))
        
        # 진행 상황 표시
        with tqdm(total=len(results), desc="이미지 생성", unit="파일") as pbar:
            for csv_file, result in results:
                try:
                    success = result.get(timeout=600)  # 10분 타임아웃
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    csv_name = Path(csv_file).stem
                    print(f"\n⚠ {csv_name}: 타임아웃 또는 에러 - {str(e)}")
                pbar.update(1)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("이미지 생성 완료!")
    print("=" * 80)
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"출력 디렉토리: {OUTPUT_BASE_DIR}")
    print("=" * 80)
    
    # CNN 학습용 폴더 리스트 생성
    completed_folders_path = os.path.join(OUTPUT_BASE_DIR, 'completed_folders.txt')
    completed_folders = []
    
    for dirname in os.listdir(OUTPUT_BASE_DIR):
        dir_path = os.path.join(OUTPUT_BASE_DIR, dirname)
        if os.path.isdir(dir_path):
            images = list(Path(dir_path).glob('connectivity_matrix_*.png'))
            if len(images) >= 6:  # 최소 6개 주파수 대역
                completed_folders.append(dirname)
    
    with open(completed_folders_path, 'w') as f:
        f.write('\n'.join(sorted(completed_folders)))
    
    print(f"\n✓ CNN 학습용 폴더 리스트 생성: {completed_folders_path}")
    print(f"  총 {len(completed_folders)}개 폴더 (주파수 대역별 이미지 포함)")
    print("\n다음 단계: CNN 학습")
    print("  python src/models/advanced_baseline_comparison.py \\")
    print(f"    --folder_list_txt {completed_folders_path} \\")
    print("    --multiview --use_both_cpcc")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Connectivity 이미지 생성 (CNN 학습 최적화)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                       help='CSV 파일 디렉토리')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_DIR,
                       help='이미지 출력 디렉토리')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='병렬 처리 워커 수 (기본: CPU 코어 수)')
    
    args = parser.parse_args()
    
    # 전역 변수 업데이트
    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.output_dir:
        OUTPUT_BASE_DIR = args.output_dir
    
    main()