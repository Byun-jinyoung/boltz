# Boltz-1 중간 좌표 추출 사용 가이드

## 개요

이 가이드는 Boltz-1 모델의 diffusion reverse process에서 중간 좌표를 추출하고 분석하는 방법을 설명합니다.

## 설치 및 설정

### 필수 요구사항
- PyTorch
- Boltz-1 모델 체크포인트
- 충분한 GPU 메모리 (중간 좌표 저장 시)

### 환경 설정
```bash
# 캐시 디렉토리 설정 (선택사항)
export BOLTZ_CACHE=~/.boltz

# GPU 메모리 최적화를 위한 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 사용 방법

### 1. 기본 CLI 사용법

```bash
# 기본 예측 (중간 좌표 저장 없음)
python -m boltz.main predict input.yaml

# 중간 좌표 저장 활성화
python -m boltz.main predict input.yaml --save_intermediate_coords

# 추가 옵션과 함께 사용
python -m boltz.main predict input.yaml \
    --save_intermediate_coords \
    --intermediate_output_format both \
    --intermediate_save_every 5 \
    --sampling_steps 50 \
    --out_dir ./my_results
```

### 2. 완전한 스크립트 사용법

```bash
# 기본 사용
python run_intermediate_coords_extraction.py input.yaml --save-intermediate-coords

# 모든 옵션 포함
python run_intermediate_coords_extraction.py input.yaml \
    --save-intermediate-coords \
    --intermediate-format both \
    --save-every 5 \
    --sampling-steps 50 \
    --output-dir ./trajectory_analysis \
    --create-animation \
    --verbose
```

## 옵션 설명

### 중간 좌표 관련 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--save_intermediate_coords` | 중간 좌표 저장 활성화 | False |
| `--intermediate_output_format` | 출력 형식 (pdb/npz/both) | pdb |
| `--intermediate_save_every` | N 스텝마다 저장 | 10 |

### 성능 관련 옵션

| 옵션 | 설명 | 기본값 | 권장값 |
|------|------|--------|--------|
| `--sampling_steps` | 샘플링 스텝 수 | 200 | 50-100 (중간 좌표용) |
| `--diffusion_samples` | 샘플 수 | 1 | 1 |
| `--devices` | GPU 개수 | 1 | 1 |

## 출력 구조

```
output_directory/
├── boltz_results_input_name/
│   ├── predictions/
│   │   ├── structure_id/
│   │   │   ├── structure_id_model_0.pdb
│   │   │   ├── confidence_structure_id_model_0.json
│   │   │   └── ...
│   │   └── trajectories/              # 새로 추가됨
│   │       └── structure_id/
│   │           └── model_0/
│   │               ├── trajectory_metadata.json
│   │               ├── trajectory_analysis.json
│   │               ├── trajectory_data.npz     # npz 형식인 경우
│   │               ├── timestep_000_sigma_*.pdb
│   │               ├── timestep_010_sigma_*.pdb
│   │               └── animate_trajectory.pml  # 애니메이션 스크립트
│   ├── processed/
│   └── lightning_logs/
└── trajectory_summary.json            # 전체 요약
```

## 출력 데이터 형식

### 1. 메타데이터 (trajectory_metadata.json)
```json
{
  "num_sampling_steps": 50,
  "multiplicity": 1,
  "init_sigma": 160.0,
  "shape": [1, 128, 3]
}
```

### 2. 분석 결과 (trajectory_analysis.json)
```json
{
  "record_id": "structure_id",
  "model_idx": 0,
  "num_timesteps": 51,
  "initial_sigma": 160.0,
  "final_sigma": 0.0,
  "mean_step_rmsd": 2.145,
  "std_step_rmsd": 0.823,
  "overall_rmsd": 15.234
}
```

### 3. NPZ 데이터 형식
```python
# NPZ 파일 로드 및 사용
import numpy as np

data = np.load("trajectory_data.npz")
timesteps = data['selected_timesteps']      # [N] - 저장된 타임스텝
sigmas = data['selected_sigmas']           # [N] - 각 타임스텝의 시그마값
noisy_coords = data['noisy_coords']        # [N, B, atoms, 3] - 노이즈 좌표
denoised_coords = data['denoised_coords']  # [N, B, atoms, 3] - 디노이즈 좌표
final_coords = data['final_coords']        # [N, B, atoms, 3] - 최종 좌표
```

## 분석 및 시각화

### 1. Python에서 데이터 분석

```python
import numpy as np
import matplotlib.pyplot as plt
import json

# 분석 데이터 로드
with open("trajectory_analysis.json") as f:
    analysis = json.load(f)

# NPZ 데이터 로드
trajectory_data = np.load("trajectory_data.npz")
timesteps = trajectory_data['selected_timesteps']
denoised_coords = trajectory_data['denoised_coords']

# RMSD 계산 및 플롯
rmsds = []
for i in range(1, len(denoised_coords)):
    prev = denoised_coords[i-1, 0]  # 첫 번째 샘플
    curr = denoised_coords[i, 0]
    rmsd = np.sqrt(np.mean((prev - curr)**2))
    rmsds.append(rmsd)

plt.figure(figsize=(10, 6))
plt.plot(timesteps[1:], rmsds, 'b-', linewidth=2)
plt.xlabel('Timestep')
plt.ylabel('RMSD (Å)')
plt.title('Structural Change During Diffusion Reverse Process')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. PyMOL 애니메이션

```bash
# PyMOL에서 애니메이션 스크립트 실행
pymol animate_trajectory.pml

# 또는 PyMOL 내에서
# load animate_trajectory.pml
```

### 3. 구조 진화 분석

```python
def analyze_structural_evolution(trajectory_dir):
    """구조 진화 패턴 분석"""
    import glob
    
    pdb_files = sorted(glob.glob(f"{trajectory_dir}/timestep_*.pdb"))
    
    # 각 타임스텝별 구조적 특성 분석
    properties = {
        'radius_of_gyration': [],
        'end_to_end_distance': [],
        'compactness': []
    }
    
    for pdb_file in pdb_files:
        coords = load_pdb_coords(pdb_file)  # 구현 필요
        
        # 회전반경 계산
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
        properties['radius_of_gyration'].append(rg)
        
        # 말단간 거리 (첫 번째와 마지막 원자)
        if len(coords) > 1:
            end_to_end = np.linalg.norm(coords[0] - coords[-1])
            properties['end_to_end_distance'].append(end_to_end)
    
    return properties
```

## 성능 최적화

### 1. 메모리 사용량 줄이기

```bash
# 저장 빈도 줄이기
--intermediate_save_every 20

# 샘플링 스텝 줄이기
--sampling_steps 50

# NPZ 형식만 사용 (PDB보다 효율적)
--intermediate_output_format npz
```

### 2. 속도 최적화

```bash
# 적은 리사이클링 스텝
--recycling_steps 1

# CPU 사용 (GPU 메모리 부족 시)
--accelerator cpu
```

## 문제 해결

### 1. 메모리 부족 오류

```bash
# 해결책 1: 저장 빈도 줄이기
--intermediate_save_every 50

# 해결책 2: 샘플링 스텝 줄이기
--sampling_steps 20

# 해결책 3: CPU 사용
--accelerator cpu
```

### 2. 빈 trajectory 결과

- `--save_intermediate_coords` 플래그 확인
- 모델이 inference 모드인지 확인
- 충분한 디스크 공간 확보

### 3. 애니메이션 생성 실패

- PDB 형식 출력 확인
- PyMOL 설치 확인
- 파일 권한 확인

## 실제 사용 예제

### 1. 작은 단백질 분석

```bash
# 빠른 테스트용 (메모리 효율적)
python run_intermediate_coords_extraction.py small_protein.yaml \
    --save-intermediate-coords \
    --sampling-steps 30 \
    --save-every 5 \
    --intermediate-format pdb \
    --create-animation
```

### 2. 큰 단백질 복합체 분석

```bash
# 메모리 절약 모드
python run_intermediate_coords_extraction.py large_complex.yaml \
    --save-intermediate-coords \
    --sampling-steps 50 \
    --save-every 20 \
    --intermediate-format npz \
    --device gpu
```

### 3. 상세 분석용

```bash
# 모든 중간 과정 저장
python run_intermediate_coords_extraction.py input.yaml \
    --save-intermediate-coords \
    --sampling-steps 100 \
    --save-every 1 \
    --intermediate-format both \
    --create-animation \
    --verbose
```

## 고급 분석 스크립트

### 1. 배치 분석

```python
#!/usr/bin/env python3
"""
여러 구조의 trajectory를 배치로 분석하는 스크립트
"""

import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

def batch_analyze_trajectories(base_dir):
    """여러 trajectory를 한번에 분석"""
    results = []
    
    trajectory_dirs = glob.glob(f"{base_dir}/*/predictions/trajectories/*/*")
    
    for traj_dir in trajectory_dirs:
        analysis_file = Path(traj_dir) / "trajectory_analysis.json"
        if analysis_file.exists():
            with open(analysis_file) as f:
                analysis = json.load(f)
            results.append(analysis)
    
    # DataFrame으로 변환하여 분석
    df = pd.DataFrame(results)
    
    # 통계 요약
    summary = {
        'total_structures': len(df),
        'mean_overall_rmsd': df['overall_rmsd'].mean(),
        'std_overall_rmsd': df['overall_rmsd'].std(),
        'mean_convergence_steps': df['num_timesteps'].mean(),
    }
    
    return df, summary

# 사용 예
df, summary = batch_analyze_trajectories("./results")
print(f"Analyzed {summary['total_structures']} structures")
print(f"Average RMSD: {summary['mean_overall_rmsd']:.3f} ± {summary['std_overall_rmsd']:.3f} Å")
```

이 가이드를 통해 Boltz-1 모델의 diffusion process를 자세히 분석하고 구조 예측 과정을 이해할 수 있습니다. 