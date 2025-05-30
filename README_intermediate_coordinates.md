# Boltz-1 Intermediate Coordinate Extraction

이 기능은 Boltz-1 모델의 diffusion reverse process 동안 각 time step에서의 중간 구조 좌표를 추출할 수 있게 해줍니다. 이를 통해 구조 예측 과정에서 단백질이 어떻게 점진적으로 형성되는지 관찰할 수 있습니다.

## 주요 특징

- **Time Step별 좌표 저장**: 각 diffusion step에서 noisy, denoised, final 좌표를 저장
- **메타데이터 포함**: 샘플링 파라미터, sigma 값, timestep 정보 포함
- **메모리 효율적**: 필요시에만 중간 좌표를 저장하는 옵션 제공
- **기존 API 호환성**: 기본값 False로 설정하여 기존 기능에 영향 없음

## 수정된 코드 구성요소

### 1. AtomDiffusion.sample() - 핵심 구현부
```python
# src/boltz/model/modules/diffusion.py
def sample(
    self,
    atom_mask,
    num_sampling_steps=None,
    multiplicity=1,
    train_accumulate_token_repr=False,
    steering_args=None,
    save_intermediate_coords=False,  # 새로 추가된 매개변수
    **network_condition_kwargs,
):
```

### 2. Boltz1.forward() - 매개변수 전파
```python
# src/boltz/model/model.py
def forward(
    self,
    feats: dict[str, Tensor],
    recycling_steps: int = 0,
    num_sampling_steps: Optional[int] = None,
    multiplicity_diffusion_train: int = 1,
    diffusion_samples: int = 1,
    run_confidence_sequentially: bool = False,
    save_intermediate_coords: bool = False,  # 새로 추가된 매개변수
) -> dict[str, Tensor]:
```

### 3. Boltz1.predict_step() - 최종 예측 인터페이스
```python
# src/boltz/model/model.py
def predict_step(
    self, 
    batch: Any, 
    batch_idx: int, 
    dataloader_idx: int = 0, 
    save_intermediate_coords: bool = False  # 새로 추가된 매개변수
) -> Any:
```

## 사용 방법

### 1. 기본 사용법

```python
import torch
from boltz.model.model import Boltz1

# 모델 로드
model = Boltz1.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# 예측 실행 (중간 좌표 저장 활성화)
with torch.no_grad():
    result = model.predict_step(
        batch=batch,
        batch_idx=0,
        save_intermediate_coords=True  # 중간 좌표 저장 활성화
    )

# 중간 trajectory 확인
if "intermediate_trajectory" in result:
    trajectory = result["intermediate_trajectory"]
    print(f"저장된 timesteps: {len(trajectory['timesteps'])}")
```

### 2. Forward 메서드를 통한 직접 호출

```python
# 더 낮은 레벨에서 직접 제어
output = model(
    feats=batch,
    recycling_steps=3,
    num_sampling_steps=50,
    diffusion_samples=1,
    save_intermediate_coords=True
)

trajectory = output.get("intermediate_trajectory")
```

## 저장되는 데이터 구조

```python
intermediate_trajectory = {
    'timesteps': [],           # 각 단계의 timestep 값
    'sigmas': [],              # 각 단계의 noise level (sigma)
    'noisy_coords': [],        # 노이즈가 추가된 좌표 (x_t)
    'denoised_coords': [],     # 디노이즈된 좌표 (x_0 예측값)
    'final_coords': [],        # 다음 단계로 전달되는 좌표
    'metadata': {
        'num_sampling_steps': int,      # 총 샘플링 단계 수
        'multiplicity': int,            # 배치 multiplicity
        'init_sigma': float,           # 초기 노이즈 레벨
        'shape': tuple                 # 좌표 텐서 shape
    }
}
```

## 데이터 분석 예제

### 1. 구조 변화 분석

```python
def analyze_structural_evolution(trajectory):
    """구조 진화 과정 분석"""
    coords_list = trajectory['denoised_coords'][1:]  # 초기 노이즈 제외
    
    # 각 단계별 RMSD 계산
    rmsds = []
    for i in range(1, len(coords_list)):
        prev_coords = coords_list[i-1]
        curr_coords = coords_list[i]
        rmsd = torch.sqrt(((prev_coords - curr_coords) ** 2).mean())
        rmsds.append(rmsd.item())
    
    return rmsds

# 사용 예
rmsds = analyze_structural_evolution(trajectory)
print(f"평균 단계별 RMSD: {np.mean(rmsds):.3f} Å")
```

### 2. 특정 timestep 좌표 추출

```python
def extract_coordinates_at_timestep(trajectory, target_timestep):
    """특정 timestep의 좌표 추출"""
    try:
        idx = trajectory['timesteps'].index(target_timestep)
        return {
            'timestep': target_timestep,
            'sigma': trajectory['sigmas'][idx],
            'noisy': trajectory['noisy_coords'][idx],
            'denoised': trajectory['denoised_coords'][idx],
            'final': trajectory['final_coords'][idx]
        }
    except ValueError:
        print(f"Timestep {target_timestep} not found")
        return None

# 중간 지점 좌표 추출
mid_coords = extract_coordinates_at_timestep(trajectory, 25)
```

### 3. PDB 파일로 저장

```python
def save_trajectory_as_pdb_series(trajectory, output_dir, save_every=10):
    """Trajectory를 PDB 파일 시리즈로 저장"""
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, timestep in enumerate(trajectory['timesteps']):
        if i % save_every == 0 and trajectory['denoised_coords'][i] is not None:
            coords = trajectory['denoised_coords'][i][0]  # 첫 번째 sample
            
            pdb_file = output_dir / f"timestep_{timestep:03d}.pdb"
            with open(pdb_file, 'w') as f:
                f.write(f"REMARK Timestep {timestep}, Sigma {trajectory['sigmas'][i]:.4f}\n")
                for j, coord in enumerate(coords):
                    f.write(f"ATOM  {j+1:5d}  CA  ALA A{j+1:4d}    "
                           f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                           f"  1.00 20.00           C\n")
                f.write("END\n")

# 사용 예
save_trajectory_as_pdb_series(trajectory, "./trajectory_output", save_every=5)
```

## 메모리 사용량 고려사항

- **전체 trajectory 저장**: 메모리 사용량이 상당할 수 있음 (timesteps × batch_size × atoms × 3 × dtype_size)
- **권장 사항**: 
  - 작은 단백질부터 테스트
  - 필요시 `save_every` 옵션으로 저장 빈도 조절
  - GPU 메모리 모니터링 필수

## 활용 사례

1. **구조 예측 과정 시각화**: 각 timestep을 애니메이션으로 구성
2. **수렴성 분석**: 구조가 안정화되는 지점 파악
3. **에러 분석**: 예측이 실패하는 구간 식별
4. **모델 개선**: Diffusion process의 효율성 분석

## 문제 해결

### 메모리 부족 오류
```python
# 더 적은 timestep으로 테스트
result = model.predict_step(
    batch=small_batch,
    batch_idx=0,
    save_intermediate_coords=True
)

# 또는 sampling_steps 줄이기
model.predict_args["sampling_steps"] = 50  # 기본값 200에서 감소
```

### 빈 trajectory
```python
# save_intermediate_coords=True 설정 확인
# inference 모드에서만 동작 (training=False)
model.eval()
```

## 향후 개선 방향

- [ ] 선택적 timestep 저장 옵션
- [ ] 메모리 효율적인 저장 방식
- [ ] 자동 PDB/mmCIF 출력 기능
- [ ] 실시간 시각화 지원

이 기능을 통해 Boltz-1의 diffusion process를 더 깊이 이해하고 분석할 수 있습니다. 