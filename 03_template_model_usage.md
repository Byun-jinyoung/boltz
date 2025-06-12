# Boltz-2 모델에서의 Template 정보 활용 상세 분석

## 개요

Boltz-2에서 Template 정보는 `TemplateModule`과 `TemplateV2Module`을 통해 신경망 모델에 통합됩니다. 이 문서에서는 추출된 template feature들이 모델에서 어떻게 처리되고 활용되는지를 코드와 연결하여 상세히 분석합니다.

## 1. TemplateModule 구조 및 초기화

### 1.1 모듈 파라미터
```python
class TemplateModule(nn.Module):
    def __init__(
        self,
        token_z: int,              # Pairwise embedding dimension
        template_dim: int,         # Template processing dimension  
        template_blocks: int,      # Number of Pairformer blocks
        min_dist: float = 3.25,    # Minimum distance for distogram
        max_dist: float = 50.75,   # Maximum distance for distogram
        num_bins: int = 38,        # Number of distance bins
        **kwargs,
    ):
```

### 1.2 핵심 구성요소
```python
# Layer normalization and projections
self.z_norm = nn.LayerNorm(token_z)
self.v_norm = nn.LayerNorm(template_dim)
self.z_proj = nn.Linear(token_z, template_dim, bias=False)
self.u_proj = nn.Linear(template_dim, token_z, bias=False)

# Template feature projection
self.a_proj = nn.Linear(
    const.num_tokens * 2 + num_bins + 5,  # Input feature dimension
    template_dim,                          # Output dimension
    bias=False,
)

# Pairformer for template processing
self.pairformer = PairformerNoSeqModule(
    template_dim,
    num_blocks=template_blocks,
    # ... other parameters
)
```

## 2. Template Feature 처리 과정

### 2.1 입력 Feature 로딩
```python
def forward(self, z: Tensor, feats: dict[str, Tensor], pair_mask: Tensor) -> Tensor:
    # Load template features from feature dictionary
    asym_id = feats["asym_id"]                    # Chain asymmetric IDs
    res_type = feats["template_restype"]          # Shape: (B, T, N, 32)
    frame_rot = feats["template_frame_rot"]       # Shape: (B, T, N, 3, 3)
    frame_t = feats["template_frame_t"]           # Shape: (B, T, N, 3)
    frame_mask = feats["template_mask_frame"]     # Shape: (B, T, N)
    cb_coords = feats["template_cb"]              # Shape: (B, T, N, 3)
    ca_coords = feats["template_ca"]              # Shape: (B, T, N, 3)
    cb_mask = feats["template_mask_cb"]           # Shape: (B, T, N)
    template_mask = feats["template_mask"]        # Shape: (B, T, N)
```

**차원 설명**:
- `B`: Batch size
- `T`: Template 개수
- `N`: Token 개수 (max_tokens)

### 2.2 Template 수 계산 및 마스크 생성
```python
# Compute number of valid templates per batch
template_mask = feats["template_mask"].any(dim=2).float()  # (B, T)
num_templates = template_mask.sum(dim=1)                   # (B,)
num_templates = num_templates.clamp(min=1)                 # Prevent division by zero

# Create pairwise masks for CB and frame validity
b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]        # (B, T, N, N)
b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :] # (B, T, N, N)
```

### 2.3 Chain Asymmetric Mask 계산
```python
# Template features only attend within the same chain
B, T = res_type.shape[:2]
asym_mask = (asym_id[:, :, None] == asym_id[:, None, :]).float()  # (B, N, N)
asym_mask = asym_mask[:, None].expand(-1, T, -1, -1)             # (B, T, N, N)
```

**목적**: Multi-chain 복합체에서 서로 다른 체인 간에는 template 정보가 전달되지 않도록 차단

## 3. Template Feature 생성

### 3.1 Distogram 계산
```python
with torch.autocast(device_type="cuda", enabled=False):
    # Compute CB-CB distance matrix
    cb_dists = torch.cdist(cb_coords, cb_coords)  # (B, T, N, N)
    
    # Create distance boundaries and bins
    boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
    boundaries = boundaries.to(cb_dists.device)
    
    # Convert distances to histogram bins
    distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()  # (B, T, N, N)
    distogram = one_hot(distogram, num_classes=self.num_bins)          # (B, T, N, N, num_bins)
```

**핵심 과정**:
1. CB 원자 간 유클리드 거리 계산
2. 거리를 사전 정의된 구간으로 분할 (3.25Å ~ 50.75Å, 38개 bin)
3. One-hot encoding으로 변환하여 거리 히스토그램 생성

### 3.2 Frame 기반 단위 벡터 계산
```python
# Compute unit vector in each frame
frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)  # (B, T, N, 3, 3)
frame_t = frame_t.unsqueeze(2).unsqueeze(-1)          # (B, T, N, 3, 1)
ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)      # (B, T, N, 1, 3, 1)

# Transform CA coordinates to local frame
vector = torch.matmul(frame_rot, (ca_coords - frame_t))  # (B, T, N, N, 3, 1)
norm = torch.norm(vector, dim=-1, keepdim=True)         # (B, T, N, N, 1, 1)
unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
unit_vector = unit_vector.squeeze(-1)                   # (B, T, N, N, 3)
```

**목적**: 
- 각 잔기의 로컬 좌표계에서 CA 원자 방향 벡터 계산
- 구조의 방향성(orientation) 정보 제공

### 3.3 Feature 연결 및 투영
```python
# Concatenate all template features
a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]  # List of tensors
a_tij = torch.cat(a_tij, dim=-1)                          # (B, T, N, N, feature_dim)
a_tij = a_tij * asym_mask.unsqueeze(-1)                   # Apply chain mask

# Add residue type information pairwise
res_type_i = res_type[:, :, :, None]                      # (B, T, N, 1, 32)
res_type_j = res_type[:, :, None, :]                      # (B, T, 1, N, 32)
res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)  # (B, T, N, N, 32)
res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)  # (B, T, N, N, 32)

a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)  # (B, T, N, N, total_dim)
a_tij = self.a_proj(a_tij)                                  # (B, T, N, N, template_dim)
```

**Feature 구성요소**:
- Distogram (38 dimensions)
- CB mask (1 dimension)
- Unit vectors (3 dimensions)  
- Frame mask (1 dimension)
- Residue types pairwise (64 dimensions = 32 + 32)
- **Total**: 107 dimensions → `template_dim`으로 투영

## 4. Pairformer 처리

### 4.1 입력 임베딩 생성
```python
# Project pairwise embeddings to template dimension
v = self.z_proj(self.z_norm(z[:, None])) + a_tij  # (B, T, N, N, template_dim)

# Reshape for processing
v = v.view(B * T, *v.shape[2:])  # (B*T, N, N, template_dim)
```

### 4.2 Pairformer 자기 주의 메커니즘
```python
# Process through Pairformer blocks
v = v + self.pairformer(v, pair_mask, use_trifast=use_trifast)
v = self.v_norm(v)
v = v.view(B, T, *v.shape[1:])  # (B, T, N, N, template_dim)
```

**Pairformer 역할**:
- Template feature에 대한 self-attention 계산
- Template 내 잔기 간 상호작용 모델링
- 여러 레이어를 통한 깊은 표현 학습

## 5. Multi-Template 집계

### 5.1 가중 평균 계산
```python
# Aggregate multiple templates using weighted average
template_mask = template_mask[:, :, None, None, None]  # (B, T, 1, 1, 1)
num_templates = num_templates[:, None, None, None]     # (B, 1, 1, 1)

u = (v * template_mask).sum(dim=1) / num_templates.to(v)  # (B, N, N, template_dim)
```

**집계 과정**:
1. 각 template의 유효성을 `template_mask`로 확인
2. 유효한 template들의 feature를 합산
3. 유효한 template 수로 나누어 평균 계산

### 5.2 최종 출력 투영
```python
# Project back to pairwise embedding dimension
u = self.u_proj(self.relu(u))  # (B, N, N, token_z)
return u
```

## 6. 모델 통합 및 활용

### 6.1 Boltz2 모델에서의 통합
```python
# From src/boltz/model/models/boltz2.py
if self.use_templates:
    if self.is_template_compiled and not self.training:
        template_module = self.template_module._orig_mod
    else:
        template_module = self.template_module
    
    z = z + template_module(z, feats, pair_mask, use_trifast=use_trifast)
```

**통합 방식**:
- Template module의 출력을 기존 pairwise embedding `z`에 **잔여 연결(residual connection)**로 추가
- Template 정보가 있으면 구조 예측에 도움을 주고, 없으면 영향을 주지 않음

### 6.2 후속 처리
Template 정보가 통합된 pairwise embedding은:
1. **MSA Module**: Multiple sequence alignment 정보와 결합
2. **트랜지션 레이어**: 추가적인 feature 처리
3. **삼각형 주의 메커니즘**: 구조적 제약조건 학습
4. **구조 예측 헤드**: 최종 3D 좌표 예측

## 7. TemplateV2Module과의 차이점

TemplateV2Module은 기본 TemplateModule과 유사하지만 몇 가지 추가 기능이 있습니다:

```python
# Additional feature in TemplateV2Module
visibility_ids = feats["visibility_ids"]  # Chain visibility tracking

# Different masking strategy  
tmlp_pair_mask = (
    visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]
).float()
```

**주요 차이점**:
- `visibility_ids`를 사용한 더 정교한 마스킹
- 올리고머 복합체에서 체인 간 가시성 제어
- 동일한 entity에서 온 체인들 간의 template 정보 공유 허용

## 8. Template 정보의 기여도

### 8.1 구조 예측에 대한 기여
1. **Distance Constraints**: CB 거리 정보로 공간적 제약조건 제공
2. **Orientation Information**: Frame 정보로 잔기 방향성 가이드
3. **Sequence-Structure Mapping**: 서열과 구조 간 대응관계 학습
4. **Multi-chain Assembly**: 복합체 구조에서 체인 간 상호작용 모델링

### 8.2 모델 성능 향상
- **Prior Knowledge**: 알려진 구조 정보를 통한 사전 지식 제공
- **Regularization**: 비물리적 구조 예측 방지
- **Convergence**: 학습 수렴 속도 향상
- **Accuracy**: 전반적인 구조 예측 정확도 개선

Template 정보는 Boltz-2 모델에서 단순한 보조 정보가 아니라, 구조 예측의 핵심적인 가이드 역할을 수행하며, 특히 복잡한 단백질 복합체나 새로운 폴드에 대한 예측 성능을 크게 향상시킵니다. 