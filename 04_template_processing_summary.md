# Boltz-2 Template 처리 종합 정리

## 개요

이 문서는 Boltz-2에서 Template 정보가 어떻게 처리되고 활용되는지에 대한 종합적인 정리입니다. Input YAML 파일부터 최종 구조 예측까지의 전체 흐름을 요약하고, 각 단계의 핵심 기능과 코드적 구현을 정리합니다.

## 1. 전체 아키텍처 요약

### 1.1 주요 구성요소
```
Input YAML → Schema Parsing → Template Feature Extraction → Neural Network → Structure Prediction
     ↓              ↓                    ↓                      ↓               ↓
[templates]   parse_boltz_schema   process_template_features   TemplateModule   Output
   section      (schema.py)         (featurizerv2.py)         (trunkv2.py)    Structure
```

### 1.2 핵심 데이터 흐름
1. **입력 단계**: YAML 파일의 templates 섹션
2. **파싱 단계**: mmCIF 파일 파싱 및 서열 정렬
3. **Feature 추출**: 신경망 입력용 feature 생성  
4. **모델 처리**: TemplateModule에서 feature 처리
5. **통합**: Pairwise embedding에 template 정보 통합

## 2. Template 처리 단계별 요약

### 2.1 Schema Parsing 단계

**위치**: `src/boltz/data/parse/schema.py` (라인 1685-1778)

**핵심 기능**:
```python
# Template 정보 파싱
for template in schema.get("templates", []):
    path = template["cif"]
    template_id = Path(path).stem
    
    # mmCIF 파일 파싱
    parsed_template = parse_mmcif(path, mols=ccd, moldir=mol_dir)
    
    # 서열 정렬 수행
    template_records = get_template_records_from_search(
        template_id, chain_ids, sequences, 
        template_chain_ids, template_sequences
    )
```

**출력**:
- `TemplateInfo` 객체들 (서열 정렬 정보 포함)
- Template 구조 데이터 딕셔너리
- Template constraint 정보 (신규 기능)

### 2.2 Feature Extraction 단계

**위치**: `src/boltz/data/feature/featurizerv2.py`

**핵심 함수들**:
1. `process_template_features()` - 메인 처리 함수
2. `compute_template_features()` - 개별 feature 계산
3. `load_dummy_templates_features()` - 더미 feature 생성

**Feature 생성 과정**:
```python
# 1. Template 그룹화
name_to_templates = {}
for template_info in data.record.templates:
    name_to_templates.setdefault(template_info.name, []).append(template_info)

# 2. Token 매핑
offset = template.template_st - template.query_st
q_indices = dict(zip(q_tokens["res_idx"], q_tokens["token_idx"]))

# 3. Feature 추출
res_type[idx] = token["res_type"]
frame_rot[idx] = token["frame_rot"].reshape(3, 3)
cb_coords[idx] = token["disto_coords"]
# ... 기타 feature들
```

**출력 Feature 딕셔너리**:
- `template_restype`: 잔기 타입 (B, T, N, 32)
- `template_frame_rot`: 좌표계 회전 (B, T, N, 3, 3)
- `template_frame_t`: 좌표계 평행이동 (B, T, N, 3)
- `template_cb`: CB 좌표 (B, T, N, 3)
- `template_ca`: CA 좌표 (B, T, N, 3)
- `template_mask_*`: 유효성 마스크들
- `visibility_ids`: 체인 가시성 ID

### 2.3 Neural Network 처리 단계

**위치**: `src/boltz/model/modules/trunkv2.py`

**TemplateModule의 핵심 처리**:
```python
# 1. Distogram 계산
cb_dists = torch.cdist(cb_coords, cb_coords)
distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
distogram = one_hot(distogram, num_classes=self.num_bins)

# 2. Unit vector 계산  
vector = torch.matmul(frame_rot, (ca_coords - frame_t))
unit_vector = vector / torch.norm(vector, dim=-1, keepdim=True)

# 3. Feature 결합
a_tij = torch.cat([distogram, b_cb_mask, unit_vector, b_frame_mask], dim=-1)
a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)

# 4. Pairformer 처리
v = self.z_proj(self.z_norm(z[:, None])) + self.a_proj(a_tij)
v = v + self.pairformer(v, pair_mask)

# 5. Multi-template 집계
u = (v * template_mask).sum(dim=1) / num_templates.to(v)
return self.u_proj(self.relu(u))
```

**처리 흐름**:
1. CB 좌표로부터 distogram 생성
2. Frame 정보로부터 방향 벡터 계산
3. 모든 feature를 연결하여 투영
4. Pairformer로 self-attention 처리
5. 여러 template을 가중 평균으로 집계
6. Pairwise embedding 차원으로 투영

## 3. 핵심 기술적 특징

### 3.1 Token 매핑 메커니즘
```python
# Offset 기반 매핑
offset = template.template_st - template.query_st
toks = [t for t in toks if t["res_idx"] - offset in q_indices]

# Query-Template 대응관계 설정
for t in toks:
    q_idx = q_indices[t["res_idx"] - offset]
    row_tokens.append({"token": t, "pdb_id": template_id, "q_idx": q_idx})
```

**특징**:
- 서열 정렬 정보를 활용한 정확한 잔기 매핑
- Missing residue 처리 가능
- Multi-chain 복합체 지원

### 3.2 Multi-Template 처리
```python
# Template별 feature 계산
for template_id, (template_name, templates) in enumerate(name_to_templates.items()):
    row_features = compute_template_features(data, row_tokens, max_tokens)
    template_features.append(row_features)

# 모든 template을 하나의 텐서로 스택킹
out = {}
for k in template_features[0]:
    out[k] = torch.stack([f[k] for f in template_features])
```

**장점**:
- 여러 template을 동시에 활용
- Template 품질에 따른 가중 평균
- 다양한 구조 정보의 종합적 활용

### 3.3 Chain-wise Attention
```python
# 체인 내에서만 attention 계산
asym_mask = (asym_id[:, :, None] == asym_id[:, None, :]).float()
a_tij = a_tij * asym_mask.unsqueeze(-1)
```

**목적**:
- Multi-chain 복합체에서 체인 간 정보 누출 방지
- 각 체인의 독립적인 구조 예측 보장

## 4. 코드 수정 사항 (신규 기능)

### 4.1 Template Constraint Generation
**위치**: `src/boltz/data/parse/template.py`

```python
class TemplateConstraintGenerator:
    def generate_template_constraints(self, query_sequence, template_structure, 
                                    template_chain_id, query_chain_id):
        # CB 좌표 추출
        cb_coords = self.extract_cb_coordinates(template_structure, template_chain_id)
        
        # 거리 맵 계산
        distance_map = self.compute_distance_map(cb_coords)
        
        # 서열 매핑
        mapping = self.map_sequences(query_sequence, template_sequence)
        
        # 제약조건 생성
        constraints = self.generate_constraints(distance_map, mapping)
        
        return constraints
```

**기능**:
- Template 구조로부터 자동 거리 제약조건 생성
- NMR/MinDistance constraint 형태로 출력
- 기존 explicit constraint와 함께 처리

### 4.2 새로운 Constraint 타입
```python
# schema.py에서 추가된 constraint 처리
elif "min_distance" in constraint:
    c1, r1, a1 = tuple(constraint["min_distance"]["atom1"])
    c2, r2, a2 = tuple(constraint["min_distance"]["atom2"])
    distance = float(constraint["min_distance"]["distance"])
    min_distances.append((c1, c2, r1, r2, a1, a2, distance))

elif "nmr_distance" in constraint:
    # NMR 거리 제약조건 처리
    lower_bound = float(constraint["nmr_distance"].get("lower_bound", 0.0))
    upper_bound = float(constraint["nmr_distance"].get("upper_bound", float('inf')))
    nmr_distances.append((c1, c2, r1, r2, a1, a2, lower_bound, upper_bound, weight))
```

## 5. 성능 최적화 특징

### 5.1 효율적인 메모리 사용
- Padding을 통한 배치 처리 최적화
- Template 수에 따른 동적 할당
- 불필요한 계산 스킵 (dummy template 사용)

### 5.2 계산 최적화
```python
# Mixed precision 비활성화 (정확도 우선)
with torch.autocast(device_type="cuda", enabled=False):
    cb_dists = torch.cdist(cb_coords, cb_coords)
    
# 벡터화된 거리 계산
distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
```

### 5.3 마스킹 전략
- CB/Frame 유효성 마스크
- Template 존재 마스크  
- Chain asymmetric 마스크
- Visibility ID 기반 마스킹

## 6. 실제 활용 효과

### 6.1 구조 예측 정확도 향상
- **Distance Guidance**: CB 거리 정보로 공간적 제약조건 제공
- **Orientation Information**: Frame 정보로 잔기 방향성 가이드
- **Multi-chain Assembly**: 복합체 구조 예측 정확도 향상

### 6.2 모델 학습 안정성
- **Prior Knowledge**: 알려진 구조 정보로 학습 가이드
- **Regularization**: 비물리적 구조 예측 방지
- **Convergence**: 빠른 수렴과 안정적인 학습

## 7. 한계점 및 개선 방향

### 7.1 현재 한계점
- Template 품질에 의존적
- 서열 유사성이 낮으면 효과 제한적
- 계산 복잡도 증가

### 7.2 향후 개선 방향
- Template 선택 알고리즘 개선
- 구조 기반 정렬 방법 도입
- 더 정교한 가중치 학습 메커니즘

## 8. 결론

Boltz-2의 Template 처리 시스템은 다음과 같은 특징을 가집니다:

1. **완전한 파이프라인**: YAML 입력부터 신경망 통합까지 end-to-end 처리
2. **유연한 아키텍처**: Template 유무에 관계없이 일관된 인터페이스 제공
3. **다양한 정보 활용**: 거리, 방향, 서열 정보의 종합적 활용
4. **확장성**: 새로운 constraint 타입과 기능 추가 가능
5. **효율성**: 메모리와 계산 최적화를 통한 실용적 성능

Template 정보는 Boltz-2에서 단순한 보조 정보가 아니라 구조 예측의 핵심 가이드로 작용하며, 특히 복잡한 단백질 복합체나 도전적인 구조 예측 문제에서 중요한 역할을 수행합니다.

전체 시스템은 모듈화되어 있어 각 구성요소를 독립적으로 개선하거나 확장할 수 있으며, 이는 향후 더 정교한 template 활용 방법을 개발하는 데 유리한 기반을 제공합니다. 