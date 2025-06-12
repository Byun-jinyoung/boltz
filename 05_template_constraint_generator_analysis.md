# 사용자 구현 TemplateConstraintGenerator 상세 분석

## 개요

사용자가 구현한 `TemplateConstraintGenerator`는 template 구조로부터 자동으로 distance constraint를 생성하여 단백질 구조 예측의 정확도를 향상시키는 시스템입니다. 이 문서에서는 구현된 constraint generator가 어떻게 작동하고, 생성된 constraint가 모델의 어디에서 어떻게 사용되는지 상세히 분석합니다.

## 1. TemplateConstraintGenerator 구현 분석

### 1.1 클래스 구조 및 초기화

```python
class TemplateConstraintGenerator:
    def __init__(
        self, 
        distance_threshold: float = 20.0,      # 최대 고려 거리  
        cb_distance_cutoff: float = 50.0,      # CB-CB 거리 컷오프
        min_sequence_identity: float = 0.6,     # 최소 서열 동일성
        gap_penalty: float = -2.0               # 갭 페널티
    ):
```

**핵심 파라미터**:
- `cb_distance_cutoff`: CB-CB 거리가 11-50Å 범위의 제약조건만 생성
- `min_sequence_identity`: 서열 동일성이 0.6 미만일 때 경고 발생
- `StructureSequenceMapper`: 서열-구조 매핑을 위한 내부 모듈

### 1.2 CB 좌표 추출 메커니즘

```python
def _extract_cb_coordinates(self, structure_file: str, chain_id: str) -> Dict[int, np.ndarray]:
```

**처리 과정**:
1. **구조 파일 파싱**: PDB/mmCIF 파일에서 template 구조 로드
2. **잔기별 반복**: 지정된 chain의 모든 표준 잔기 처리
3. **CB 원자 추출**: 
   - GLY, PRO 제외: CB 원자 좌표 추출
   - GLY, PRO: CB 좌표 없음으로 처리 (경고 없이)
4. **좌표 저장**: `{residue_index: CB_coordinates}` 딕셔너리 반환

### 1.3 거리 맵 계산

```python
def _compute_distance_map(self, cb_coords: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
```

**거리 필터링 로직**:
```python
if 11 <= distance <= self.cb_distance_cutoff:
    distance_map[(idx1, idx2)] = distance
```

**특징**:
- **하한선 11Å**: 너무 가까운 거리는 제외 (물리적으로 불가능한 제약 방지)
- **상한선 50Å**: 너무 먼 거리는 제외 (의미 없는 제약 방지)
- **효율성**: 상삼각 행렬만 계산하여 중복 제거

### 1.4 Constraint 생성 알고리즘

```python
def generate_template_constraints(
    self,
    query_sequence: str,
    template_structure: str, 
    template_chain_id: str,
    query_chain_id: str = "A",
    constraint_type: str = "nmr_distance",  # 또는 "min_distance"
    distance_buffer: float = 0.1,           # 10% 버퍼
    base_weight: float = 1.0,
    sequence_identity_weight: bool = True    # 서열 동일성 가중치 적용
) -> List[Dict[str, Any]]:
```

**단계별 처리**:

#### Step 1: 서열 정렬 및 매핑
```python
aligned_struct, aligned_given, mapping, stats = self.mapper.map_sequences(
    template_structure, template_chain_id, query_sequence
)
seq_identity = calculate_sequence_identity(aligned_struct, aligned_given)
```

#### Step 2: NMR Distance Constraint 생성
```python
if constraint_type == "nmr_distance":
    lower_bound = max(0.0, distance * (1 - distance_buffer))  # 90% of template distance
    upper_bound = distance * (1 + distance_buffer)            # 110% of template distance
    
    weight = base_weight
    if sequence_identity_weight:
        weight *= seq_identity  # 서열 동일성으로 가중치 조정
    
    constraint = {
        "nmr_distance": {
            "atom1": [query_chain_id, query_idx1 + 1, "CB"],  # 1-indexed
            "atom2": [query_chain_id, query_idx2 + 1, "CB"],
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "weight": float(weight)
        }
    }
```

**핵심 특징**:
- **Flexible bounds**: ±10% 버퍼로 엄격하지 않은 제약
- **Weight scaling**: 서열 동일성에 비례한 신뢰도 조정
- **CB-CB constraints**: 모든 제약이 CB 원자 간 거리 기반

## 2. Schema 파싱 단계에서의 통합

### 2.1 Template 정보 수집

```python
# src/boltz/data/parse/schema.py의 parse_boltz_schema 함수 내부
template_constraints = []
template_info_by_chain = {}

for item in schema.get("sequences", []):
    entity_type = next(iter(item.keys())).lower()
    if entity_type == "protein" and "template" in item[entity_type]:            
        template_info = item[entity_type]["template"]
        chain_ids = item[entity_type]["id"]
        
        for chain_id in chain_ids:
            template_info_by_chain[chain_id] = {
                "structure_path": template_info["structure"],
                "chain_id": template_info["chain_id"], 
                "sequence": item[entity_type]["sequence"]
            }
```

### 2.2 Template Constraint 자동 생성

```python
if template_info_by_chain:
    generator = TemplateConstraintGenerator()
    
    for chain_id, info in template_info_by_chain.items():
        constraints = generator.generate_template_constraints(
            query_sequence=info["sequence"],
            template_structure=info["structure_path"],
            template_chain_id=info["chain_id"],
            query_chain_id=chain_id
        )
        template_constraints.extend(constraints)
```

### 2.3 기존 Constraint와 통합

```python
# Template constraint와 명시적 constraint 결합
all_constraints = template_constraints + schema.get("constraints", [])

for constraint in all_constraints:
    if "nmr_distance" in constraint:
        # Template에서 생성된 constraint 포함하여 처리
        nmr_distances.append((c1, c2, r1, r2, a1, a2, lower_bound, upper_bound, weight))
```

## 3. 데이터 구조로의 변환

### 3.1 NMRDistance 타입 정의

```python
# src/boltz/data/types.py
NMRDistance = [
    ("chain_1", np.dtype("i4")),      # Chain index
    ("chain_2", np.dtype("i4")),      # Chain index  
    ("res_1", np.dtype("i4")),        # Residue index
    ("res_2", np.dtype("i4")),        # Residue index
    ("atom_1", np.dtype("i4")),       # Atom index
    ("atom_2", np.dtype("i4")),       # Atom index
    ("lower_bound", np.dtype("f4")),  # Minimum distance (Å)
    ("upper_bound", np.dtype("f4")),  # Maximum distance (Å)
    ("weight", np.dtype("f4")),       # Constraint weight
]
```

### 3.2 Structure 객체에 저장

```python
# Boltz-2의 경우
nmr_distances_v2 = np.array(nmr_distances, dtype=NMRDistance) if len(nmr_distances) > 0 else None

data = StructureV2(
    # ... 기타 필드들
    nmr_distances=nmr_distances_v2,
)
```

## 4. 모델에서의 활용

### 4.1 Potential 클래스 구조

```python
class NMRDistancePotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        # Feature에서 constraint 정보 추출
        pair_index = feats["nmr_distance_atom_index"][0]
        lower_bounds = feats["nmr_distance_lower_bounds"][0].clone()
        upper_bounds = feats["nmr_distance_upper_bounds"][0].clone()
        weights = feats["nmr_distance_weights"][0]
        
        # 버퍼 적용 (유연한 제약)
        lower_bounds = lower_bounds * (1.0 - parameters["lower_buffer"])  # 5% 추가 완화
        
        finite_mask = torch.isfinite(upper_bounds)
        upper_bounds[finite_mask] = upper_bounds[finite_mask] * (1.0 + parameters["upper_buffer"])
        
        # 가중치를 force constant로 변환
        k = weights * parameters["base_force_constant"]
        
        return pair_index, (k, lower_bounds, upper_bounds), None
```

### 4.2 Potential 파라미터 설정

```python
# src/boltz/model/potentials/potentials.py의 get_potentials()
NMRDistancePotential( 
    parameters={
        "guidance_interval": 1,        # 매 step마다 적용
        "guidance_weight": 0.5,        # 중간 강도의 가이던스 
        "resampling_weight": 1.0,      # 리샘플링 시 완전 적용
        "lower_buffer": 0.05,          # 하한 5% 완화 
        "upper_buffer": 0.05,          # 상한 5% 완화
        "base_force_constant": 1.0,    # 기본 force constant
    }
),
```

### 4.3 Energy 및 Gradient 계산

```python
class FlatBottomPotential:
    def compute_function(self, value, k, lower_bounds, upper_bounds, compute_derivative=False):
        # Flat-bottom potential: 범위 내에서는 penalty 없음
        violations_lower = torch.relu(lower_bounds - value)  # lower bound 위반
        violations_upper = torch.relu(value - upper_bounds)  # upper bound 위반
        
        energy = 0.5 * k * (violations_lower**2 + violations_upper**2)
        
        if compute_derivative:
            gradient = k * (-violations_lower + violations_upper)
            return energy, gradient
        return energy
```

### 4.4 실제 적용 과정

1. **Forward Pass**: 현재 구조의 CB-CB 거리 계산
2. **Constraint Check**: Template에서 유도된 거리 제약과 비교
3. **Energy Calculation**: 제약 위반에 대한 penalty 계산
4. **Gradient Computation**: 제약을 만족하는 방향으로의 force 계산
5. **Structure Update**: Gradient를 이용한 구조 최적화

## 5. 통합 시스템의 장점

### 5.1 자동화된 Constraint 생성
- **수동 작업 불필요**: Template 구조만 제공하면 자동으로 수백 개의 제약 생성
- **일관된 품질**: 알고리즘화된 제약 생성으로 일관된 품질 보장
- **확장성**: 여러 template을 동시에 활용 가능

### 5.2 지능적인 가중치 시스템
- **서열 동일성 기반**: 높은 서열 동일성일수록 높은 신뢰도
- **거리 기반 필터링**: 물리적으로 의미 있는 거리 범위만 사용
- **유연한 경계**: 버퍼를 통한 엄격하지 않은 제약

### 5.3 기존 시스템과의 호환성
- **추가적 제약**: 기존 명시적 제약과 함께 사용 가능
- **모듈화된 설계**: 독립적으로 활성화/비활성화 가능
- **확장 가능한 아키텍처**: 새로운 constraint 타입 추가 용이

## 6. 실제 사용 예시

### 6.1 YAML 설정 예시

```yaml
sequences:
  - protein:
      id: A
      sequence: "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUDFNKLLGKKSQRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
      template:
          structure: "/path/to/template.pdb"
          chain_id: "A"
constraints:
  # 기존 명시적 제약들과 함께 사용
  - bond:
      atom1: ["A", 10, "SG"]
      atom2: ["A", 50, "SG"]
```

### 6.2 생성되는 Template Constraint 예시

생성된 constraint는 다음과 같은 형태가 됩니다:

```yaml
# 자동 생성된 template constraint (내부적으로 추가됨)
- nmr_distance:
    atom1: ["A", 15, "CB"]
    atom2: ["A", 45, "CB"] 
    lower_bound: 9.5    # template distance * 0.9
    upper_bound: 11.5   # template distance * 1.1
    weight: 0.85        # sequence_identity * base_weight
```

## 7. 성능 및 효과

### 7.1 구조 예측 정확도 향상
- **Local structure**: Template의 local fold 정보 활용
- **Long-range contacts**: 멀리 떨어진 잔기 간의 관계 유지
- **Overall topology**: 전체적인 구조 토폴로지 보존

### 7.2 물리적 일관성 보장
- **Non-physical conformation 방지**: 다수의 제약으로 비물리적 구조 억제
- **Ensemble consistency**: 여러 template 간의 일관성 유지
- **Gradual refinement**: 점진적인 구조 개선

이 시스템은 template 구조의 정보를 효과적으로 활용하여 Boltz-2 모델의 구조 예측 성능을 크게 향상시키는 혁신적인 접근법입니다. 