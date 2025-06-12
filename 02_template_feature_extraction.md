# Boltz-2 Template Feature 추출 상세 분석

## 개요

Boltz-2에서 Template Feature 추출은 `src/boltz/data/feature/featurizerv2.py`에서 주로 3개의 함수를 통해 처리됩니다:

1. `process_template_features()` - 메인 template feature 처리 함수
2. `compute_template_features()` - 개별 template feature 계산 함수  
3. `load_dummy_templates_features()` - template이 없을 때 더미 feature 생성

## 1. Feature 생성 부분 상세 분석

### 1.1 입력 데이터 구조

```python
def process_template_features(
    data: Tokenized,  # Query 구조의 토큰화된 데이터
    max_tokens: int,  # 최대 토큰 수
) -> dict[str, torch.Tensor]:
```

**입력 정보**:
- `data.record.templates`: TemplateInfo 객체들의 리스트 (서열 정렬 정보 포함)
- `data.templates`: Template 구조 데이터 딕셔너리 
- `data.template_tokens`: Template의 토큰화된 구조 정보
- `data.structure.chains`: Query 구조의 체인 정보

### 1.2 처리 과정

#### Step 1: Template 그룹화 및 매핑 설정
```python
# Group templates by name
name_to_templates: dict[str, list[TemplateInfo]] = {}
for template_info in data.record.templates:
    name_to_templates.setdefault(template_info.name, []).append(template_info)

# Map chain name to asym_id  
chain_name_to_asym_id = {}
for chain in data.structure.chains:
    chain_name_to_asym_id[chain["name"]] = chain["asym_id"]
```

- **목적**: 같은 이름의 template들을 그룹화하고, chain name과 asym_id 매핑 테이블 생성
- **결과**: `name_to_templates` 딕셔너리와 `chain_name_to_asym_id` 매핑

#### Step 2: Token 매핑 및 Feature 추출
```python
for template_id, (template_name, templates) in enumerate(name_to_templates.items()):
    row_tokens = []
    template_structure = data.templates[template_name]
    template_tokens = data.template_tokens[template_name]
    
    for template in templates:
        offset = template.template_st - template.query_st
        
        # Get query and template tokens to map residues
        query_tokens = data.tokens
        chain_id = chain_name_to_asym_id[template.query_chain]
        q_tokens = query_tokens[query_tokens["asym_id"] == chain_id]
        q_indices = dict(zip(q_tokens["res_idx"], q_tokens["token_idx"]))
        
        # Get the template tokens at the query residues
        chain_id = tmpl_chain_name_to_asym_id[template.template_chain]
        toks = template_tokens[template_tokens["asym_id"] == chain_id]
        toks = [t for t in toks if t["res_idx"] - offset in q_indices]
        
        for t in toks:
            q_idx = q_indices[t["res_idx"] - offset]
            row_tokens.append({
                "token": t,           # Template token 정보
                "pdb_id": template_id, # Template ID
                "q_idx": q_idx,       # Query token index
            })
```

- **목적**: Template과 Query 간의 잔기 대응관계 설정
- **핵심 계산**: `offset = template.template_st - template.query_st`
- **결과**: `row_tokens` 리스트 - 매핑된 토큰 정보들

#### Step 3: Feature 계산 및 스택킹
```python
# Compute template features for each row
row_features = compute_template_features(data, row_tokens, max_tokens)
template_features.append(row_features)

# Stack each feature
out = {}
for k in template_features[0]:
    out[k] = torch.stack([f[k] for f in template_features])
```

- **목적**: 개별 template feature 계산 후 모든 template을 하나의 텐서로 결합
- **결과**: Template dimension이 추가된 feature 딕셔너리

## 2. compute_template_features() 상세 분석

### 2.1 Feature 배열 초기화
```python
def compute_template_features(
    query_tokens: Tokenized,
    tmpl_tokens: list[dict],  # 매핑된 토큰 정보
    num_tokens: int,
) -> dict:
    # Allocate features
    res_type = np.zeros((num_tokens,), dtype=np.int64)
    frame_rot = np.zeros((num_tokens, 3, 3), dtype=np.float32)
    frame_t = np.zeros((num_tokens, 3), dtype=np.float32)
    cb_coords = np.zeros((num_tokens, 3), dtype=np.float32)
    ca_coords = np.zeros((num_tokens, 3), dtype=np.float32)
    frame_mask = np.zeros((num_tokens,), dtype=np.float32)
    cb_mask = np.zeros((num_tokens,), dtype=np.float32)
    template_mask = np.zeros((num_tokens,), dtype=np.float32)
    query_to_template = np.zeros((num_tokens,), dtype=np.int64)
    visibility_ids = np.zeros((num_tokens,), dtype=np.float32)
```

### 2.2 Template Token 정보 추출
```python
for token_dict in tmpl_tokens:
    idx = token_dict["q_idx"]          # Query token index
    pdb_id = token_dict["pdb_id"]      # Template ID
    token = token_dict["token"]        # Template token data
    query_token = query_tokens.tokens[idx]
    
    # Extract template features from token
    res_type[idx] = token["res_type"]                    # Residue type
    frame_rot[idx] = token["frame_rot"].reshape(3, 3)    # Frame rotation matrix
    frame_t[idx] = token["frame_t"]                      # Frame translation vector
    cb_coords[idx] = token["disto_coords"]               # CB coordinates  
    ca_coords[idx] = token["center_coords"]              # CA coordinates
    cb_mask[idx] = token["disto_mask"]                   # CB validity mask
    frame_mask[idx] = token["frame_mask"]                # Frame validity mask
    template_mask[idx] = 1.0                             # Template presence mask
```

### 2.3 Visibility ID 설정
```python
# Set visibility_id for templated chains
for asym_id, pdb_id in asym_id_to_pdb_id.items():
    indices = (query_tokens.tokens["asym_id"] == asym_id).nonzero()
    visibility_ids[indices] = pdb_id

# Set visibility for non templated chain + oligomerics  
for asym_id in np.unique(query_tokens.structure.chains["asym_id"]):
    if asym_id not in asym_id_to_pdb_id:
        indices = (query_tokens.tokens["asym_id"] == asym_id).nonzero()
        visibility_ids[indices] = -1 - asym_id  # Negative ID for non-templated
```

## 3. 최종 출력 Feature 딕셔너리

### 3.1 출력 변수와 의미

```python
return {
    "template_restype": res_type,        # Shape: (T, N, 32) - One-hot encoded residue types
    "template_frame_rot": frame_rot,     # Shape: (T, N, 3, 3) - Local coordinate frame rotations  
    "template_frame_t": frame_t,         # Shape: (T, N, 3) - Local coordinate frame translations
    "template_cb": cb_coords,            # Shape: (T, N, 3) - CB atom coordinates
    "template_ca": ca_coords,            # Shape: (T, N, 3) - CA atom coordinates  
    "template_mask_cb": cb_mask,         # Shape: (T, N) - CB atom validity mask
    "template_mask_frame": frame_mask,   # Shape: (T, N) - Frame validity mask
    "template_mask": template_mask,      # Shape: (T, N) - Overall template mask
    "query_to_template": query_to_template,  # Shape: (T, N) - Query to template mapping
    "visibility_ids": visibility_ids,    # Shape: (T, N) - Chain visibility identifiers
}
```

**차원 설명**:
- `T`: Template 개수 (최대 template 수)
- `N`: Token 개수 (max_tokens)

### 3.2 각 Feature의 역할

1. **`template_restype`**: 
   - Template 잔기 타입의 one-hot encoding
   - 모델이 template의 서열 정보를 이해하는데 사용

2. **`template_frame_rot/t`**:
   - 각 잔기의 로컬 좌표계 정보
   - 3D 구조의 방향성과 위치 정보 제공

3. **`template_cb/ca`**:
   - CB와 CA 원자의 3D 좌표
   - 거리 계산과 구조적 제약조건 생성에 사용

4. **`template_mask_*`**:
   - 각 위치에서 정보의 유효성을 나타내는 마스크
   - Missing residue나 원자 처리에 중요

5. **`visibility_ids`**:
   - 어떤 체인이 어떤 template에서 왔는지 추적
   - Multi-chain 복합체에서 중요

## 4. Token 매핑 상세 분석

### 4.1 매핑 과정

```python
# 1. Offset 계산
offset = template.template_st - template.query_st

# 2. Query token 인덱스 생성
q_tokens = query_tokens[query_tokens["asym_id"] == chain_id]
q_indices = dict(zip(q_tokens["res_idx"], q_tokens["token_idx"]))

# 3. Template token 필터링 및 매핑  
toks = template_tokens[template_tokens["asym_id"] == chain_id]
toks = [t for t in toks if t["res_idx"] - offset in q_indices]

# 4. 최종 매핑
for t in toks:
    q_idx = q_indices[t["res_idx"] - offset]
    row_tokens.append({"token": t, "pdb_id": template_id, "q_idx": q_idx})
```

### 4.2 매핑 로직 설명

1. **Offset 계산**: Template와 Query 서열의 시작점 차이 계산
2. **인덱스 매핑**: Query 잔기 번호를 token 인덱스로 변환하는 딕셔너리 생성
3. **유효 Token 필터링**: Offset을 적용하여 Query에 대응되는 Template token만 선택
4. **최종 매핑**: Template token을 해당하는 Query token 위치에 매핑

## 5. 모델에서의 활용 연결점

### 5.1 TemplateModule에서의 사용

Template feature들은 `TemplateModule`에서 다음과 같이 사용됩니다:

```python
# From TemplateModule.forward()
res_type = feats["template_restype"]        # 잔기 타입 정보
frame_rot = feats["template_frame_rot"]     # 좌표계 회전
frame_t = feats["template_frame_t"]         # 좌표계 평행이동  
cb_coords = feats["template_cb"]            # CB 좌표
ca_coords = feats["template_ca"]            # CA 좌표
cb_mask = feats["template_mask_cb"]         # CB 마스크
template_mask = feats["template_mask"]      # 전체 마스크
```

### 5.2 주요 활용 방식

1. **Distogram 계산**: CB 좌표로부터 거리 히스토그램 생성
2. **Frame 변환**: 좌표계 정보를 이용한 단위 벡터 계산  
3. **Feature 결합**: Template 정보와 잔기 타입을 결합
4. **Attention 계산**: Pairformer에서 template 정보 처리
5. **Multi-template 집계**: 여러 template 정보를 가중 평균으로 통합

## 6. Dummy Template 처리

Template이 없는 경우 `load_dummy_templates_features()`가 호출되어 영값으로 초기화된 동일한 구조의 feature 딕셔너리를 반환합니다:

```python
def load_dummy_templates_features(tdim: int, num_tokens: int) -> dict:
    # 모든 feature를 0으로 초기화하여 일관된 인터페이스 제공
    res_type = np.zeros((tdim, num_tokens), dtype=np.int64)
    frame_rot = np.zeros((tdim, num_tokens, 3, 3), dtype=np.float32)
    # ... 기타 feature들도 동일하게 0으로 초기화
```

이를 통해 template 유무에 관계없이 모델이 일관된 인터페이스로 동작할 수 있습니다. 