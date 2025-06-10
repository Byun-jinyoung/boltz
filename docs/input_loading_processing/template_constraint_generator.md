# Template-based Distance Constraint Generator Documentation

## 개요 (Overview)

`TemplateConstraintGenerator`는 단백질 구조 예측을 위한 템플릿 기반 거리 제약 조건 생성기입니다. 이 클래스는 기지의 단백질 구조(템플릿)를 사용하여 새로운 단백질 구조 예측 시 물리적으로 불가능한 형태를 방지하기 위한 거리 제약 조건을 생성합니다.

## 주요 기능 (Key Features)

- **템플릿 기반 제약 조건 생성**: 기존 단백질 구조를 템플릿으로 사용하여 거리 제약 조건 생성
- **서열 정렬 기반 매핑**: 쿼리 서열과 템플릿 구조 간의 정렬을 통한 정확한 잔기 매핑
- **Cβ-Cβ 거리 계산**: Cβ 원자 간 거리를 기반으로 한 거리 맵 생성
- **Boltz 스키마 호환**: Boltz 모델의 입력 스키마에 호환되는 제약 조건 형식

## 클래스 구조 (Class Architecture)

```mermaid
classDiagram
    class TemplateConstraintGenerator {
        -distance_threshold: float
        -cb_distance_cutoff: float
        -min_sequence_identity: float
        -mapper: StructureSequenceMapper
        
        +__init__(distance_threshold, cb_distance_cutoff, min_sequence_identity, gap_penalty)
        +generate_template_constraints(query_sequence, template_structure, template_chain_id, query_chain_id, constraint_type)
        +generate_constraints_for_boltz_schema(schema_data, template_info)
        
        -_calculate_sequence_identity(aligned_seq1, aligned_seq2)
        -_extract_cb_coordinates(structure_file, chain_id)
        -_compute_distance_map(cb_coords)
    }
    
    class StructureSequenceMapper {
        +map_sequences()
        +_get_structure_parser()
    }
    
    TemplateConstraintGenerator --> StructureSequenceMapper : uses
```

## 전체 워크플로우 (Complete Workflow)

```mermaid
flowchart TD
    A["Query Sequence + Template Structure<br/><code>query_sequence, template_structure</code>"] --> B["Initialize TemplateConstraintGenerator<br/><code>__init__()</code>"]
    B --> C["Sequence Alignment<br/><code>self.mapper.map_sequences()</code>"]
    C --> D{"Sequence Identity Check<br/><code>_calculate_sequence_identity()</code><br/><code>seq_identity >= min_sequence_identity</code>"}
    D -->|Identity < threshold| E["Warning: Low Identity<br/><code>warnings.warn()</code>"]
    D -->|Identity ≥ threshold| F["Extract Cβ Coordinates<br/><code>_extract_cb_coordinates()</code><br/><code>cb_coords</code>"]
    E --> F
    F --> G["Compute Distance Map<br/><code>_compute_distance_map()</code><br/><code>distance_map</code>"]
    G --> H["Map Template to Query Residues<br/><code>template_to_query = {t_idx: q_idx for q_idx, t_idx in mapping}</code>"]
    H --> I["Generate Constraints<br/><code>constraints = []</code><br/>Loop through distance_map"]
    I --> J["Remove Duplicates<br/><code>unique_constraints, seen_pairs</code>"]
    J --> K["Return Constraints List<br/><code>return unique_constraints</code>"]
    
    style D fill:#ffd700
    style E fill:#ffcccc
    style K fill:#ccffcc
```

## 세부 프로세스 분석 (Detailed Process Analysis)

### 1. 서열 정렬 및 매핑 (Sequence Alignment & Mapping)

```mermaid
flowchart LR
    A["Query Sequence<br/><code>query_sequence</code>"] --> B["StructureSequenceMapper<br/><code>self.mapper.map_sequences()</code>"]
    C["Template Structure<br/><code>template_structure, template_chain_id</code>"] --> B
    B --> D["Aligned Template Sequence<br/><code>aligned_template</code>"]
    B --> E["Aligned Query Sequence<br/><code>aligned_query</code>"]
    B --> F["Residue Mapping<br/><code>mapping</code>"]
    B --> G["Alignment Statistics<br/><code>stats</code>"]
    
    D --> H["Calculate Sequence Identity<br/><code>_calculate_sequence_identity()</code>"]
    E --> H
    H --> I{"Identity ≥ min_sequence_identity?<br/><code>seq_identity >= self.min_sequence_identity</code>"}
    I -->|No| J["Warning Message<br/><code>warnings.warn()</code>"]
    I -->|Yes| K["Proceed to Coordinate Extraction<br/><code>_extract_cb_coordinates()</code>"]
```

### 2. Cβ 좌표 추출 (Cβ Coordinate Extraction)

```mermaid
flowchart TD
    A["Template Structure File<br/><code>structure_file</code>"] --> B["Parse Structure<br/><code>self.mapper._get_structure_parser()</code>"]
    B --> C["Find Target Chain<br/><code>chain.id == chain_id</code>"]
    C --> D["Iterate Through Residues<br/><code>for residue in chain</code><br/><code>residue_idx = 0</code>"]
    D --> E{"Standard Residue?<br/><code>residue.id[0] == ' '</code>"}
    E -->|No| D
    E -->|Yes| F{"Has CB Atom?<br/><code>'CB' in residue</code>"}
    F -->|Yes| G["Extract CB Coordinate<br/><code>cb_atom = residue['CB']</code>"]
    F -->|No| H{"Has CA Atom?<br/><code>'CA' in residue</code>"}
    H -->|Yes| I["Extract CA Coordinate<br/>for Glycine<br/><code>cb_atom = residue['CA']</code>"]
    H -->|No| J["Skip Residue<br/><code>cb_atom = None</code>"]
    G --> K["Store in Dict<br/><code>cb_coords[residue_idx] = np.array(cb_atom.get_coord())</code>"]
    I --> K
    J --> D
    K --> L{"More Residues?<br/><code>residue_idx += 1</code>"}
    L -->|Yes| D
    L -->|No| M["Return Coordinate Dict<br/><code>return cb_coords</code>"]
    
    style F fill:#e1f5fe
    style H fill:#fff3e0
```

### 3. 거리 맵 계산 (Distance Map Computation)

```mermaid
flowchart TD
    A["Cβ Coordinates Dict<br/><code>cb_coords</code>"] --> B["Get Residue Indices<br/><code>residue_indices = list(cb_coords.keys())</code>"]
    B --> C["Initialize Distance Map<br/><code>distance_map = {}</code>"]
    C --> D["Select Residue Pair i,j<br/><code>for i, idx1 in enumerate(residue_indices)</code><br/><code>for idx2 in residue_indices[i+1:]</code>"]
    D --> E["Calculate Euclidean Distance<br/><code>distance = np.linalg.norm(coord1 - coord2)</code>"]
    E --> F{"Distance ≤ cutoff?<br/><code>distance <= self.cb_distance_cutoff</code>"}
    F -->|Yes| G["Store in Distance Map<br/><code>distance_map[(idx1, idx2)] = distance</code>"]
    F -->|No| H["Skip Pair"]
    G --> I{"More Pairs?"}
    H --> I
    I -->|Yes| D
    I -->|No| J["Return Distance Map<br/><code>return distance_map</code>"]
    
    style F fill:#e8f5e8
    style J fill:#f0f8ff
```

### 4. 제약 조건 생성 (Constraint Generation)

```mermaid
flowchart TD
    A["Distance Map + Residue Mapping<br/><code>distance_map, mapping</code>"] --> B["Initialize Constraints List<br/><code>constraints = []</code><br/><code>template_to_query = {t_idx: q_idx for q_idx, t_idx in mapping}</code>"]
    B --> C["For Each Distance Pair<br/><code>for (temp_i, temp_j), distance in distance_map.items()</code>"]
    C --> D{"Both Residues<br/>in Mapping?<br/><code>temp_i in template_to_query and temp_j in template_to_query</code>"}
    D -->|No| E["Skip Pair"]
    D -->|Yes| F["Get Query Residue Indices<br/><code>query_i = template_to_query[temp_i]</code><br/><code>query_j = template_to_query[temp_j]</code>"]
    F --> G{"Sequence Separation > 10?<br/><code>abs(query_i - query_j) > 10</code>"}
    G -->|No| H["Skip Close Residues"]
    G -->|Yes| I["Determine Atom Types<br/><code>query_sequence[query_i/j]</code>"]
    I --> J{"Glycine?<br/><code>query_sequence[query_i] == 'G'</code>"}
    J -->|Yes| K["Use CA Atom<br/><code>atom_name = 'CA'</code>"]
    J -->|No| L["Use CB Atom<br/><code>atom_name = 'CB'</code>"]
    K --> M["Create Constraint Dict<br/><code>constraint = {constraint_type: {...}}</code>"]
    L --> M
    M --> N["Add to Constraints List<br/><code>constraints.append(constraint)</code>"]
    N --> O{"More Pairs?"}
    E --> O
    H --> O
    O -->|Yes| C
    O -->|No| P["Remove Duplicates<br/><code>unique_constraints, seen_pairs</code>"]
    P --> Q["Return Unique Constraints<br/><code>return unique_constraints</code>"]
    
    style G fill:#fff8dc
    style J fill:#ffefd5
    style Q fill:#f0fff0
```

### 5. 중복 제거 알고리즘 (Deduplication Algorithm)

```mermaid
flowchart TD
    A["Constraints List<br/><code>constraints</code>"] --> B["Initialize Seen Pairs Set<br/><code>seen_pairs = set()</code>"]
    B --> C["Initialize Unique Constraints List<br/><code>unique_constraints = []</code>"]
    C --> D["For Each Constraint<br/><code>for constraint in constraints</code>"]
    D --> E["Extract Atom1 and Atom2 Info<br/><code>atom1_info = tuple(constraint[constraint_type]['atom1'])</code><br/><code>atom2_info = tuple(constraint[constraint_type]['atom2'])</code>"]
    E --> F{"Atom1 > Atom2?<br/><code>if atom1_info > atom2_info</code>"}
    F -->|Yes| G["Swap Atom1 and Atom2<br/><code>atom1_info, atom2_info = atom2_info, atom1_info</code>"]
    F -->|No| H["Keep Order"]
    G --> I["Create Pair Key<br/><code>pair_key = (atom1_info, atom2_info)</code>"]
    H --> I
    I --> J{"Pair Already Seen?<br/><code>if pair_key not in seen_pairs</code>"}
    J -->|Yes| K["Skip Constraint"]
    J -->|No| L["Add to Seen Pairs<br/><code>seen_pairs.add(pair_key)</code>"]
    L --> M["Add to Unique Constraints<br/><code>unique_constraints.append(constraint)</code>"]
    M --> N{"More Constraints?"}
    K --> N
    N -->|Yes| D
    N -->|No| O["Return Unique Constraints<br/><code>return unique_constraints</code>"]
    
    style F fill:#ffe4e1
    style J fill:#e6f3ff
    style O fill:#e8f5e8
```

## 매개변수 설정 (Parameter Configuration)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `distance_threshold` | 35.0 Å | 제약 조건 고려를 위한 최대 거리 |
| `cb_distance_cutoff` | 20.0 Å | Cβ-Cβ 거리 제약 조건 생성을 위한 최대 거리 |
| `min_sequence_identity` | 0.6 | 신뢰할 수 있는 정렬을 위한 최소 서열 동일성 |
| `gap_penalty` | -2.0 | 서열 정렬을 위한 갭 페널티 |

## 제약 조건 스키마 (Constraint Schema)

생성되는 제약 조건은 다음과 같은 Boltz 호환 형식을 가집니다:

```json
{
  "min_distance": {
    "atom1": ["A", 15, "CB"],
    "atom2": ["A", 42, "CB"],
    "distance": 8.547
  }
}
```

### 스키마 구성 요소:
- **atom1/atom2**: `[chain_id, residue_number, atom_name]`
- **chain_id**: 체인 식별자 (예: "A")
- **residue_number**: 1-기반 잔기 번호
- **atom_name**: 원자 이름 ("CB" 또는 글라이신의 경우 "CA")
- **distance**: 원자 간 거리 (Ångström, 소수점 3자리)

## 사용 예제 (Usage Examples)

### 기본 사용법

```python
from boltz.data.parse.template import TemplateConstraintGenerator

# 제약 조건 생성기 초기화
generator = TemplateConstraintGenerator(
    distance_threshold=35.0,
    cb_distance_cutoff=20.0,
    min_sequence_identity=0.6
)

# 템플릿 기반 제약 조건 생성
constraints = generator.generate_template_constraints(
    query_sequence="MKVLFVAS...",
    template_structure="template.pdb",
    template_chain_id="A",
    query_chain_id="A"
)
```

### Boltz 스키마와 통합

```python
from boltz.data.parse.template import apply_template_constraints

# 기존 스키마 데이터에 템플릿 제약 조건 추가
modified_schema = apply_template_constraints(
    schema_data=original_schema,
    template_structure="template.pdb",
    template_chain_id="A",
    target_chain_id="A"
)
```

## 에러 처리 및 경고 (Error Handling & Warnings)

시스템은 다음과 같은 상황에서 경고 또는 에러를 발생시킵니다:

1. **낮은 서열 동일성**: 설정된 임계값보다 낮은 서열 동일성
2. **좌표 추출 실패**: 템플릿 구조에서 Cβ 좌표 추출 실패
3. **체인 누락**: 지정된 체인 ID를 찾을 수 없음
4. **구조 파싱 에러**: PDB/mmCIF 파일 파싱 실패

## 성능 고려사항 (Performance Considerations)

- **메모리 사용량**: 큰 단백질의 경우 거리 맵이 O(n²) 메모리를 사용
- **계산 복잡도**: 거리 계산은 O(n²) 시간 복잡도
- **중복 제거**: 제약 조건 수에 비례하는 추가 처리 시간

## 제한사항 (Limitations)

1. **서열 길이**: 매우 긴 단백질에서는 메모리 사용량이 증가
2. **템플릿 품질**: 낮은 품질의 템플릿 구조는 부정확한 제약 조건 생성
3. **서열 동일성**: 낮은 서열 동일성에서는 제약 조건의 신뢰성 감소
4. **원자 타입**: 현재 Cβ/Cα 원자만 지원 