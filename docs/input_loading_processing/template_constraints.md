# Template-Based Distance Constraints

## 개요 (Overview)

Template-based distance constraint 시스템은 단일 원자 쌍 제약조건의 한계를 극복하기 위해 개발되었습니다. 
단백질 구조 예측에서 하나의 거리 제약조건만 적용할 경우 비물리적인 구조가 생성될 수 있는 문제를 해결합니다.

## 문제 해결 방법 (Solution Approach)

1. **Query Sequence와 Template Structure 정렬**
   - Needleman-Wunsch 알고리즘 사용
   - 시퀀스 동일성 검증

2. **Template에서 Cb-Cb Distance Map 계산**
   - Template 구조의 Cb 좌표 추출
   - 거리 임계값 내 residue pair 식별

3. **Multiple Distance Constraints 생성**
   - 정렬된 residue mapping 기반
   - 물리적으로 타당한 거리 제약조건들 생성

## 사용법 (Usage)

### YAML 파일 설정

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: "MADQLTEEQIAEFKEAFSLF"
      msa: empty
      template:
        structure: "path/to/template.pdb"
        chain_id: "A"

# 추가적인 explicit constraints도 함께 사용 가능
constraints:
  - min_distance:
      atom1: ["A", 10, "CA"]
      atom2: ["A", 50, "CA"]
      distance: 15.0
```

### 프로그래밍 인터페이스

```python
from boltz.data.parse.template import TemplateConstraintGenerator

# Initialize generator
generator = TemplateConstraintGenerator(
    cb_distance_cutoff=15.0,
    min_sequence_identity=0.3
)

# Generate constraints
constraints = generator.generate_template_constraints(
    query_sequence="MADQLTEEQIAEFKEAFSLF",
    template_structure="template.pdb",
    template_chain_id="A",
    query_chain_id="A"
)
```

## 주요 특징 (Key Features)

### 1. 시퀀스 정렬 기반 매핑
- **High-quality alignment**: 시퀀스 동일성 검증
- **Gap handling**: 정렬 gap 처리
- **Confidence scoring**: 정렬 신뢰도 평가

### 2. 거리 제약조건 필터링
- **Distance cutoff**: 15Å 이내 residue pairs만 사용
- **Sequence separation**: 4 잔기 이상 떨어진 쌍만 선택
- **Duplicate removal**: 중복 제약조건 자동 제거

### 3. 품질 관리
- **Sequence identity threshold**: 최소 30% 동일성 요구
- **Cb coordinate validation**: 좌표 추출 검증
- **Error handling**: 실패 시 graceful degradation

## 파라미터 설정 (Parameters)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `cb_distance_cutoff` | 15.0 | Cb-Cb 거리 임계값 (Å) |
| `min_sequence_identity` | 0.3 | 최소 시퀀스 동일성 |
| `gap_penalty` | -2.0 | 정렬 gap penalty |
| `distance_threshold` | 20.0 | 최대 고려 거리 (Å) |

## 시스템 통합 (System Integration)

Template constraints는 Boltz schema 파싱 과정에서 자동으로 처리됩니다:

1. **Sequences 파싱**: Template 정보 수집
2. **Template constraints 생성**: 각 chain별 처리
3. **Explicit constraints 병합**: 사용자 정의 제약조건과 통합
4. **최종 Structure 생성**: 모든 제약조건 적용

## 성능 고려사항 (Performance Considerations)

- **메모리 사용량**: Template 구조 크기에 비례
- **계산 시간**: 시퀀스 길이의 제곱에 비례
- **I/O 오버헤드**: Template 파일 읽기

## 제한사항 (Limitations)

1. **Template 의존성**: 고품질 template 구조 필요
2. **시퀀스 유사성**: 낮은 동일성 시 신뢰도 저하
3. **Cb 원자 의존**: Glycine의 경우 CA 사용

## 예상 결과 (Expected Results)

- **구조적 일관성**: 전역적으로 일관된 거리 패턴
- **물리적 타당성**: Template 기반 현실적 제약조건
- **과제약 방지**: 적절한 개수의 제약조건 생성

## 문제 해결 (Troubleshooting)

### 일반적인 문제들

1. **Template 파일 로드 실패**
   ```
   [WARNING] Failed to extract Cb coordinates: [error message]
   ```
   - 파일 경로 확인
   - 파일 형식 검증 (PDB/mmCIF)

2. **낮은 시퀀스 동일성**
   ```
   Low sequence identity (0.15). Constraints may not be reliable.
   ```
   - Template 변경 고려
   - 임계값 조정 (`min_sequence_identity`)

3. **제약조건 생성 실패**
   ```
   [WARNING] Failed to generate template constraints: [error]
   ```
   - 시퀀스 정렬 확인
   - Template chain ID 검증

### 디버깅 옵션

```python
import warnings
warnings.simplefilter("always")  # 모든 warning 표시

generator = TemplateConstraintGenerator(
    min_sequence_identity=0.1  # 임계값 낮춤
)
``` 