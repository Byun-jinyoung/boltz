# Boltz YAML Parsing Analysis

이 문서는 `boltz` 프로젝트에서 YAML 입력 파일이 어떻게 파싱되고 메타데이터 객체들이 생성되는지를 상세히 분석합니다. 특히 Target, Record, StructureInfo, ChainInfo 등 핵심 메타데이터 객체의 생성 과정에 초점을 맞춥니다.

## YAML 파싱 전체 플로우

```mermaid
flowchart TD
    A[YAML 파일 입력] --> B[parse_yaml 함수<br/>src/boltz/data/parse/yaml.py]
    B --> C[yaml.safe_load<br/>YAML 데이터 로드]
    C --> D[parse_boltz_schema 함수<br/>src/boltz/data/parse/schema.py]
    
    D --> E[Version 검증<br/>version == 1]
    E --> F[Entity 그룹화<br/>동일한 시퀀스별로 그룹화]
    
    F --> G{Entity 타입별 처리}
    G -->|protein/dna/rna| H[Polymer 처리<br/>parse_polymer 함수]
    G -->|ligand + ccd| I[CCD Ligand 처리<br/>parse_ccd_residue 함수]
    G -->|ligand + smiles| J[SMILES Ligand 처리<br/>RDKit 분자 생성]
    
    H --> K[ParsedChain 객체 생성]
    I --> K
    J --> K
    
    K --> L[Constraints 처리<br/>bond, pocket 제약조건]
    L --> M[Raw 데이터 수집<br/>atoms, bonds, residues, chains]
    
    M --> N[NumPy 배열 변환<br/>Structure 타입으로 변환]
    N --> O[메타데이터 객체 생성<br/>Record, ChainInfo, StructureInfo]
    
    O --> P[Target 객체 생성<br/>최종 결과물]

    %% 스타일링
    classDef input fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A input
    class P output
    class E,F,G decision
    class B,C,D,H,I,J,K,L,M,N,O process
```

## 주요 단계별 상세 분석

### 1. YAML 스키마 구조 이해

YAML 입력 파일은 다음과 같은 구조를 가집니다:

```yaml
version: 1
sequences:
    - protein:
        id: A
        sequence: "MADQLTEEQIAEFKEAFSLF"
        msa: path/to/msa1.a3m
        modifications:
          - position: 5
            ccd: MSE
        cyclic: false
    - ligand:
        id: B
        smiles: "CC1=CC=CC=C1"
    - ligand:
        id: C
        ccd: ATP
constraints:
    - bond:
        atom1: [A, 1, CA]
        atom2: [A, 2, N]
    - pocket:
        binder: B
        contacts: [[A, 1], [A, 2]]
```

### 2. Entity 그룹화 과정

```mermaid
flowchart LR
    A[YAML sequences 섹션] --> B[Entity Type 추출<br/>protein/dna/rna/ligand]
    B --> C[Sequence 추출<br/>amino acid/nucleotide/smiles/ccd]
    C --> D{동일한 타입과<br/>시퀀스 확인}
    D -->|예| E[기존 그룹에 추가<br/>items_to_group]
    D -->|아니오| F[새 그룹 생성<br/>items_to_group]
    E --> G[Entity ID 할당]
    F --> G
    G --> H[MSA 정보 처리<br/>protein만 해당]

    %% 스타일링
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class B,C,H process
    class D decision
    class E,F,G output
```

### 3. 메타데이터 객체 생성 과정

```mermaid
flowchart TD
    A[ParsedChain 객체들] --> B[Chain 데이터 수집<br/>chain_data 리스트]
    B --> C[Residue 데이터 수집<br/>res_data 리스트]
    C --> D[Atom 데이터 수집<br/>atom_data 리스트]
    D --> E[Bond 데이터 수집<br/>bond_data 리스트]
    
    E --> F[Constraints 처리<br/>connections, pocket_binders]
    F --> G[NumPy 배열 변환<br/>Atom, Bond, Residue, Chain 타입]
    
    G --> H[Structure 객체 생성<br/>모든 구조 데이터 포함]
    H --> I[StructureInfo 생성<br/>num_chains 등 메타데이터]
    
    I --> J[ChainInfo 리스트 생성<br/>각 체인별 메타데이터]
    J --> K[InferenceOptions 생성<br/>pocket binder 정보]
    K --> L[Record 객체 생성<br/>모든 메타데이터 통합]
    
    L --> M[ResidueConstraints 생성<br/>기하학적 제약조건]
    M --> N[Target 객체 생성<br/>최종 결과물]

    %% 서브그래프
    subgraph "Raw Data Collection"
        B
        C
        D
        E
    end
    
    subgraph "NumPy Array Conversion"
        F
        G
        H
    end
    
    subgraph "Metadata Object Creation"
        I
        J
        K
        L
        M
        N
    end

    %% 스타일링
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef convert fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef metadata fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class B,C,D,E data
    class F,G,H convert
    class I,J,K,L,M,N metadata
```

## 핵심 객체 구조 분석

### Target 객체 구조

Target 객체는 YAML 파싱의 최종 결과물로, 다음과 같은 구조를 가집니다:

```python
@dataclass(frozen=True)
class Target:
    """Target datatype."""
    
    record: Record                                    # 메타데이터 정보
    structure: Structure                              # 구조 데이터 (atoms, bonds, etc.)
    sequences: Optional[dict[str, str]] = None        # Entity별 시퀀스 정보
    residue_constraints: Optional[ResidueConstraints] = None  # 기하학적 제약조건
```

### Record 객체 구조

Record 객체는 처리된 데이터의 핵심 메타데이터를 포함합니다:

```python
@dataclass(frozen=True)
class Record(JSONSerializable):
    """Record datatype."""
    
    id: str                                          # 파일명에서 추출된 고유 ID
    structure: StructureInfo                         # 구조 메타데이터
    chains: list[ChainInfo]                          # 체인별 상세 정보
    interfaces: list[InterfaceInfo]                  # 인터페이스 정보 (YAML에서는 빈 리스트)
    inference_options: Optional[InferenceOptions] = None  # 추론 옵션 (pocket 정보)
```

### ChainInfo 객체 구조

각 체인의 상세 정보를 담고 있습니다:

```python
@dataclass(frozen=True)
class ChainInfo:
    """ChainInfo datatype."""
    
    chain_id: int                                    # 0부터 시작하는 체인 인덱스 (asym_id)
    chain_name: str                                  # YAML에서 지정한 체인 이름 (A, B, C, ...)
    mol_type: int                                    # 분자 타입 (protein=1, dna=2, rna=3, ligand=5)
    cluster_id: Union[str, int]                      # 클러스터 ID (-1로 초기화)
    msa_id: Union[str, int]                          # MSA 파일 경로 또는 ID
    num_residues: int                                # 잔기 수
    valid: bool = True                               # 유효성 플래그
    entity_id: Optional[Union[str, int]] = None      # 엔티티 ID
```

### Structure 객체 구조

실제 원자 및 결합 정보를 담고 있는 NumPy 배열들:

```python
@dataclass(frozen=True)
class Structure(NumpySerializable):
    """Structure datatype."""
    
    atoms: np.ndarray                                # Atom 타입의 배열
    bonds: np.ndarray                                # Bond 타입의 배열
    residues: np.ndarray                             # Residue 타입의 배열
    chains: np.ndarray                               # Chain 타입의 배열
    connections: np.ndarray                          # Connection 타입의 배열 (YAML constraints)
    interfaces: np.ndarray                           # Interface 타입의 배열 (빈 배열)
    mask: np.ndarray                                 # 체인 마스크 배열
```

## Entity별 특수 처리 과정

### Protein/DNA/RNA 처리

```mermaid
flowchart TD
    A[Polymer Entity<br/>protein/dna/rna] --> B[Token Map 선택<br/>prot/dna/rna_letter_to_token]
    B --> C[Sequence 토큰화<br/>amino acid/nucleotide → token]
    C --> D{Modifications<br/>있는가?}
    D -->|예| E[Modified Residue 적용<br/>position, ccd 정보 사용]
    D -->|아니오| F[parse_polymer 함수 호출]
    E --> F
    F --> G[Standard Residue 처리<br/>CCD 컴포넌트 사용]
    G --> H[ParsedResidue 리스트 생성]
    H --> I[ParsedChain 객체 생성<br/>cyclic_period 설정]

    %% 스타일링
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class B,C,E,F,G,H process
    class D decision
    class I output
```

### Ligand 처리 (CCD)

```mermaid
flowchart TD
    A[Ligand Entity<br/>ccd 코드] --> B[CCD Dictionary 조회<br/>ccd 매개변수 사용]
    B --> C{CCD 컴포넌트<br/>존재하는가?}
    C -->|예| D[parse_ccd_residue 함수<br/>CCD Mol 객체 처리]
    C -->|아니오| E[Error 발생<br/>CCD component not found]
    D --> F[RDKit Mol 처리<br/>3D conformer 생성]
    F --> G[Geometry Constraints 계산<br/>bounds, chiral, stereo, planar]
    G --> H[ParsedResidue 생성<br/>원자, 결합, 제약조건 포함]
    H --> I[ParsedChain 객체 생성<br/>NONPOLYMER 타입]

    %% 스타일링
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class B,D,F,G,H process
    class C decision
    class I output
    class E error
```

### Ligand 처리 (SMILES)

```mermaid
flowchart TD
    A[Ligand Entity<br/>smiles 문자열] --> B[RDKit MolFromSmiles<br/>분자 객체 생성]
    B --> C{분자 생성<br/>성공했는가?}
    C -->|예| D[compute_3d_conformer<br/>ETKDG 알고리즘 사용]
    C -->|아니오| E[Error 발생<br/>Invalid SMILES]
    D --> F{3D 좌표<br/>생성 성공?}
    F -->|예| G[Geometry Constraints 계산]
    F -->|아니오| H[Warning 출력<br/>기본 좌표 사용]
    G --> I[ParsedResidue 생성]
    H --> I
    I --> J[ParsedChain 객체 생성<br/>NONPOLYMER 타입]

    %% 스타일링
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef warning fill:#fffde7,stroke:#f57f17,stroke-width:2px
    
    class B,D,G,I process
    class C,F decision
    class J output
    class E error
    class H warning
```

## Constraints 처리

### Bond Constraints

YAML의 `constraints` 섹션에서 정의된 공유결합을 처리합니다:

```yaml
constraints:
    - bond:
        atom1: [A, 1, CA]  # [chain_name, residue_index, atom_name]
        atom2: [A, 2, N]
```

이는 다음과 같이 처리됩니다:

1. **Atom Index Mapping**: `(chain_name, residue_idx, atom_name)` → `(asym_id, res_idx, atom_idx)`
2. **Connection 배열 생성**: `(chain1, chain2, res1, res2, atom1, atom2)` 형태의 튜플
3. **NumPy 배열 변환**: `Connection` 타입의 배열로 변환

### Pocket Constraints

Ligand binding pocket 정보를 처리합니다:

```yaml
constraints:
    - pocket:
        binder: B           # 결합하는 체인 (ligand)
        contacts: [[A, 1], [A, 2]]  # 접촉하는 잔기들
```

이는 `InferenceOptions` 객체로 변환되어 나중에 모델 추론 시 사용됩니다.

## 데이터 변환 플로우

```mermaid
flowchart LR
    A[YAML Raw Data] --> B[Python Dict<br/>yaml.safe_load]
    B --> C[ParsedChain Objects<br/>중간 표현]
    C --> D[Raw Data Lists<br/>atom_data, bond_data, etc.]
    D --> E[NumPy Arrays<br/>structured arrays]
    E --> F[Structure Object<br/>NumpySerializable]
    F --> G[Target Object<br/>최종 결과물]

    %% 서브그래프
    subgraph "Parsing Stage"
        A
        B
        C
    end
    
    subgraph "Data Collection Stage"
        D
        E
    end
    
    subgraph "Object Creation Stage"
        F
        G
    end

    %% 스타일링
    classDef parse fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef collect fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef create fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class A,B,C parse
    class D,E collect
    class F,G create
```

## 코드 분석 결과 요약

### 주요 발견사항

1. **Modular Architecture**: YAML 파싱은 여러 단계의 중간 객체(`ParsedChain`, `ParsedResidue`, `ParsedAtom`)를 거쳐 최종 `Target` 객체로 변환됩니다.

2. **Entity Grouping**: 동일한 시퀀스를 가진 체인들은 하나의 entity로 그룹화되어 처리되며, 이는 메모리 효율성을 높입니다.

3. **Constraint Processing**: YAML에서 정의된 제약조건들은 별도로 처리되어 `ResidueConstraints`와 `InferenceOptions`로 분리됩니다.

4. **RDKit Integration**: 리간드 처리에는 RDKit이 광범위하게 사용되며, 3D 좌표 생성과 기하학적 제약조건 계산이 자동으로 수행됩니다.

### 코드 구조의 장점

- **Type Safety**: 강타입 시스템과 dataclass를 사용하여 데이터 무결성을 보장합니다.
- **Serialization Support**: `NumpySerializable`과 `JSONSerializable` 믹스인을 통해 데이터 저장/로딩이 용이합니다.
- **Extensibility**: 새로운 entity 타입이나 제약조건을 쉽게 추가할 수 있는 구조입니다.

### 분석 방향 제안

추가적인 코드 분석을 위해서는 다음 방향을 고려할 수 있습니다:

1. **Performance Profiling**: 큰 분자 복합체에 대한 파싱 성능 분석
2. **Error Handling**: 다양한 잘못된 입력에 대한 에러 처리 메커니즘 분석
3. **Memory Usage**: 대규모 시스템에서의 메모리 사용 패턴 분석
4. **Validation**: 입력 데이터 검증 로직의 완전성 확인

이러한 분석을 통해 `boltz`의 YAML 파싱 시스템에 대한 포괄적인 이해를 얻을 수 있습니다. 