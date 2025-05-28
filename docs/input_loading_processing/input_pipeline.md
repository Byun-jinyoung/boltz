# Boltz-1x Input Loading & Processing Pipeline

이 문서는 Boltz-1x 모델의 전체 입력 처리 파이프라인을 체계적으로 설명합니다. YAML 설정 파일부터 최종 모델 예측까지의 모든 단계를 시각화하고 상세히 해설합니다.

## 전체 파이프라인 개요

```mermaid
flowchart TD
    %% Input Processing Phase
    A[YAML/FASTA Input Files] --> B[parse_yaml/parse_fasta<br/>(src/boltz/data/parse/yaml.py<br/>src/boltz/data/parse/fasta.py)]
    B --> C[parse_boltz_schema<br/>(src/boltz/data/parse/schema.py)]
    C --> D[Target Object<br/>(Structure, ResidueConstraints, Record)]
    
    %% MSA Generation Phase
    D --> E[process_inputs<br/>(src/boltz/main.py)]
    E --> F{MSA Required?}
    F -->|Yes| G[compute_msa<br/>(MMSeqs2 Server)]
    F -->|No| H[Skip MSA Generation]
    G --> I[parse_a3m/parse_csv<br/>(src/boltz/data/parse/a3m.py<br/>src/boltz/data/parse/csv.py)]
    H --> I
    I --> J[Save NPZ Files<br/>(structures/, msa/, constraints/)]
    J --> K[Generate manifest.json<br/>(Record metadata)]
    
    %% Data Loading Phase`
    K --> L[BoltzInferenceDataModule<br/>(src/boltz/data/module/inference.py)]
    L --> M[PredictionDataset.__getitem__<br/>(load_input function)]
    M --> N[Input Object<br/>(Structure, MSA, ResidueConstraints)]
    
    %% Tokenization & Featurization Phase
    N --> O[BoltzTokenizer.tokenize<br/>(src/boltz/data/tokenize/boltz.py)]
    O --> P[Tokenized Object<br/>(token_data, token_bonds)]
    P --> Q[BoltzCropper.crop<br/>(src/boltz/data/feature/cropper.py)]
    Q --> R[BoltzFeaturizer.process<br/>(src/boltz/data/feature/featurizer.py)]
    R --> S[Feature Dict<br/>(token, atom, msa features)]
    
    %% Data Module Phase
    S --> T[collate function<br/>(batch processing)]
    T --> U[DataLoader<br/>(batched tensors)]
    U --> V[transfer_batch_to_device<br/>(GPU/CPU transfer)]
    
    %% Model Forward Phase
    V --> W[Boltz1.forward<br/>(src/boltz/model/model.py)]
    W --> X[InputEmbedder<br/>(src/boltz/model/modules/trunk.py)]
    X --> Y[MSAModule<br/>(src/boltz/model/modules/trunk.py)]
    Y --> Z[PairformerModule<br/>(src/boltz/model/modules/trunk.py)]
    Z --> AA[DistogramModule<br/>(src/boltz/model/modules/trunk.py)]
    AA --> BB[AtomDiffusion.sample<br/>(src/boltz/model/modules/diffusion.py)]
    BB --> CC[ConfidenceModule<br/>(src/boltz/model/modules/confidence.py)]
    CC --> DD[Prediction Output<br/>(coordinates, confidence scores)]

    %% Styling
    classDef inputPhase fill:#e1f5fe
    classDef processPhase fill:#f3e5f5
    classDef dataPhase fill:#e8f5e8
    classDef modelPhase fill:#fff3e0
    
    class A,B,C,D inputPhase
    class E,F,G,H,I,J,K processPhase
    class L,M,N,O,P,Q,R,S,T,U,V dataPhase
    class W,X,Y,Z,AA,BB,CC,DD modelPhase
```

## 주요 데이터 타입 변환 흐름

| 단계 | 입력 타입 | 출력 타입 | 설명 |
|------|-----------|-----------|------|
| YAML Parsing | `dict` (YAML schema) | `Target` (dataclass) | Structure, ResidueConstraints, Record 객체 생성 |
| Input Processing | `Target` | `NPZ files + manifest.json` | 구조/MSA/제약조건을 npz 형태로 직렬화 |
| Data Loading | `Record` (from manifest) | `Input` (dataclass) | NPZ 파일들을 메모리로 로딩 |
| Tokenization | `Input` | `Tokenized` (dataclass) | 원자/잔기를 토큰 단위로 변환 |
| Featurization | `Tokenized` | `dict[str, Tensor]` | 모델 입력을 위한 텐서 피처 생성 |
| Model Forward | `dict[str, Tensor]` | `dict[str, Tensor]` | 구조 예측 및 신뢰도 점수 출력 |

## 단계별 상세 설명

### 1. YAML/FASTA 파싱 단계
**파일**: `src/boltz/data/parse/schema.py`의 `parse_boltz_schema`
**역할**: 원시 입력을 내부 데이터 구조로 변환

```python
# Input: YAML schema dict
schema = {
    "sequences": [
        {"protein": {"id": "A", "sequence": "MADQLT..."}},
        {"ligand": {"id": "B", "smiles": "CC1=CC=CC=C1"}}
    ],
    "constraints": [...]
}

# Output: Target object containing Structure, ResidueConstraints, Record
target = parse_boltz_schema(name="example", schema=schema, ccd=ccd_components)
```

**핵심 처리 과정**:
- `sequences` 블록을 순회하며 entity 타입별로 그룹화 (protein/DNA/RNA vs ligand)
- Polymer: `parse_polymer` 호출 → `ParsedChain` 생성
- Non-polymer: `parse_ccd_residue` 호출 → `ParsedResidue` 생성
- Template 제약: `TemplateConstraintGenerator`로 거리 제약 생성
- 원자/잔기/체인 테이블을 numpy structured array로 변환

### 2. 입력 전처리 단계
**파일**: `src/boltz/main.py`의 `process_inputs`
**역할**: NPZ 파일 및 manifest 생성

```python
# For each target, save processed data
target.structure.dump(structure_dir / f"{target_id}.npz")
target.residue_constraints.dump(constraints_dir / f"{target_id}.npz")
msa.dump(msa_dir / f"{target_id}_{msa_idx}.npz")

# Create manifest with Record metadata
manifest = Manifest(records)
manifest.dump(out_dir / "processed" / "manifest.json")
```

**핵심 처리 과정**:
- MSA 필요 시 MMSeqs2 서버를 통한 자동 생성
- A3M/CSV → `parse_a3m`/`parse_csv` → MSA 객체 → NPZ 저장
- 각 Target의 Structure/ResidueConstraints를 NPZ로 직렬화
- Record 메타데이터를 manifest.json으로 저장

### 3. 데이터 로딩 단계
**파일**: `src/boltz/data/module/inference.py`
**역할**: NPZ 파일들을 메모리로 로딩하고 배치 생성

```python
def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
    # Load structure from NPZ
    structure = Structure.load(target_dir / f"{record.id}.npz")
    
    # Load MSAs for each chain
    msas = {}
    for chain in record.chains:
        if chain.msa_id != -1:
            msa = MSA.load(msa_dir / f"{chain.msa_id}.npz")
            msas[chain.chain_id] = msa
    
    return Input(structure, msas, residue_constraints)
```

**핵심 처리 과정**:
- `PredictionDataset.__getitem__`에서 Record별로 `load_input` 호출
- NPZ 파일들을 Structure/MSA/ResidueConstraints 객체로 역직렬화
- `collate` 함수로 리스트를 배치 텐서로 변환
- `DataLoader`를 통한 효율적인 배치 처리

### 4. 토큰화 단계
**파일**: `src/boltz/data/tokenize/boltz.py`의 `BoltzTokenizer`
**역할**: 원자/잔기를 토큰 단위로 변환

```python
class BoltzTokenizer:
    def tokenize(self, data: Input) -> Tokenized:
        # Standard residues → one token per residue
        # Non-standard residues → one token per atom
        token_data = []
        for chain in data.structure.chains:
            for res in chain.residues:
                if res["is_standard"]:
                    # Create single token for residue
                    token = TokenData(token_idx=token_idx, ...)
                else:
                    # Create token for each atom
                    for atom in res.atoms:
                        token = TokenData(token_idx=token_idx, ...)
```

**핵심 처리 과정**:
- 표준 잔기: 잔기당 1개 토큰 (center/disto atom 정의)
- 비표준 잔기: 원자당 1개 토큰
- 토큰 간 결합 정보 (`token_bonds`) 생성
- 원자 인덱스 → 토큰 인덱스 매핑 테이블 구축

### 5. 피처화 단계
**파일**: `src/boltz/data/feature/featurizer.py`의 `BoltzFeaturizer`
**역할**: 모델 입력을 위한 텐서 피처 생성

```python
class BoltzFeaturizer:
    def process(self, data: Tokenized) -> dict[str, Tensor]:
        # Token features: residue type, position, chain info
        token_features = process_token_features(data)
        
        # Atom features: coordinates, element, charge
        atom_features = process_atom_features(data)
        
        # MSA features: sequence alignment, deletion info
        msa_features = process_msa_features(data)
        
        return {**token_features, **atom_features, **msa_features}
```

**핵심 피처 타입**:
- **Token features**: `res_type`, `asym_id`, `entity_id`, `token_bonds`
- **Atom features**: `ref_pos`, `ref_element`, `atom_to_token`, `frames_idx`
- **MSA features**: `msa`, `msa_paired`, `deletion_value`, `profile`
- **Constraint features**: `rdkit_bounds_index`, `min_distance_atom_index`

### 6. 모델 실행 단계
**파일**: `src/boltz/model/model.py`의 `Boltz1.forward`
**역할**: 피처 텐서를 입력받아 구조 예측 수행

```python
def forward(self, feats: dict[str, Tensor]) -> dict[str, Tensor]:
    # Embed input features
    s_inputs, s, z = self.input_embedder(feats)
    
    # Process MSA information
    s, z = self.msa_module(s, z, feats)
    
    # Pairformer attention blocks
    s, z = self.pairformer_module(s, z)
    
    # Distance prediction
    pdistogram = self.distogram_module(z)
    
    # Structure generation via diffusion
    structure_out = self.structure_module.sample(s, z, feats)
    
    # Confidence prediction
    confidence_out = self.confidence_module(s, z, feats)
    
    return {**structure_out, **confidence_out, "pdistogram": pdistogram}
```

**주요 모듈 역할**:
- **InputEmbedder**: 토큰/원자/MSA 피처를 임베딩 벡터로 변환
- **MSAModule**: MSA 정보를 이용한 single/pair representation 업데이트
- **PairformerModule**: Attention 기반 representation 정제
- **DistogramModule**: 토큰 간 거리 분포 예측
- **AtomDiffusion**: Diffusion process를 통한 3D 좌표 생성
- **ConfidenceModule**: 예측 신뢰도 점수 계산

## 주요 최적화 및 특징

### 메모리 효율성
- **Cropping**: 큰 구조를 모델이 처리 가능한 크기로 분할
- **Padding**: 배치 처리를 위한 동적 패딩
- **Gradient Checkpointing**: 메모리 사용량 최적화

### 성능 최적화
- **Trifast Attention**: GPU에서 빠른 attention 연산
- **Model Compilation**: PyTorch 2.0 compile 기능 활용
- **Mixed Precision**: 16-bit 연산으로 속도 향상

### 제약 조건 처리
- **Template Constraints**: 기존 구조를 참조한 거리 제약
- **Chemical Constraints**: RDKit 기반 화학적 제약
- **User Constraints**: 사용자 정의 거리/결합 제약

## 에러 핸들링 및 주의사항

### 일반적인 문제점
1. **MSA 생성 실패**: `--use_msa_server` 플래그 확인
2. **메모리 부족**: `max_tokens`, `max_atoms` 값 조정
3. **CUDA OOM**: `diffusion_samples` 값 감소

### 디버깅 팁
- `manifest.json`에서 Record 메타데이터 확인
- NPZ 파일 크기로 데이터 로딩 상태 점검
- Tensorboard 로그로 학습/추론 과정 모니터링

이 파이프라인을 통해 Boltz-1x는 복잡한 생분자 구조를 정확하고 효율적으로 예측할 수 있습니다.
