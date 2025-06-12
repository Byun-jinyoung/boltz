# Boltz-2 Template 정보 처리 흐름도

## Template Processing Flow Chart

```mermaid
graph TD
    A[Input YAML File] --> B[parse_boltz_schema]
    
    B --> C{Check Templates Section}
    C -->|Has templates| D[Parse Template Information]
    C -->|No templates| E[Load Dummy Template Features]
    
    D --> F[Template Processing Flow]
    F --> F1[Parse mmCIF Template File]
    F --> F2[Extract Template Sequences]
    F --> F3[Sequence Alignment<br/>get_local_alignments]
    F --> F4[Create TemplateInfo Records]
    
    F4 --> G[Template Feature Extraction]
    G --> G1[process_template_features]
    G1 --> G2[Extract CB/CA Coordinates]
    G1 --> G3[Extract Frame Information<br/>frame_rot, frame_t]
    G1 --> G4[Extract Residue Types]
    G1 --> G5[Create Template Masks]
    
    G --> H[Template Features Dict]
    H --> H1[template_restype: Residue types]
    H --> H2[template_frame_rot: Frame rotations]
    H --> H3[template_frame_t: Frame translations]
    H --> H4[template_cb: CB coordinates]
    H --> H5[template_ca: CA coordinates]
    H --> H6[template_mask_cb: CB masks]
    H --> H7[template_mask_frame: Frame masks]
    H --> H8[template_mask: Overall masks]
    
    I[Template Constraint Generation<br/>Modified Code] --> I1[TemplateConstraintGenerator]
    I1 --> I2[Extract CB Coordinates from Template]
    I1 --> I3[Compute Distance Map]
    I1 --> I4[Map Template→Query Sequences]
    I1 --> I5[Generate NMR/MinDistance Constraints]
    I5 --> I6[Add to schema constraints]
    
    H --> J[Neural Network Model]
    J --> J1[TemplateModule/TemplateV2Module]
    J1 --> J2[Template Feature Processing]
    J2 --> J3[Compute Distogram from CB coords]
    J2 --> J4[Compute Unit Vectors in Frames]
    J2 --> J5[Concatenate Features + Residue Types]
    J2 --> J6[Project to Template Dimension]
    J2 --> J7[Pass through Pairformer Blocks]
    J2 --> J8[Apply Template Masks]
    J2 --> J9[Aggregate Multiple Templates]
    J2 --> J10[Project Back to Token Dimension]
    
    J10 --> K[Update Pairwise Embeddings z]
    K --> L[Continue Model Forward Pass]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style F fill:#fff3e0
    style G fill:#e8f5e8
    style H fill:#fce4ec
    style I fill:#fff8e1
    style J fill:#e3f2fd
    style K fill:#f1f8e9
```

## 흐름도 설명

### 입력 단계
- **Input YAML File**: 사용자가 제공하는 구조 예측 설정 파일
- **parse_boltz_schema**: YAML 파일을 파싱하여 내부 데이터 구조로 변환

### Template 정보 처리
- **Template Section Check**: YAML에 templates 섹션이 있는지 확인
- **Template Processing**: mmCIF 파일 파싱, 서열 정렬, TemplateInfo 객체 생성
- **Template Constraint Generation**: 새로운 기능으로 template 구조로부터 거리 제약조건 자동 생성

### Feature 추출
- **Template Feature Extraction**: Template 구조로부터 신경망 입력용 feature 추출
- **Feature Dict Creation**: 8가지 주요 template feature 생성

### 모델 적용
- **Neural Network Integration**: TemplateModule에서 feature 처리
- **Pairwise Embedding Update**: Template 정보를 pairwise embedding z에 통합
- **Structure Prediction**: 업데이트된 embedding으로 구조 예측 수행 