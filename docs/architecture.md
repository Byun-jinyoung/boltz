# Boltz Architecture Overview

This document provides a high-level overview of the data processing and model architecture within the Boltz project, visualized using Mermaid.

```mermaid
flowchart TD

    A[Input Data] --> B[Data Loading & Preparation (DataModule)];
    B --> C[Tokenization (BoltzTokenizer)];
    C --> D[Cropping (BoltzCropper)];
    D --> E[Featurization (BoltzFeaturizer)];
    E --> F[Boltz Model (Boltz1)];
    F --> G[Output Data];

    B --> C; % Redundant, but makes the flow clearer sometimes
    C --> D;
    D --> E;
    E --> F;
    F --> G;


    subgraph Data Processing Pipeline
        B
        C
        D
        E
    end

    subgraph Model
        F
    end

    A -.-> MSA Data; % Optional MSA data input
    MSA Data --> E; % MSA data used in Featurization
    Input Data -.-> Residue Constraints; % Optional constraints input
    Residue Constraints --> E; % Constraints used in Featurization

    G --> H[Output Files];

    %% Styling for clarity (optional)
    classDef submodule fill:#f9f,stroke:#333,stroke-width:2px;
    class B,C,D,E submodule;
    class F submodule;


```

## Explanation of Components

*   **Input Data**: Raw biological data, likely in formats like PDB or mmCIF.
*   **MSA Data**: Multiple Sequence Alignment data, providing evolutionary context.
*   **Residue Constraints**: Optional constraints related to specific residues or atoms.
*   **Data Loading & Preparation (DataModule)**: Handles loading the raw data and preparing it for the processing pipeline, including managing datasets and dataloaders.
*   **Tokenization (BoltzTokenizer)**: Converts raw structural and sequence data into a tokenized representation.
*   **Cropping (BoltzCropper)**: Selects a relevant subset of the tokenized data, potentially focusing on specific chains or interfaces.
*   **Featurization (BoltzFeaturizer)**: Transforms the tokenized and cropped data into numerical features suitable for the neural network model.
*   **Boltz Model (Boltz1)**: The main deep learning model that takes features as input and predicts structural properties.
*   **Output Data**: The raw output from the model.
*   **Output Files**: Saved predictions in standard formats like PDB or mmCIF.

This diagram provides a simplified view. The actual data flow within the model (e.g., between the Input Embedder, MSA Module, Pairformer, Diffusion Module, etc.) is more complex and can be diagrammed separately if needed.

To view this diagram, open this markdown file in an editor that supports Mermaid rendering (like VS Code, GitHub, or a dedicated Mermaid viewer).

## Input Data Loading and Preparation

Data loading and preparation is handled by the DataModule, which reads the manifest and loads individual records. The following diagram details this process:

```mermaid
flowchart LR

    A[Manifest.json] --> B[Manifest 로드 (Manifest.load)];
    B --> C[각 Record 순회 (Dataset)];
    C --> D[Structure 데이터 로드 (.npz)];
    C --> E[MSA 데이터 로드 (.npz)];
    C --> F[Residue Constraints 로드 (Optional)];
    D, E, F --> G[Input 객체 생성];
    G --> H[Batch로 수집 (collate 함수)];
    H --> I[다음 단계 (토큰화/피처화)];

    subgraph Data Loading & Preparation
        B
        C
        D
        E
        F
        G
        H
    end

    %% Styling
    classDef file fill:#ccf,stroke:#333,stroke-width:2px;
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    class A file;
    class I process;
```

### 설명

*   **Manifest.json**: 처리할 데이터 항목들의 목록을 담고 있는 JSON 파일입니다.
*   **Manifest 로드**: Manifest 파일을 읽어 `Manifest` 객체로 로드합니다.
*   **각 Record 순회**: 로드된 Manifest에서 각 Record (데이터 항목)를 하나씩 가져와 처리합니다. 이 과정은 Dataset에서 이루어집니다.
*   **Structure 데이터 로드**: Record에 명시된 경로에서 단백질/핵산 등의 구조 데이터를 `.npz` 파일로부터 로드합니다.
*   **MSA 데이터 로드**: Record에 명시된 경로에서 Multiple Sequence Alignment (MSA) 데이터를 `.npz` 파일로부터 로드합니다.
*   **Residue Constraints 로드 (Optional)**: Record에 명시되어 있다면, 아미노산/원자 제약 조건 데이터를 로드합니다. 이는 선택 사항입니다.
*   **Input 객체 생성**: 로드된 Structure, MSA, Residue Constraints 데이터를 묶어 `Input` 객체를 생성합니다.
*   **Batch로 수집 (collate 함수)**: 여러 `Input` 객체를 모아 배치(Batch)를 구성하고, 필요한 경우 패딩(padding) 등을 수행합니다. 이는 DataModule의 collate 함수에서 처리됩니다.
*   **다음 단계 (토큰화/피처화)**: 준비된 데이터 배치는 토큰화 및 피처화 단계로 전달됩니다.

이 플로우차트는 데이터 로딩 및 준비 과정의 주요 단계를 보여줍니다. 실제 구현에서는 오류 처리 및 예외 상황에 대한 추가 로직이 포함될 수 있습니다. 