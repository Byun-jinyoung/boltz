graph TD
    A[YAML Input] --> B[parse_boltz_schema]
    B --> C{Entity Type}
    C -->|Ligand| D[parse_ccd_residue]
    C -->|Polymer| E[parse_polymer]
    
    D --> F[ParsedResidue with bonds]
    E --> G[ParsedResidue without bonds]
    
    F --> H[Structure.bonds array]
    G --> I[Structure with empty bonds for polymers]
    
    H --> J[BoltzTokenizer.tokenize]
    I --> J
    
    J --> K[token_bonds generation]
    K --> L[process_token_features]
    L --> M["Token bonds matrix for model"]
    
    subgraph "Ligand Processing"
        D1[RDKit molecule from SMILES/CCD]
        D2[Extract bonds from RDKit]
        D3[Create ParsedBond objects]
        D1 --> D2 --> D3 --> F
    end
    
    subgraph "Polymer Processing" 
        E1[Standard residues only]
        E2[No inter-atom bonds extracted]
        E3[Empty bonds list]
        E1 --> E2 --> E3 --> G
    end
    
    subgraph "Current Bond Constraint System"
        N[RDKit bounds constraints]
        O[Chiral atom constraints]
        P[Stereo bond constraints]
        Q[Planar bond constraints]
        N --> R[ResidueConstraints]
        O --> R
        P --> R
        Q --> R
    end
    
    style D fill:#e1f5fe
    style E fill:#fff3e0
    style F fill:#c8e6c9
    style G fill:#ffcdd2