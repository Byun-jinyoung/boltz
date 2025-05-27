```  
```  
## 단계별 상세 설명  

1. parse_boltz_schema (schema.py)  
   - 입력 YAML을 로드하고 schema dict 생성  
   - sequences 블록별 entity grouping (protein/DNA/RNA vs ligand)  
   - parse_polymer 또는 parse_ccd_residue 호출하여 ParsedChain/ParsedResidue 객체 생성  
   - template 정보가 있으면 TemplateConstraintGenerator 호출하여 min_distance 제약 생성  
   - 최종적으로 Structure, ResidueConstraints 리스트를 포함하는 Target 객체 반환  

2. process_inputs (main.py)  
   - Target 객체를 기반으로 MSA, structure, constraint NPZ 파일 작성  
   - 각 레코드별 .npz 파일과 manifest.json 구성  

3. load_input(record) (inference.py)  
   - Record에서 id, chain 정보 읽어옴  
   - 로컬 target_dir/.npz 파일 로드: structure, msa(npz 로드), constraint(npz 로드)  
   - Input dataclass에 Structure, MSA, ResidueConstraints 인스턴스 바인딩  

4. PredictionDataset.__getitem__ (inference.py)  
   - manifest에서 레코드 순회  
   - load_input 호출 후 리턴된 Input 객체로부터 tensor batch 구성  

5. collate(data) (inference.py)  
   - list[dict(tensors)]를 하나의 배치 dict로 변환  
   - 패딩, 마스크, stacking 기능 수행  

6. BoltzInferenceDataModule.predict_dataloader (inference.py)  
   - PredictionDataset으로 DataLoader 생성  
   - batch_size=1, collate_fn=collate, num_workers 설정  

7. transfer_batch_to_device (inference.py)  
   - 'record'나 메타데이터 제외한 모든 tensor를 GPU/CPU로 이동  

8. Boltz1.forward (model/model.py)  
   - feats dict 수신  
   - InputEmbedder→MSAModule→PairformerModule→DistogramModule→AtomDiffusion 혹은 ConfidenceModule 순으로 연산  
   - 최종 예측 tensor dict 출력  

```  
```  
### I/O Data Types 간단 설명
- YAML Schema (dict): raw input configuration
- Target (dataclass): Structure, ResidueConstraints 저장
- NPZ files: StructureData(np.ndarray), MSA(msa arrays), Constraints(dict)
- Record → Input (dataclass): Structure, MSA, ResidueConstraints → raw tensors
- Tokenized (dataclass): token_data(np.ndarray of Token), token_bonds
- feats (dict[str, Tensor]): model 입력 토치 텐서
- out (dict[str, Tensor]): 예측된 거리/좌표 등 출력 tensors

```mermaid
flowchart TD
    H --> I1[BoltzTokenizer.tokenize<br/>(src/boltz/data/tokenize/boltz.py)]
    I1 --> I2[BoltzCropper.crop<br/>(src/boltz/data/feature/cropper.py)]
    I2 --> I3[BoltzFeaturizer.featurize<br/>(src/boltz/data/feature/featurizer.py)]
    I3 --> I[Boltz1.forward(feats)<br/>(src/boltz/model/model.py)]
```

## Chain-of-Thought: 문서 보강 제안
1. **모듈별 시퀀스 다이어그램**: 각 단계 내부(`parse_boltz_schema`, `process_inputs`, `load_input` 등) 호출 관계를 세부적으로 나타내면 이해도가 올라감.  
2. **데이터 Shape 테이블**: 각 I/O 변수(tensors, arrays)의 차원·dtype을 표로 정리하여 모델이 요구하는 형식을 명확히 할 것.  
3. **예시 코드 스니펫**: 주요 함수(`parse_boltz_schema` 등) 입출력 예시를 짤막하게 포함하면 독자가 따라 하기 편리.  
4. **에러 핸들링 섹션**: 파이프라인 중 발생 가능한 워닝/예외 유형과 대처 방안을 별도 챕터로 추가 권장.
