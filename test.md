# 1. 몇가지 테스트를 통해 얻은 생각
## A. 
 - Cb distance cutoff: C$_{\bi}$ -- C$_{\bj}$ 사이의 minimum distance을 의미 -> 이게 커지면 다른 영역들이 적당해지기 전에 얘만 늘어나버려서 non-physical 구조가 나와버림
 - 단지 하나의 atom i-j pair에 대해서만 constraint을 주면 거기만 늘어난 이상한 구조가 예측되어서 여러 개의 (atom i-j) restraints을 사용해야 함
 - k도 중요
 - 따라서 아래와 같이 test을 수행

## B.
 - Cb distance cutoff: TemplateGenerator
 - k: MinDistancePotential k -> 모든 constraint에 대해 동일하게 적용
 - 다른 조건 동일
 - 괄호 안 숫자: RMSD from 6be6 or 8esv calculated from chimeraX mmaker
 - O/X: 눈으로 보기에 physical/non-physcial

| Cb distance cutoff / k | 5                 | 15                | 30                | 100 |
|------------------------| :-----------------|-------------------|-------------------|-----|
| 1                      | closed (0.743, O) | closed (1.014, O) | closed (1.XXX, O) |     |
| 2                      |                   |                   |                   |     |
| 5                      |                   |                   | closed (X.XXX, X) |     |
| 10                     | closed (0.951, O) | closed (  )       |                   | X (그나마 나음)   |
| 50                     |                   |                   |                   |     |
| 100                    | closed (0.764, O) | X                 | X                 | X   |


 - k100: 너무 큰 감이 있는거 같음
 - d5: 너무 작아서 closed form만 나옴 -> 오히려 d100이 무너지지만 원하는 '방향'으로는 나옴    