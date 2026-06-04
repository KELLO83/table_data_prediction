# 모델 후보군 설명

## 목적

이 문서는 현재 프로젝트의 테이블형 회귀 예측 실험 후보 12개를 한국어로 설명한다. 코드 기준 source of truth는 `ml/src/models/candidates.py`의 `EXPERIMENT_MODEL_GROUPS`이다.

해석 기준:

- 그룹 순서: 큰 실험 우선순위
- 그룹 내부 순서: 해당 계열 안에서 먼저 돌릴 추천 순위
- 모든 모델은 5개 OpenML 회귀 데이터셋에서 동일 split/metric 프로토콜로 비교해야 한다.

## 전체 후보군

```text
GBDT / 전통 ML
1. lightgbm
2. catboost

Neural tabular
1. realmlp
2. tabm
3. tabr
4. dcnv2
5. node

Transformer / Attention
1. ft_transformer
2. tab_transformer
3. tabnet

Pretrained / Foundation
1. tabpfn
2. tabiclv2
```

## 1. GBDT / 전통 ML

### lightgbm

LightGBM은 gradient boosting decision tree 계열의 강한 테이블 데이터 기본 모델이다. 수치형 dense 데이터, 중대형 데이터, feature scale이 섞인 일반적인 회귀 문제에서 먼저 확인할 가치가 가장 높다.

이 프로젝트에서의 역할:

- 전체 benchmark의 1순위 기준선
- `superconductivity`, `SGEMM_GPU_kernel_performance`, `Santander_transaction_value`에서 우선 확인할 모델
- 딥러닝 모델들이 실제로 LightGBM을 넘는지 판단하는 기준

장점:

- 학습 속도와 성능 균형이 좋다.
- 수치형 tabular 회귀에서 매우 강한 기본값이다.
- feature importance 확인이 쉬워 error analysis에 유리하다.

주의점:

- categorical-heavy 데이터에서는 인코딩 방식에 따라 성능 차이가 날 수 있다.
- high-cardinality categorical은 CatBoost와 반드시 같이 비교해야 한다.

### catboost

CatBoost도 GBDT 계열이지만 categorical feature 처리에 강점이 있다. 범주형 변수가 많거나 cardinality가 높은 데이터에서 LightGBM과 다른 강점을 보일 수 있다.

이 프로젝트에서의 역할:

- `Allstate_Claims_Severity`, `Mercedes_Benz_Greener_Manufacturing`의 핵심 비교 모델
- LightGBM이 categorical encoding에 민감할 때 대안 baseline

장점:

- categorical feature를 모델 내부에서 다루는 설계가 강하다.
- ordered boosting 계열 아이디어로 target leakage/overfitting 위험을 줄이려는 구조를 갖는다.
- 전처리를 단순화할 수 있다.

주의점:

- 학습 시간이 LightGBM보다 길어질 수 있다.
- CPU/GPU 설정과 categorical column 전달 여부에 따라 동작이 달라질 수 있다.

## 2. Neural tabular

### realmlp

RealMLP는 tabular classification/regression에서 강한 pre-tuned MLP baseline을 목표로 한 모델이다. 단순 MLP가 아니라 activation, regularization, 학습 설정을 tabular에 맞게 강하게 조정한 계열로 보는 것이 맞다.

이 프로젝트에서의 역할:

- neural tabular 계열 1순위
- GBDT가 아닌 딥러닝 모델이 기본 회귀 benchmark에서 경쟁 가능한지 확인
- numeric dense 데이터에서 `lightgbm` 다음으로 우선 확인

장점:

- 구조가 비교적 단순해서 transformer 계열보다 해석과 운영이 쉽다.
- 중간 규모 tabular 데이터에서 시간 대비 성능을 확인하기 좋다.

주의점:

- categorical-heavy 데이터는 전처리 품질에 민감하다.
- scaling, batch size, early stopping 조건을 고정해야 비교가 공정하다.

### tabm

TabM은 MLP 기반 모델에 parameter-efficient ensembling 아이디어를 결합한 tabular deep learning 모델이다. 하나의 모델이 여러 MLP ensemble처럼 동작하도록 만들어 성능과 비용의 균형을 노린다.

이 프로젝트에서의 역할:

- RealMLP 다음 neural tabular 확장 후보
- 딥러닝 계열에서 ensemble 효과가 성능을 끌어올리는지 확인

장점:

- MLP 기반이라 transformer보다 상대적으로 단순하다.
- ensemble 성격 때문에 단일 MLP보다 안정적인 성능을 기대할 수 있다.

주의점:

- 모델 구현체의 default가 데이터 크기에 적절한지 확인해야 한다.
- 반복 실험 시 seed variance를 기록하는 것이 좋다.

### tabr

TabR은 retrieval-augmented tabular deep learning 모델이다. 예측 대상 row에 대해 학습 데이터에서 가까운 이웃을 검색하고, 그 이웃의 feature와 label 정보를 함께 사용해 예측한다.

이 프로젝트에서의 역할:

- neural tabular 중 성능 잠재력이 큰 연구형 후보
- 비슷한 row끼리 target 패턴이 강하게 공유되는 데이터에서 확인

장점:

- kNN류 inductive bias와 neural model을 결합한다.
- 단순 feed-forward 모델이 놓치는 local pattern을 잡을 수 있다.

주의점:

- retrieval 비용과 메모리 사용량을 확인해야 한다.
- 대용량 데이터에서는 neighbor 검색 설정이 병목이 될 수 있다.

### dcnv2

DCNv2는 Deep & Cross Network v2 계열 모델이다. 핵심은 명시적인 feature cross layer를 통해 feature interaction을 학습하는 것이다. 원래 대규모 ranking/recommendation 환경에서 많이 쓰이는 구조지만, feature crossing이 중요한 tabular regression에서도 비교 가치가 있다.

이 프로젝트에서의 역할:

- feature interaction이 중요한 데이터에서 neural baseline 확장
- `Mercedes`, `Allstate`처럼 categorical/binary interaction이 많을 수 있는 데이터에서 확인

장점:

- feature cross를 모델 구조로 직접 학습한다.
- wide/deep 성격의 tabular 문제에 맞는 inductive bias가 있다.

주의점:

- 모든 일반 회귀 데이터에서 GBDT보다 강하다고 가정하면 안 된다.
- categorical embedding 설정과 cross layer 크기에 민감할 수 있다.

### node

NODE는 Neural Oblivious Decision Ensembles의 약자다. differentiable oblivious decision tree를 쌓아 end-to-end backpropagation으로 학습하는 neural tree 계열 모델이다.

이 프로젝트에서의 역할:

- tree inductive bias를 가진 neural model 비교 후보
- GBDT와 neural model 사이의 중간 성격을 확인

장점:

- decision tree류 구조를 neural training으로 학습한다.
- tabular data 전용 구조라 일반 MLP보다 inductive bias가 명확하다.

주의점:

- 구현체와 hyperparameter에 따라 학습 안정성이 달라질 수 있다.
- 최신 tabular deep learning 후보군과 비교하면 우선순위는 낮다.

## 3. Transformer / Attention

### ft_transformer

FT-Transformer는 tabular feature를 token처럼 변환한 뒤 transformer encoder로 처리하는 모델이다. 수치형 feature는 linear transform, categorical feature는 embedding을 거쳐 feature token으로 만들고 attention을 적용하는 방식이다.

이 프로젝트에서의 역할:

- transformer/attention 계열 1순위
- tabular transformer가 GBDT와 MLP 계열 대비 이득이 있는지 확인

장점:

- tabular transformer 계열에서 표준적인 비교 모델로 쓰기 좋다.
- 수치형과 범주형 feature를 모두 token 기반 표현으로 다룰 수 있다.

주의점:

- 데이터가 작으면 overfitting 또는 불안정한 성능이 나올 수 있다.
- 학습 시간이 GBDT보다 길 수 있다.

### tab_transformer

TabTransformer는 categorical feature embedding을 transformer self-attention으로 contextualize하는 모델이다. 범주형 변수 사이의 관계를 attention으로 학습하는 데 초점이 있다.

이 프로젝트에서의 역할:

- categorical-heavy 데이터에서 transformer 계열 비교 후보
- `Allstate`, `Mercedes`에서 CatBoost/LightGBM 대비 attention 계열의 가치를 확인

장점:

- categorical feature 관계를 명시적으로 모델링한다.
- 범주형 column이 많을 때 실험 의미가 크다.

주의점:

- numeric-only 데이터에서는 FT-Transformer보다 우선순위가 낮다.
- categorical preprocessing과 embedding cardinality 설정이 중요하다.

### tabnet

TabNet은 sequential attention을 사용해 각 decision step에서 어떤 feature를 볼지 선택하는 tabular neural model이다. Transformer라기보다는 attention 기반 feature selection 모델에 가깝다.

이 프로젝트에서의 역할:

- attention 기반 interpretability 비교 후보
- feature selection mask를 통한 설명 가능성이 필요한 경우 보조 후보

장점:

- instance-wise feature selection 관점의 해석 가능성을 제공한다.
- raw feature를 attention으로 선택하며 단계적으로 의사결정을 한다.

주의점:

- 최신 benchmark에서 항상 강한 후보라고 보기는 어렵다.
- 튜닝과 early stopping에 민감할 수 있어 1차 우선순위는 낮다.

## 4. Pretrained / Foundation

### tabpfn

TabPFN은 tabular foundation model 계열이다. 사전학습된 transformer가 학습 데이터와 예측 대상을 context로 받아, 별도 parameter update 없이 in-context 방식으로 예측하는 구조다. 최신 버전/체크포인트에 따라 regression, categorical, missing value 지원 범위가 달라질 수 있으므로 실행 환경의 설치 버전을 명시해야 한다.

이 프로젝트에서의 역할:

- pretrained/foundation 계열 1순위
- 작은/중간 크기 table에서 zero-shot 또는 training-free 성능 확인
- GBDT와 완전히 다른 패러다임의 비교 축

장점:

- 별도 hyperparameter tuning 없이 빠르게 강한 결과를 낼 수 있다.
- small-to-medium 데이터에서 실험 가치가 크다.

주의점:

- 대용량 데이터 전체를 그대로 넣는 방식에는 제약이 있을 수 있다.
- checkpoint/license/token 접근 여부를 사전에 확인해야 한다.
- 비교 시 sampling 여부와 context size 제한을 반드시 기록해야 한다.

### tabiclv2

TabICL 계열은 tabular foundation model을 in-context learning 방식으로 사용하는 접근이다. 학습 데이터를 context로 제공하고, parameter update 없이 single forward pass 성격으로 예측하는 방향이라는 점에서 TabPFN과 같은 foundation/in-context 계열에 둔다.

이 프로젝트에서의 역할:

- TabPFN 다음 foundation 확장 후보
- 대형 tabular context를 다루는 in-context model이 5개 회귀 데이터셋에서 어떤 성능/비용 특성을 보이는지 확인

장점:

- pretrained/in-context 계열의 두 번째 축으로 비교 의미가 크다.
- TabPFN과 비슷한 사용 목적이지만 scaling/구현 특성이 다를 수 있다.

주의점:

- `tabiclv2`라는 local wrapper가 실제 어떤 checkpoint/API를 사용하는지 실행 로그에 남겨야 한다.
- version, checkpoint, max context, sampling 정책을 결과표에 같이 기록해야 한다.

## 데이터셋별 우선 확인 포인트

| Dataset | 먼저 볼 모델 | 이유 |
|---|---|---|
| `superconductivity` | `lightgbm`, `realmlp`, `ft_transformer`, `tabpfn` | dense numeric 소재 데이터라 기본 우선순위 그대로 적용 |
| `Allstate_Claims_Severity` | `catboost`, `lightgbm`, `tab_transformer`, `tabpfn` | categorical-heavy 보험 데이터 |
| `SGEMM_GPU_kernel_performance` | `lightgbm`, `realmlp`, `ft_transformer` | 대용량 numeric HPC 성능 예측 |
| `Mercedes_Benz_Greener_Manufacturing` | `catboost`, `lightgbm`, `tab_transformer`, `dcnv2` | high-cardinality categorical 및 binary feature interaction 가능성 |
| `Santander_transaction_value` | `lightgbm`, `realmlp`, `tabpfn`, `tabr` | 초고차원 sparse-ish numeric feature |

## 결과 기록 규칙

모델별 결과를 비교할 때는 다음 정보를 반드시 같이 남긴다.

- dataset name / OpenML id
- target column
- train/valid split seed
- row count, feature count
- categorical/numeric feature count
- model name과 model group
- package version, checkpoint/version이 있는 경우 checkpoint 이름
- sampling/context limit 사용 여부
- RMSE, MAE, WAPE
- runtime, GPU/CPU 여부

## 참고 문헌과 출처

- LightGBM documentation, Features: <https://lightgbm.readthedocs.io/en/v4.2.0/Features.html>
- CatBoost documentation, Categorical features: <https://catboost.ai/docs/en/features/categorical-features>
- CatBoost paper, "CatBoost: unbiased boosting with categorical features": <https://arxiv.org/abs/1706.09516>
- RealMLP paper, "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data": <https://arxiv.org/abs/2407.04491>
- TabM official repository: <https://github.com/yandex-research/tabm>
- TabM paper: <https://arxiv.org/abs/2410.24210>
- TabR paper: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html>
- DCNv2 paper: <https://arxiv.org/abs/2008.13535>
- NODE paper: <https://arxiv.org/abs/1909.06312>
- FT-Transformer official implementation / paper context: <https://github.com/yandex-research/rtdl-revisiting-models>
- FT-Transformer paper: <https://papers.nips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf>
- TabTransformer paper: <https://huggingface.co/papers/2012.06678>
- TabNet paper: <https://ojs.aaai.org/index.php/AAAI/article/view/16826>
- TabPFN official repository: <https://github.com/PriorLabs/TabPFN>
- TabPFN Nature paper: <https://www.nature.com/articles/s41586-024-08328-6>
- TabICL paper: <https://proceedings.mlr.press/v267/qu25d.html>
