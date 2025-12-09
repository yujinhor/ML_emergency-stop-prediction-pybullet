##  수정사항


### 1. 전처리 파이프라인 고도화 (Preprocessing Refinement)
초기 실험 단계의 MinMax Scaling 방식이 이상치(Outlier)에 민감하게 반응하는 문제를 개선하고자, 전처리 방식을 전면 수정하였습니다.

* **StandardScaler 적용:** 데이터의 평균을 0, 분산을 1로 맞추는 정규화(Standardization)를 통해, 변수 간 스케일 차이를 줄이고 GRU/DA-RNN 모델이 **가우시안 분포(Gaussian Distribution)**에 가까운 데이터에서 더 빠르고 안정적으로 수렴하도록 유도하였습니다.
* **Feature Selection (`Brake_Torque` 제거):** 분석 결과 `Brake_Torque` 변수는 고장 발생 시점과 과도한 상관관계(Data Leakage)를 가지거나, 예측의 인과성을 왜곡할 우려가 있어 학습 변수에서 제외하였습니다. 이를 통해 모델이 **"결과"가 아닌 "원인"에 집중**하도록 설계하였습니다.
* **클래스 불균형 해소 (Handling Class Imbalance):** 정상 데이터 대비 고장 데이터가 현저히 적은 불균형 문제를 해결하기 위해, 손실 함수(Loss Function)에 **`pos_weight`** 가중치를 부여하였습니다. 이를 통해 모델이 소수 클래스인 '고장(Failure)'을 놓치지 않고 민감하게 학습하도록 최적화하였습니다.

### 2. 효율적인 학습 및 분석 전략 (Sampling Strategy)
방대한 시계열 데이터와 복잡한 RNN 계열 모델의 연산 비용을 고려하여, 효율적인 샘플링 전략을 채택하였습니다.

* **DA-RNN 학습 전략:** 전체 데이터를 무리하게 학습시키는 대신, 데이터의 대표성을 훼손하지 않는 범위 내에서 **샘플링(Sampling) 기법**을 활용하여 학습 효율성을 높이고 과적합(Overfitting)을 방지하였습니다.
* **SHAP 기반 XAI 분석 전략:** 시계열 모델(RNN/LSTM)에 대한 SHAP `KernelExplainer`는 막대한 연산 비용을 요구합니다. 따라서 전체 테스트 데이터(약 2,033개) 중 **무작위 추출된 대표 표본 100개(Randomly Sampled Instances)**를 선정하여 분석하였습니다.
    * **Iterative Accumulation:** 메모리 과부하를 방지하기 위해 10개씩 배치(Batch) 단위로 나누어 SHAP 값을 계산한 후 최종적으로 병합(Concatenate)하는 방식을 사용했습니다.
    * **Global Explanation:** 추출된 100개 샘플의 SHAP 값을 평균 내어 모델 전체의 **전역적 변수 중요도(Global Feature Importance)**를 도출하였습니다.

---
##  XAI Analysis Results: From "Reactive" to "Physics-Aware"

SHAP(Shapley Additive Explanations)과 Input Attention 메커니즘을 활용하여 두 모델의 의사결정 과정을 비교 분석하였습니다. 분석 결과, 단순 성능 차이를 넘어 데이터를 바라보는 **모델의 관점(Perspective)**에서 근본적인 차이가 발견되었습니다.

### 1. Baseline (GRU): 현상 반응형 모델 (Reactive & Kinematic Focus)
기본 GRU 모델의 Global SHAP 분석 결과, 모델은 주로 **직관적이고 운동학적(Kinematic)인 변수**에 의존하는 경향을 보였습니다.

![GRU SHAP Result](./results/gru_global_shap_bar.png)
*(Fig 1. GRU 모델의 Global Feature Importance)*

* **주요 변수:** `Speed` (1위), `Mass_kg` (2위), `Dist_to_Wall` (3위)
* **해석:** GRU 모델은 **"현재 속도가 빠르고, 로봇이 무거우며, 벽이 가까우면 위험하다"**는 1차원적인 판단 로직을 학습했습니다.
* **한계:** 이는 인간의 직관과 유사하지만, 주행 환경의 보이지 않는 물리적 조건(공기 저항, 마찰 변화 등)은 상대적으로 간과하고 있어 복잡한 동적 환경에서의 일반화 성능이 떨어질 우려가 있습니다.

---

### 2. Proposed (DA-RNN): 물리/맥락 인지형 모델 (Physics-Aware & Contextual Focus)
반면, 제안하는 DA-RNN 모델은 **환경적 요인과 동역학적(Dynamic) 상호작용**을 반영하는 변수들을 최상위 중요도로 선정하였습니다.

![DA-RNN SHAP Result](./results/darnn_global_shap_bar.png)
*(Fig 2. DA-RNN 모델의 Global Feature Importance)*

* **주요 변수:** **`Air_Density` (1위)**, `Init_Speed` (2위), `Is_Braking` (3위), `Drag_Force` (4위)
* **도메인 해석 (Physics Domain Interpretation):**
    * **`Air_Density` (공기 밀도) & `Drag_Force` (항력):** 모델은 단순히 현재 속도만 보는 것이 아니라, 주행 안정성에 결정적인 영향을 미치는 **'공기 저항(Aerodynamic Drag)'**의 메커니즘을 파악하고 있습니다. 공기 밀도가 높을수록 항력이 커져 제동 거리에 영향을 준다는 물리적 인과관계를 학습한 것으로 보입니다.
    * **`Init_Speed` & `Is_Braking`:** 초기 운동 에너지 상태에서 제동 명령이 입력되었을 때의 결과적 위험도를 예측합니다.
* **의의:** DA-RNN은 단순한 센서 값 매핑을 넘어, **"왜(Why) 고장이 발생하는가?"**에 대한 물리적 맥락(Physics Context)을 이해하는 고도화된 추론 능력을 보여줍니다.

---

### 3. Model Reliability: Attention과 SHAP의 정합성 검증
XAI 분석의 신뢰도를 검증하기 위해, DA-RNN 모델 내부의 **Input Attention Weight(과정)**와 **SHAP Value(결과)**를 교차 검증(Cross-Validation)하였습니다.




* **분석 결과:**
    1.  DA-RNN의 Attention 메커니즘은 **`Air_Density`**에 가장 높은 가중치(0.1602)를 부여하여, 학습 과정에서 이 변수를 가장 집중적으로 모니터링했음을 보여줍니다.
    2.  이는 SHAP 분석에서 `Air_Density`가 1위로 선정된 결과와 완벽하게 일치합니다.
* **결론:** 모델이 내부적으로 중요하게 **주목(Attention)**한 변수가 실제로도 예측 결과에 가장 큰 **기여(Contribution)**를 했습니다. 이는 DA-RNN이 우연히 맞춘 것이 아니라, **논리적 타당성을 가지고 학습되었음**을 강력하게 시사합니다.

> ** Summary**
> 본 연구를 통해 **DA-RNN**은 단순한 시계열 예측 성능 향상뿐만 아니라, **물리적 환경 변수(공기 밀도 등)를 고려한 설명 가능한(Explainable) 의사결정**을 수행함을 입증하였습니다. 이는 실제 산업 현장에서의 신뢰성(Reliability) 확보에 있어 매우 중요한 요소입니다.
