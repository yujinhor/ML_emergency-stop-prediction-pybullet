Emergency Braking Failure Prediction
PyBullet 시뮬레이터를 이용하여 곡선 도로에서의 로봇 및 자율주행 차량의 급제동 실패를 실시간으로 예측하는 딥러닝 기반 시스템입니다.

📋 프로젝트 개요
문제 상황
돌발 상황에서 로봇이나 자율주행 차량의 급제동은 생명과 재산 문제와 직결됩니다. 하지만 현실의 물리 변수들은 끊임없이 변하기 때문에, 순수 물리 공식만으로는 실패를 실시간으로 예측하기 어렵습니다.

솔루션
본 프로젝트는 LSTM/GRU 시계열 분석 모델을 활용하여:

다양한 물리 변수 조합에서의 급제동 성공/실패 데이터 수집

실패에 높은 영향을 주는 변수 순위화

실패 확률이 높은 임계 조건 도출

실시간 경보 정책 시스템 구현

🚗 실험 시나리오
주요 물리 변수
노면 상황: 건조, 젖음, 결빙 (3가지)

로봇 질량: 무적재 ~ 최대적재 사이 (랜덤)

도로 곡률: 실제 한국 도로 기준 (고정)

급제동 시점: 곡선 진입 전후 (-20m, -10m, 0m, +10m) (변수화)

초기속도, 공기저항: 현실 데이터 기반 (고정)

실패 판정 기준
급제동 후 로봇이 정지선을 넘거나 옆 차선을 침범하는 경우

🎯 Success Criteria
실시간 예측 및 경보 시스템

PyBullet 시뮬레이션 환경에서 급제동 이벤트 후 지연 없이 성공/실패 예측

경보 시스템 무지연 작동

모델 비교 분석

LSTM과 GRU의 정확도 비교 및 성능 차이 분석

정량적 예측 정확도

임계 조건과 변수 순위화 조건에서 95% 이상의 실패 예측 정확도

📦 프로젝트 산출물
A. 예측 모델 (Prediction Models)
LSTM 예측 모델

GRU 예측 모델

실시간 경보 시스템 스크립트

예측 검증 성능 리포트 (모델 성능 지표, 실시간 예측 속도, 오류 사례 분석)

B. 임계값 분석 (Threshold Analysis)
임계값 도출 테이블 (CSV)

Heatmap 시각화 (PNG) - 노면별 실패 확률 기반 색상 분류

C. 변수 중요도 (Variable Importance)
변수 중요도 순위 테이블 (CSV)

상세 분석 리포트 (PDF) - 변수 중요도 요약, 심화 분석, 물리적 해석

🛠️ 기술 스택
시뮬레이션: PyBullet (280Hz 기본 주기)

데이터 샘플링: 10Hz 간격 (정확도 100% 유지)

모델링: LSTM, GRU with Attention Layer

모델 해석: SHAP (TimeSHAP, FasterSHAP)

손실함수: Custom Loss Function (정확도 + 신속성 최적화)

프로그래밍: Python

🔧 핵심 기술 솔루션
1. 실시간 예측의 정확도-속도 트레이드오프
문제: LSTM/GRU 모델의 정확도와 속도 상충 관계

해결 방안:

Attention 레이어 추가로 임계 패턴 도출 및 변수 순위화

SHAP 라이브러리를 통한 변수 중요도 계산

2. 데이터 샘플링의 정보 손실
문제: PyBullet의 기본 주기 280Hz는 너무 방대

해결 방안:

논문 기반 10Hz 간격 샘플링 구현

정확도 100% 유지 확인

3. 모델 해석의 어려움
문제: RNN 기반 모델의 블랙박스 특성

해결 방안:

새로운 손실함수 모델 구현 (정확도 + 신속성)

Attention 가중치를 통한 임계 패턴 시각화

📋 재현성 (Reproducibility)
Random Seed 관리
python
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
환경 통일
bash
pip install -r requirements.txt
📚 참고 논문
Liu, J., et al., 2020, "How much information is lost when sampling driving behavior data? Indicators to quantify the extent of information loss"

Mori, A., et al., 2017, "Early classification of time series with deep learning"

Qin, Y., et al., 2017, "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"

Liu, Y., et al., 2024, "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"

Nie, Y., et al., 2023, "A Time Series is Worth 64 Words: Long-term Time Series Forecasting with Transformers"

Bento, J., et al., 2020, "TimeSHAP: Explaining Recurrent Models through Sequence Perturbations"

Osco, L. P., et al., 2023, "FasterSHAP: Fast and Accurate Shapley Values for Time Series Explanations"

Cai, Y., et al., 2024, "Hybrid physics and neural network model for lateral vehicle dynamic state prediction"

📁 디렉토리 구조
text
emergency-braking-failure-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     # PyBullet 시뮬레이션 원본 데이터
│   └── processed/               # 전처리된 학습 데이터
├── models/
│   ├── lstm_model.py           # LSTM 모델
│   ├── gru_model.py            # GRU 모델
│   └── trained_models/          # 학습된 모델 가중치
├── src/
│   ├── simulation.py            # PyBullet 시뮬레이션 스크립트
│   ├── data_preprocessing.py    # 데이터 전처리
│   ├── model_training.py        # 모델 학습
│   ├── prediction.py            # 예측 시스템
│   └── alert_system.py          # 실시간 경보 시스템
├── evaluation/
│   ├── performance_metrics.py   # 성능 평가
│   ├── threshold_analysis.py    # 임계값 분석
│   └── variable_importance.py   # 변수 중요도 분석
├── visualization/
│   ├── heatmap_generator.py     # Heatmap 생성
│   └── plots.py                 # 기타 시각화
└── reports/
    ├── performance_report.pdf    # 성능 분석 리포트
    ├── threshold_results.csv     # 임계값 테이블
    └── variable_importance.csv   # 변수 중요도 테이블
🚀 빠른 시작
1. 환경 설정
bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. 데이터 생성 (시뮬레이션)
bash
python src/simulation.py --scenarios 1000 --output data/raw/
3. 데이터 전처리
bash
python src/data_preprocessing.py --input data/raw/ --output data/processed/
4. 모델 학습
bash
python src/model_training.py --model lstm --epochs 100 --batch_size 32
python src/model_training.py --model gru --epochs 100 --batch_size 32
5. 모델 평가
bash
python evaluation/performance_metrics.py --model trained_models/lstm_model.h5
python evaluation/performance_metrics.py --model trained_models/gru_model.h5
6. 실시간 예측 실행
bash
python src/prediction.py --model trained_models/lstm_model.h5 --input_stream simulation
👥 팀 구성
역할	이름	전공
팀장	호예진	물리학과
팀원	전유빈	스마트운행체 공학과
팀원	최요한	산업공학과
팀원	허유진	스마트운행체 공학과
