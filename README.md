# 실시간 로봇 급제동 실패 예측과 경보 정책 (Real-time Prediction of Robot Emergency Braking Failure and Warning Policy)

본 프로젝트는 건국대학교 2025학년도 2학기 **기계학습** 과목의 **T7조** 기말 프로젝트 결과물입니다. 

<br>

| 이름 (직책) | 학번 | 학과 | 역할 분담 | GitHub |
| :--- | :--- | :--- | :--- | :--- |
| **호예진** (팀장) | 202315207 | 물리학과 | - PyBullet 시나리오 환경 및 물리 변수 설계<br>- 데이터 전처리<br>- 손실함수(Loss Function) 설계<br>- Transformer 모델 분석<br>- 경보 정책 System 임계값 산출  | @hoyejin  |
| **전유빈** (팀원) | 202112306 | 스마트운행체공학과 | - PyBullet 시나리오 환경 개발<br>- 데이터 생성 및 자동화 파이프라인 구축<br>- Transformer 모델 추론 로직 통합<br>- 경보 정책 System 구현  | @fsdsa  |
| **최요한** (팀원) | 202214223 | 산업공학과 | - Attention 메커니즘 기반 GRU 모델 아키텍처 구축<br>- 로직 개선 및 수치 최적화<br>- SHAP 분석 및 변수 순위화<br>- 모델 성능 평가  | @johnnyjohn1158  |
| **허유진** (팀원) | 202112318 | 스마트운행체공학과 | - PyBullet 시나리오 환경 설계<br>- SW 시스템 설계<br>- Transformer 모델 구축 및 변수 순위화<br>- 모델 성능 평가 및 임계값 최적화| @yujinhor  |


<br>

* **Platform**: Python Environment (Anaconda) 
* **Deep Learning Framework**: PyTorch, CUDA Toolkit 
* **Libraries**: PyBullet (물리 시뮬레이션), Pandas (데이터 처리), SHAP (설명 가능한 AI 분석), Matplotlib (시각화) 
* **Hardware**: RTX 4060 GPU Workstation (Ubuntu 22.04) 

<br>

### 데이터

* **데이터** : https://drive.google.com/drive/folders/1uMH3sl9O4RyyEAookkKQj1yIpHY2p_Iv?usp=drive_link

* **데이터(전처리)** : https://drive.google.com/drive/folders/1xvrGReZB15PfyJa3uBjVTTJz1InclRAR
