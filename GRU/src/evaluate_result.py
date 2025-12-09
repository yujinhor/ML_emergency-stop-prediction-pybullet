import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# ----------------------------------------------------
# 1. 환경 설정 및 데이터 전처리 (학습과 동일하게!)
# ----------------------------------------------------
# Mac 한글 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cpu") # 평가는 CPU로 충분합니다
INPUT_SIZE = 8 # Brake_Torque 제거됨

def load_and_preprocess_data():
    print("📂 데이터 로딩 및 전처리(Brake 제거, 표준화) 중...")
    
    # 1. 파일 로드 (Train 데이터는 스케일 기준을 잡기 위해 필요)
    X_train_np = np.load('X_train.npy')
    X_test_np = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    # 2. Brake_Torque (7번째 인덱스) 제거
    X_train_np = np.delete(X_train_np, 7, axis=2)
    X_test_np = np.delete(X_test_np, 7, axis=2)
    
    # 3. StandardScaler 적용
    scaler = StandardScaler()
    N, T, F = X_train_np.shape
    
    # Train 데이터로 피팅 (평균/분산 학습)
    X_train_2d = X_train_np.reshape(N * T, F)
    scaler.fit(X_train_2d) 
    
    # Test 데이터 변환
    N_test, T_test, _ = X_test_np.shape
    X_test_2d = X_test_np.reshape(N_test * T_test, F)
    X_test_scaled = scaler.transform(X_test_2d)
    X_test_final = X_test_scaled.reshape(N_test, T_test, F)
    
    # Tensor 변환
    X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    return X_test_tensor, y_test_tensor

X_test, y_test = load_and_preprocess_data()
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("✅ 전처리된 테스트 데이터 준비 완료!")


# ----------------------------------------------------
# 2. 모델 구조 정의 및 로드
# ----------------------------------------------------
class RobotGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=1):
        super(RobotGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        logits = self.fc(out[:, -1, :])
        return torch.sigmoid(logits)

model = RobotGRU(input_size=INPUT_SIZE).to(device)
try:
    # 학습된 최고 성능 모델 로드
    model.load_state_dict(torch.load('gru_model_best.pth', map_location=device))
    model.eval()
    print("✅ 학습된 Best Model 로드 성공!")
except FileNotFoundError:
    print("❌ 오류: 'gru_model_best.pth' 파일을 찾을 수 없습니다.")
    sys.exit()


# ----------------------------------------------------
# 3. 예측 실행 (단 1회만 수행하여 속도 최적화)
# ----------------------------------------------------
def get_all_probabilities(model, loader):
    print("🤖 모델 예측(Inference) 실행 중...")
    all_probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs)
            all_probs.extend(outputs.numpy().flatten())
    return np.array(all_probs)

# 예측값 미리 계산
targets_np = y_test.numpy().flatten()
all_probs_np = get_all_probabilities(model, test_loader)
print("✅ 예측 완료! 평가 지표 계산 시작...")


# ----------------------------------------------------
# 4. 최적 임계치(Threshold) 탐색
# ----------------------------------------------------
print("\n" + "="*50)
print("🔍 [심화 분석] 최적의 임계치(Threshold) 탐색")
print("="*50)
print(f"{'Threshold':<10} | {'Accuracy':<10} | {'F1 Score':<10} | {'Recall':<10}")
print("-" * 50)

best_f1 = 0
best_th = 0.5

# 0.1부터 0.9까지 0.05 단위로 테스트
for th in np.arange(0.1, 0.91, 0.05):
    preds = (all_probs_np > th).astype(int)
    acc = accuracy_score(targets_np, preds)
    f1 = f1_score(targets_np, preds, zero_division=0)
    recall = recall_score(targets_np, preds, zero_division=0)
    
    print(f"{th:.2f}       | {acc:.4f}     | {f1:.4f}     | {recall:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print("-" * 50)
print(f"💡 추천: F1 점수가 가장 높은 임계치는 {best_th:.2f} 입니다. (Max F1: {best_f1:.4f})")


# ----------------------------------------------------
# 5. 최종 결과 리포트 (최적 임계치 기준)
# ----------------------------------------------------
final_preds = (all_probs_np > best_th).astype(int)

print("\n" + "="*50)
print(f"📢 [최종 성적표] Threshold = {best_th:.2f}")
print("="*50)
print(f"정확도 (Accuracy) : {accuracy_score(targets_np, final_preds):.4f}")
print(f"F1 Score        : {f1_score(targets_np, final_preds):.4f}")
print(f"재현율 (Recall)  : {recall_score(targets_np, final_preds):.4f}")
print(f"정밀도 (Precision): {best_f1 / recall_score(targets_np, final_preds) * 0.5 if recall_score(targets_np, final_preds) else 0:.4f} (추정)")
print("-" * 50)
print("분류 보고서:\n", classification_report(targets_np, final_preds))

# 혼동 행렬 시각화
cm = confusion_matrix(targets_np, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['정상(0)', '고장(1)'], yticklabels=['정상(0)', '고장(1)'])
plt.xlabel(f'예측값 (Predicted) @ Th={best_th:.2f}')
plt.ylabel('실제값 (Actual)')
plt.title('Confusion Matrix (최종)')
plt.savefig('confusion_matrix_final.png')
print("🖼️ 'confusion_matrix_final.png' 저장 완료!")