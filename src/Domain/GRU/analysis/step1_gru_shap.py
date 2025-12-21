import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import os

# ----------------------------------------------------
# 1. 환경 설정
# ----------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cpu")  # SHAP은 CPU가 안정적

INPUT_SIZE = 8
TIME_STEPS = 2400 
FEATURE_COUNT = 8 

feature_names = [
    "Speed", "Dist_to_Wall", "Drag_Force", "Is_Braking", 
    "Mass_kg", "Friction_Cond", "Air_Density", "Init_Speed"
]

# ----------------------------------------------------
# 2. 데이터 준비
# ----------------------------------------------------
print("📂 [GRU] 데이터 로딩 및 전처리 중...")
try:
    X_train_np = np.load('X_train.npy')
    X_test_np = np.load('X_test.npy')
except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다.")
    sys.exit()

# Brake_Torque 제거 (Index 7)
if X_train_np.shape[2] > 8:
    X_train_np = np.delete(X_train_np, 7, axis=2)
    X_test_np = np.delete(X_test_np, 7, axis=2)

# Scaling
scaler = StandardScaler()
N, T, F = X_train_np.shape
X_train_2d = X_train_np.reshape(N * T, F)
scaler.fit(X_train_2d)

N_test, T_test, _ = X_test_np.shape
X_test_2d = X_test_np.reshape(N_test * T_test, F)
X_test_scaled = scaler.transform(X_test_2d)

X_test_final = X_test_scaled.reshape(N_test, T_test, F)
X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32).to(device)

print(f"✅ 데이터 준비 완료 (총 {N_test}개 샘플)")

# ----------------------------------------------------
# 3. 모델 정의 및 로드
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
    model.load_state_dict(torch.load('gru_model_best.pth', map_location=device))
    model.eval()
    print("✅ GRU 모델 로드 성공")
except:
    print("❌ 'gru_model_best.pth' 로드 실패")
    sys.exit()

# ----------------------------------------------------
# 4. SHAP 분석 (N=100)
# ----------------------------------------------------
def predict_func(X_np):
    model.eval()
    X_reshaped = X_np.reshape(X_np.shape[0], TIME_STEPS, FEATURE_COUNT) 
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        if logits.dim() == 3: output = logits[:, -1, :]
        elif logits.dim() == 2: output = logits
        else: output = logits
        return output.cpu().numpy().flatten()

print("\n📊 GRU Global SHAP 분석 시작...")

# 데이터 Flatten
def flatten(t): return t.cpu().numpy().reshape(t.shape[0], -1)

# 배경 데이터 (20개)
bg_idx = np.random.choice(N_test, 20, replace=False)
background = flatten(X_test_tensor[bg_idx])

# 타겟 데이터 (100개)
TARGET_N = 100
BATCH_SIZE = 10
if N_test < TARGET_N: TARGET_N = N_test
target_idx = np.random.choice(N_test, TARGET_N, replace=False)

explainer = shap.KernelExplainer(predict_func, background)

shap_values_list = []
test_data_list = []

for i in range(0, TARGET_N, BATCH_SIZE):
    curr_idx = target_idx[i : i+BATCH_SIZE]
    batch_tensor = X_test_tensor[curr_idx]
    batch_flat = flatten(batch_tensor)
    
    # 계산
    shap_val = explainer.shap_values(batch_flat, nsamples=100, silent=True)
    if isinstance(shap_val, list): shap_val = shap_val[0]
    
    shap_values_list.append(shap_val)
    test_data_list.append(batch_flat)
    print(f"   Running Batch... {i + len(curr_idx)}/{TARGET_N}")

# 통합
shap_total = np.concatenate(shap_values_list, axis=0)
data_total = np.concatenate(test_data_list, axis=0)

# ----------------------------------------------------
# 5. 시각화
# ----------------------------------------------------
shap_3d = shap_total.reshape(-1, TIME_STEPS, FEATURE_COUNT)
shap_mean = shap_3d.mean(axis=1) # 시간축 평균
data_3d = data_total.reshape(-1, TIME_STEPS, FEATURE_COUNT)
data_mean = data_3d.mean(axis=1)

# Bar Plot
plt.figure()
shap.summary_plot(shap_mean, data_mean, feature_names=feature_names, plot_type="bar", show=False)
plt.title("[GRU] Global Feature Importance")
plt.tight_layout()
plt.savefig('gru_global_shap_bar.png')
print("✅ 저장 완료: gru_global_shap_bar.png")

# Dot Plot
plt.figure()
shap.summary_plot(shap_mean, data_mean, feature_names=feature_names, show=False)
plt.title("[GRU] Global SHAP Distribution")
plt.tight_layout()
plt.savefig('gru_global_shap_dot.png')
print("✅ 저장 완료: gru_global_shap_dot.png")