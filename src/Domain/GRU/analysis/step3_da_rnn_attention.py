import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import os

# ----------------------------------------------------
# 1. 환경 설정
# ----------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cpu") # Attention 추출은 CPU로도 순식간에 끝납니다.

INPUT_SIZE = 8
HIDDEN_SIZE = 64 

feature_names = [
    "Speed", "Dist_to_Wall", "Drag_Force", "Is_Braking", 
    "Mass_kg", "Friction_Cond", "Air_Density", "Init_Speed"
]

# ----------------------------------------------------
# 2. 데이터 준비
# ----------------------------------------------------
print("📂 [Attention] 데이터 로딩 및 전처리 중...")
try:
    X_train_np = np.load('X_train.npy')
    X_test_np = np.load('X_test.npy')
except FileNotFoundError:
    print("❌ 오류: 데이터 파일을 찾을 수 없습니다.")
    sys.exit()

if X_train_np.shape[2] > 8:
    X_train_np = np.delete(X_train_np, 7, axis=2)
    X_test_np = np.delete(X_test_np, 7, axis=2)

scaler = StandardScaler()

# [수정됨] F 대신 n_feats를 사용하여 충돌 방지
N, T, n_feats = X_train_np.shape
X_train_2d = X_train_np.reshape(N * T, n_feats)
scaler.fit(X_train_2d)

N_test, T_test, _ = X_test_np.shape
X_test_2d = X_test_np.reshape(N_test * T_test, n_feats)
X_test_scaled = scaler.transform(X_test_2d)

X_test_final = X_test_scaled.reshape(N_test, T_test, n_feats)
X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32).to(device)
print(f"✅ 데이터 준비 완료 (총 {N_test}개 샘플)")

# ----------------------------------------------------
# 3. 모델 정의 (DA-RNN)
# ----------------------------------------------------
class InputAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputAttention, self).__init__()
        self.linear = nn.Linear(hidden_size + input_size, input_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x_t, h_prev):
        combined = torch.cat((x_t, h_prev), dim=1)
        scores = torch.tanh(self.linear(combined))
        alpha = self.softmax(scores) 
        return alpha * x_t, alpha

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, encoder_outputs):
        scores = torch.tanh(self.linear(encoder_outputs))
        # 여기서 F는 이제 안전하게 torch.nn.functional을 가리킵니다.
        beta = F.softmax(scores, dim=1) 
        context = torch.sum(beta * encoder_outputs, dim=1)
        return context, beta

class DA_RNN_Explainable(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DA_RNN_Explainable, self).__init__()
        self.hidden_size = hidden_size
        self.input_attention = InputAttention(input_size, hidden_size)
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.temporal_attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        encoder_outputs = []
        alpha_list = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            weighted_x_t, alpha = self.input_attention(x_t, h_t)
            h_t = self.gru_cell(weighted_x_t, h_t)
            encoder_outputs.append(h_t.unsqueeze(1))
            alpha_list.append(alpha.unsqueeze(1))
        encoder_outputs = torch.cat(encoder_outputs, dim=1)
        input_attention_weights = torch.cat(alpha_list, dim=1)
        context_vector, beta = self.temporal_attention(encoder_outputs)
        logits = self.fc(context_vector)
        return torch.sigmoid(logits), input_attention_weights, beta

print("📂 DA-RNN 모델 로딩 중...")
model = DA_RNN_Explainable(INPUT_SIZE, HIDDEN_SIZE, 1).to(device)

# 모델 파일 로드 (Step 2와 동일한 로직)
try:
    model.load_state_dict(torch.load('da_rnn_best_maxpool.pth', map_location=device))
    model.eval()
    print("✅ DA-RNN 모델 로드 성공!")
except:
    print("⚠️ 'da_rnn_best_maxpool.pth' 실패 -> 'da_rnn_best.pth' 시도")
    try:
        model.load_state_dict(torch.load('da_rnn_best.pth', map_location=device))
        print("✅ DA-RNN 모델 로드 성공!")
    except:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        sys.exit()

# ----------------------------------------------------
# 4. Global Attention 계산
# ----------------------------------------------------
print("\n🧠 Global Attention 계산 시작 (목표: 100개 샘플 평균)...")

TARGET_N = 100
if N_test < TARGET_N: TARGET_N = N_test

# 랜덤하게 100개 추출 (또는 전체)
indices = np.random.choice(N_test, TARGET_N, replace=False)
selected_data = X_test_tensor[indices]

with torch.no_grad():
    # 2번째 리턴값이 Input Attention Weights입니다.
    # (Batch, Time, Features)
    _, input_att_weights, _ = model(selected_data)

# 전체 평균 (Global Importance)
# 1. Time 축 평균: 시계열 전체에서 변수의 기여도
# 2. Batch 축 평균: 모든 샘플에 대한 평균
global_att_importance = input_att_weights.mean(dim=1).mean(dim=0).cpu().numpy()

print("✅ 계산 완료!")

# ----------------------------------------------------
# 5. 시각화 및 저장
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

# DA-RNN Attention은 보라색 계열로 표시 (SHAP과 구분)
bars = plt.bar(feature_names, global_att_importance, color='rebeccapurple', alpha=0.8)

plt.title(f"[DA-RNN] Global Input Attention Importance (N={TARGET_N})")
plt.ylabel("Mean Attention Weight")
plt.xlabel("Features")
plt.grid(axis='y', alpha=0.3)

# 막대 위에 수치 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', 
             va='bottom', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('darnn_global_attention.png', dpi=300)
print("🎉 결과 이미지 저장 완료: darnn_global_attention.png")
plt.show()