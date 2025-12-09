import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import os

# ----------------------------------------------------
# 1. 환경 설정 및 데이터 로드
# ----------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cpu") # SHAP 계산 시 CPU 권장

INPUT_SIZE = 8
TIME_STEPS = 2400 
FEATURE_COUNT = 8 
HIDDEN_SIZE = 64 # DA-RNN Hidden Size

feature_names = [
    "Speed", "Dist_to_Wall", "Drag_Force", "Is_Braking", 
    "Mass_kg", "Friction_Cond", "Air_Density", "Init_Speed"
]

def load_data():
    print("📂 [DA-RNN] 데이터 로딩 및 전처리 중...")
    try:
        X_train_np = np.load('X_train.npy')
        X_test_np = np.load('X_test.npy')
    except FileNotFoundError:
        print("❌ 오류: 데이터 파일을 찾을 수 없습니다.")
        sys.exit()

    if X_test_np.shape[2] > 8:
        X_train_np = np.delete(X_train_np, 7, axis=2)
        X_test_np = np.delete(X_test_np, 7, axis=2)

    scaler = StandardScaler()
    
    # [수정된 부분] F라는 변수명 대신 n_feats를 사용해 충돌 방지
    N, T, n_feats = X_train_np.shape 
    X_train_2d = X_train_np.reshape(N * T, n_feats)
    scaler.fit(X_train_2d)

    N_test, T_test, _ = X_test_np.shape
    X_test_2d = X_test_np.reshape(N_test * T_test, n_feats)
    X_test_scaled = scaler.transform(X_test_2d)
    
    X_test_final = X_test_scaled.reshape(N_test, T_test, n_feats)
    return torch.tensor(X_test_final, dtype=torch.float32).to(device)

X_test_tensor = load_data()
total_samples = X_test_tensor.shape[0]
print(f"✅ 데이터 준비 완료! (총 {total_samples}개)")


# ----------------------------------------------------
# 2. DA-RNN 모델 클래스 정의
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
        # 여기서 F는 이제 정상적으로 torch.nn.functional을 가리킵니다.
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

# 모델 파일명 확인 (예: da_rnn_best_maxpool.pth)
try:
    model.load_state_dict(torch.load('da_rnn_best_maxpool.pth', map_location=device))
    model.eval()
    print("✅ 모델 로드 성공!")
except:
    print("⚠️ 파일명 확인 필요: 'da_rnn_best_maxpool.pth'가 없어서 'da_rnn_best.pth' 시도")
    try:
        model.load_state_dict(torch.load('da_rnn_best.pth', map_location=device))
        print("✅ 모델 로드 성공!")
    except:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        sys.exit()


# ----------------------------------------------------
# 3. [중요] DA-RNN 전용 Wrapper 함수
# ----------------------------------------------------
def predict_func_darnn(X_np):
    model.eval()
    N_samples = X_np.shape[0]
    X_reshaped = X_np.reshape(N_samples, TIME_STEPS, FEATURE_COUNT) 
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        logits = outputs[0]  # [0]번 인덱스가 예측값
        
        if logits.dim() == 3: output = logits[:, -1, :]
        elif logits.dim() == 2: output = logits
        else: output = logits
        
        return output.cpu().numpy().flatten()


# ----------------------------------------------------
# 4. Global SHAP 분석 (N=100, 배치 실행)
# ----------------------------------------------------
print("\n📊 DA-RNN Global SHAP 분석 시작 (목표: 100개)")
import warnings
warnings.filterwarnings("ignore") # 경고 메시지 끄기

def flatten_data(tensor):
    return tensor.cpu().numpy().reshape(tensor.shape[0], -1)

# 배경 데이터
idx_bg = np.random.choice(total_samples, 20, replace=False)
background_np_flat = flatten_data(X_test_tensor[idx_bg])

# Explainer 생성
explainer = shap.KernelExplainer(predict_func_darnn, background_np_flat)

# 타겟 샘플 선정
TARGET_N = 100
BATCH_SIZE = 10
if total_samples < TARGET_N: TARGET_N = total_samples
all_indices = np.random.choice(total_samples, TARGET_N, replace=False)

all_shap_values = []
all_test_data = []

for i in range(0, TARGET_N, BATCH_SIZE):
    current_indices = all_indices[i : i + BATCH_SIZE]
    batch_tensor = X_test_tensor[current_indices]
    batch_flat = flatten_data(batch_tensor)
    
    # SHAP 계산
    shap_vals_batch = explainer.shap_values(batch_flat, nsamples=100, silent=True)
    if isinstance(shap_vals_batch, list): shap_vals_batch = shap_vals_batch[0]
        
    all_shap_values.append(shap_vals_batch)
    all_test_data.append(batch_flat)
    print(f"   Running... [{i + len(current_indices)} / {TARGET_N} 완료]")

# 결과 통합
shap_vals_total = np.concatenate(all_shap_values, axis=0)
test_data_total = np.concatenate(all_test_data, axis=0)

# ----------------------------------------------------
# 5. 시각화 및 저장
# ----------------------------------------------------
print("\n📈 DA-RNN SHAP 결과 저장 중...")

# 3D 변환 및 평균
shap_vals_3d = shap_vals_total.reshape(-1, TIME_STEPS, FEATURE_COUNT)
shap_vals_mean = shap_vals_3d.mean(axis=1)
test_data_3d = test_data_total.reshape(-1, TIME_STEPS, FEATURE_COUNT)
test_data_mean = test_data_3d.mean(axis=1)

# Bar Plot
plt.figure()
shap.summary_plot(shap_vals_mean, test_data_mean, feature_names=feature_names, plot_type="bar", show=False)
plt.title("DA-RNN 모델 전체 변수 중요도 (SHAP)")
plt.tight_layout()
plt.savefig('darnn_global_shap_bar.png', dpi=300)
print("  --> 저장 완료: darnn_global_shap_bar.png")

# Dot Plot
plt.figure()
shap.summary_plot(shap_vals_mean, test_data_mean, feature_names=feature_names, show=False)
plt.title("DA-RNN 모델 변수 영향력 분포 (SHAP)")
plt.tight_layout()
plt.savefig('darnn_global_shap_dot.png', dpi=300)
print("  --> 저장 완료: darnn_global_shap_dot.png")

print("\n🎉 DA-RNN 분석 완료!")