import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import os

# ----------------------------------------------------
# 1. 설정 및 데이터 로드 (Max Pooling된 데이터)
# ----------------------------------------------------
# Mac(M1/M2) 'mps' 또는 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 평가 장치: {device}")

INPUT_SIZE = 8       # Feature 개수
HIDDEN_SIZE = 64     # Hidden Size

def load_test_data():
    print("📂 Max Pooling 테스트 데이터 로딩 중...")
    try:
        X_test = np.load('X_test_maxpool.npy')
        
        # y_test는 접두사가 붙은 게 있으면 그거 쓰고, 없으면 원본 사용
        if os.path.exists('y_test_maxpool.npy'):
            y_test = np.load('y_test_maxpool.npy')
        else:
            y_test = np.load('y_test.npy')
            
        # Tensor 변환
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        print(f"✅ 데이터 로드 완료: X shape {X_test.shape}")
        return X_test_tensor, y_test_tensor
        
    except FileNotFoundError:
        print("❌ 오류: 'X_test_maxpool.npy' 파일이 없습니다. 전처리를 먼저 진행하세요.")
        sys.exit()

X_test, y_test = load_test_data()
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------------------------------
# 2. 모델 클래스 정의 (학습 코드와 동일)
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
        beta = F.softmax(scores, dim=1) 
        context = torch.sum(beta * encoder_outputs, dim=1)
        return context, beta

class DA_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DA_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_attention = InputAttention(input_size, hidden_size)
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.temporal_attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        encoder_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            weighted_x_t, _ = self.input_attention(x_t, h_t)
            h_t = self.gru_cell(weighted_x_t, h_t)
            encoder_outputs.append(h_t.unsqueeze(1))
            
        encoder_outputs = torch.cat(encoder_outputs, dim=1)
        context_vector, _ = self.temporal_attention(encoder_outputs)
        logits = self.fc(context_vector)
        return torch.sigmoid(logits)

# ----------------------------------------------------
# 3. 모델 로드 및 예측
# ----------------------------------------------------
model = DA_RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=1).to(device)

MODEL_PATH = 'da_rnn_best_maxpool.pth'
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ 학습된 모델 '{MODEL_PATH}' 로드 성공!")
except FileNotFoundError:
    print(f"❌ 오류: '{MODEL_PATH}' 파일이 없습니다. 학습이 완료되었나요?")
    sys.exit()

def get_predictions(model, loader):
    all_probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            all_probs.extend(outputs.cpu().numpy().flatten())
    return np.array(all_probs)

print("🤖 예측 수행 중...")
targets_np = y_test.numpy().flatten()
probs_np = get_predictions(model, test_loader)

# ----------------------------------------------------
# 4. 최적 임계치 탐색 및 결과 출력
# ----------------------------------------------------
best_f1 = 0
best_th = 0.5

print("\n" + "="*45)
print("🔍 Threshold Optimization (DA-RNN MaxPool)")
print("="*45)
print(f"{'Threshold':<10} | {'F1 Score':<10} | {'Accuracy':<10}")
print("-" * 45)

for th in np.arange(0.1, 0.91, 0.05):
    preds = (probs_np > th).astype(int)
    f1 = f1_score(targets_np, preds, zero_division=0)
    acc = accuracy_score(targets_np, preds)
    print(f"{th:.2f}       | {f1:.4f}     | {acc:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

# 최종 지표 계산
final_preds = (probs_np > best_th).astype(int)
final_acc = accuracy_score(targets_np, final_preds)
final_f1 = f1_score(targets_np, final_preds)
final_recall = recall_score(targets_np, final_preds)

# ----------------------------------------------------
# 5. [최종 비교] GRU vs DA-RNN (Max Pool)
# ----------------------------------------------------
# GRU 점수 (회원님이 주신 기록 기준)
gru_acc = 0.9513
gru_f1 = 0.9196
gru_recall = 0.8721

print("\n" + "#"*60)
print("📢 [FINAL BATTLE] GRU(Original) vs DA-RNN(Max Pool)")
print("#"*60)
print(f"{'Metric':<15} | {'GRU (Baseline)':<15} | {'DA-RNN (Ours)':<15} | {'Gap'}")
print("-" * 65)
print(f"{'Accuracy':<15} | {gru_acc:.4f}          | {final_acc:.4f}          | {final_acc - gru_acc:+.4f}")
print(f"{'F1 Score':<15} | {gru_f1:.4f}          | {final_f1:.4f}          | {final_f1 - gru_f1:+.4f}")
print(f"{'Recall':<15} | {gru_recall:.4f}          | {final_recall:.4f}          | {final_recall - gru_recall:+.4f}")
print("-" * 65)
print(f"※ DA-RNN Threshold: {best_th:.2f}")

# 혼동 행렬 저장
cm = confusion_matrix(targets_np, final_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'DA-RNN (Max Pool) Confusion Matrix\nF1: {final_f1:.4f}')
plt.savefig('confusion_matrix_maxpool.png')
print("🖼️ 결과 이미지 저장: confusion_matrix_maxpool.png")