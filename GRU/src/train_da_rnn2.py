import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import sys

# ---------------------------------------------------------
# 1. 환경 설정 및 데이터 로드
# ---------------------------------------------------------
# Mac(M1/M2) 사용시 'mps' 사용 가능하면 사용, 아니면 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 학습 장치(Device): {device}")

# 하이퍼파라미터 설정
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001 # 학습이 너무 빠르면 0.0005로 줄이세요
BATCH_SIZE = 64
EPOCHS = 50           # 데이터가 줄어서 Epoch를 넉넉히 주셔도 빠릅니다

def load_data():
    print("\n📂 Max Pooling 데이터 로딩 중...")
    try:
        # 1. Max Pooling된 X 데이터 로드
        X_train = np.load('X_train_maxpool.npy')
        X_test = np.load('X_test_maxpool.npy')
        
        # 2. y 데이터 로드 (y는 pooling 영향 없으므로 원본 혹은 복사본 사용)
        # 만약 y_train.npy가 없다면 원본 생성 코드를 확인해주세요.
        if os.path.exists('y_train.npy'):
            y_train = np.load('y_train.npy')
        else:
            # y_train이 없으면 임시방편으로 y_test와 같은 로직으로 가정 (실제로는 y_train 필수)
            print("⚠️ 경고: 'y_train.npy'를 찾을 수 없습니다. 원본 폴더 확인 필요.")
            sys.exit()

        if os.path.exists('y_test_maxpool.npy'):
            y_test = np.load('y_test_maxpool.npy')
        else:
            y_test = np.load('y_test.npy')

        print(f"✅ 데이터 로드 완료!")
        print(f"   X_train: {X_train.shape} (Time Step이 줄어들었는지 확인!)")
        print(f"   y_train: {y_train.shape}")
        
        return X_train, y_train, X_test, y_test

    except FileNotFoundError as e:
        print(f"❌ 오류: 파일이 없습니다. {e}")
        sys.exit()

X_train_np, y_train_np, X_test_np, y_test_np = load_data()

# Tensor 변환
train_data = TensorDataset(torch.FloatTensor(X_train_np), torch.FloatTensor(y_train_np))
test_data = TensorDataset(torch.FloatTensor(X_test_np), torch.FloatTensor(y_test_np))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 입력 차원 자동 설정 (Feature 개수)
INPUT_SIZE = X_train_np.shape[2] 

# ---------------------------------------------------------
# 2. 모델 정의 (들여쓰기 수정 완료됨)
# ---------------------------------------------------------
class InputAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputAttention, self).__init__()
        self.linear = nn.Linear(hidden_size + input_size, input_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_t, h_prev):
        # x_t: (Batch, Input), h_prev: (Batch, Hidden)
        combined = torch.cat((x_t, h_prev), dim=1)
        scores = torch.tanh(self.linear(combined))
        alpha = self.softmax(scores) 
        return alpha * x_t, alpha

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        # encoder_outputs: (Batch, Seq, Hidden)
        scores = torch.tanh(self.linear(encoder_outputs))
        beta = F.softmax(scores, dim=1) 
        context = torch.sum(beta * encoder_outputs, dim=1)
        return context, beta

class DA_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DA_RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Input Attention + Encoder
        self.input_attention = InputAttention(input_size, hidden_size)
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        
        # 2. Temporal Attention
        self.temporal_attention = TemporalAttention(hidden_size)
        
        # 3. Classifier
        self.fc = nn.Linear(hidden_size, output_size)
    
    # [수정됨] forward가 __init__ 밖으로 나왔습니다.
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        encoder_outputs = []
        
        # [Stage 1] Loop over time steps
        for t in range(seq_len):
            x_t = x[:, t, :]
            weighted_x_t, _ = self.input_attention(x_t, h_t)
            h_t = self.gru_cell(weighted_x_t, h_t)
            encoder_outputs.append(h_t.unsqueeze(1))
            
        encoder_outputs = torch.cat(encoder_outputs, dim=1)
        
        # [Stage 2] Temporal Attention
        context_vector, _ = self.temporal_attention(encoder_outputs)
        
        # [Stage 3] Prediction
        logits = self.fc(context_vector)
        return torch.sigmoid(logits)

# ---------------------------------------------------------
# 3. 학습 루프 (Training Loop)
# ---------------------------------------------------------
model = DA_RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=1).to(device)
criterion = nn.BCELoss() # 모델이 sigmoid를 포함하고 있으므로 BCELoss 사용
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n🔥 학습 시작! (Epochs: {EPOCHS})")
print("-" * 60)

train_losses = []
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 차원 맞추기 (Batch, 1) -> (Batch,)
        outputs = outputs.squeeze()
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # 진행 상황 출력 (5 epoch 마다)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.5f}")

    # Best Model 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'da_rnn_best_maxpool.pth')

print("-" * 60)
print(f"✅ 학습 완료! Best Loss: {best_loss:.5f}")
print("💾 모델 저장 완료: da_rnn_best_maxpool.pth")

# ---------------------------------------------------------
# 4. 학습 결과 시각화
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('DA-RNN Training Loss (with Max Pooling Data)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()