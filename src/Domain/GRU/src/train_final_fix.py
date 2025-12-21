import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler  # 표준화 도구 추가
import sys
import os

# ----------------------------------------------------
# 1. 하이퍼파라미터 및 설정
# ----------------------------------------------------
INPUT_SIZE = 8        # Brake_Torque 제거로 8개
HIDDEN_SIZE = 64      
NUM_LAYERS = 2        
OUTPUT_SIZE = 1       

SEQ_LEN = 2400        
BATCH_SIZE = 32       
LEARNING_RATE = 0.001
EPOCHS = 20
TIME_PENALTY = 1.0 

POS_WEIGHT_VAL = 2.0  # 황금 비율 유지

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): 
    device = torch.device("mps")
    print("🚀 Mac MPS 가속 활성화됨")
else:
    print(f"Using device: {device}")


# ----------------------------------------------------
# 2. 손실함수
# ----------------------------------------------------
class TimeAwareBCELoss(nn.Module):
    def __init__(self, time_penalty_weight=1.0, pos_weight=1.0):
        super(TimeAwareBCELoss, self).__init__()
        self.time_penalty_weight = time_penalty_weight
        self.pos_weight = pos_weight

    def forward(self, predictions, targets, inputs):
        if targets.dim() == 1: targets = targets.view(-1, 1)
        targets_expanded = targets.unsqueeze(1).expand_as(predictions)
        
        weights = torch.ones_like(targets_expanded)
        weights[targets_expanded == 1] = self.pos_weight
        
        bce_loss = F.binary_cross_entropy(predictions, targets_expanded, weight=weights, reduction='none')
        
        batch_size, seq_len, _ = predictions.shape
        time_steps = torch.linspace(0, 1, seq_len, device=predictions.device).view(1, -1, 1)
        time_weights = 1.0 + (self.time_penalty_weight * time_steps)
        weighted_loss = bce_loss * time_weights
        
        mask = (torch.abs(inputs).sum(dim=-1, keepdim=True) > 0).float()
        masked_loss = weighted_loss * mask
        
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
        return loss


# ----------------------------------------------------
# 3. 데이터 로딩 및 표준화 (핵심 변경!)
# ----------------------------------------------------
def load_and_scale_data():
    print("📂 데이터 로딩 중...")
    X_train_np = np.load('X_train.npy')
    X_test_np = np.load('X_test.npy')
    
    # 1. 죽은 변수(Brake_Torque, 7번) 제거
    print("✂️ 'Brake_Torque' 변수 제거 중...")
    X_train_np = np.delete(X_train_np, 7, axis=2)
    X_test_np = np.delete(X_test_np, 7, axis=2)
    
    # 2. 표준화 (StandardScaler) 적용
    # 3차원(Sample, Time, Feature) -> 2차원(Sample*Time, Feature)로 펴서 스케일링 후 다시 복구
    print("⚖️ 데이터 표준화(StandardScaler) 적용 중...")
    
    scaler = StandardScaler()
    
    N, T, F = X_train_np.shape
    # Train 데이터로 피팅 (평균, 분산 계산)
    X_train_2d = X_train_np.reshape(N * T, F)
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_train_np = X_train_scaled.reshape(N, T, F) # 다시 3차원으로 복구
    
    # Test 데이터는 Train의 기준으로 변환만 (Transform)
    N_test, T_test, _ = X_test_np.shape
    X_test_2d = X_test_np.reshape(N_test * T_test, F)
    X_test_scaled = scaler.transform(X_test_2d)
    X_test_np = X_test_scaled.reshape(N_test, T_test, F)
    
    print("✅ 데이터 전처리 완료!")
    
    # Tensor 변환
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(np.load('y_train.npy'), dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(np.load('y_test.npy'), dtype=torch.float32).unsqueeze(1)
    
    return X_train, y_train, X_test, y_test

# 실행
X_train, y_train, X_test, y_test = load_and_scale_data()

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ----------------------------------------------------
# 4. 모델 정의
# ----------------------------------------------------
class RobotGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RobotGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        logits = self.fc(out)
        probs = torch.sigmoid(logits) 
        return probs

model = RobotGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)


# ----------------------------------------------------
# 5. 학습 루프
# ----------------------------------------------------
criterion = TimeAwareBCELoss(time_penalty_weight=TIME_PENALTY, pos_weight=POS_WEIGHT_VAL)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0.0
best_model_path = 'gru_model_best.pth'
final_model_path = 'gru_model_last.pth'

def train_epoch(model, loader, epoch_idx):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch_idx+1} Train", leave=False)
    
    for inputs, targets in loop:
        inputs_dev, targets_dev = inputs.to(device), targets.to(device)
        outputs = model(inputs_dev)
        loss = criterion(outputs.cpu(), targets.cpu(), inputs.cpu())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs_dev = inputs.to(device)
            outputs = model(inputs_dev)
            
            loss = criterion(outputs.cpu(), targets.cpu(), inputs.cpu())
            total_loss += loss.item()
            
            final_preds = (outputs[:, -1, :] > 0.5).float().cpu()
            all_preds.extend(final_preds.numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = 100 * np.mean(all_preds == all_targets)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return total_loss / len(loader), acc, f1

print("\n[Training Start] 학습 시작 (Brake 제거 + 표준화 + 가중치)...")

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, epoch)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Loss: {train_loss:.4f} | "
          f"Acc: {test_acc:.2f}% | "
          f"F1 Score: {test_f1:.4f}")
    
    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> 🎉 F1 최고점 갱신! 모델 저장됨 (F1: {best_f1:.4f})")

print("\n✅ 학습 종료.")
torch.save(model.state_dict(), final_model_path)
print(f"💾 마지막 모델 저장됨: {final_model_path}")