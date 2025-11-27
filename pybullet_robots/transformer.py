import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.colab import files
import io
import copy


# ==========================================
# 1. 파일 업로드 (4개 파일 동시 업로드)
# ==========================================
print("Step 1: 데이터 파일 4개를 모두 업로드해주세요.")
print("(train_steps.csv, train_summary.csv, val_steps.csv, val_summary.csv)")
uploaded = files.upload()

# 파일 분류
files_map = {
    'train_steps': None, 'train_summary': None,
    'val_steps': None, 'val_summary': None
}

for filename in uploaded.keys():
    if "train" in filename and "steps" in filename: files_map['train_steps'] = filename
    elif "train" in filename and "summary" in filename: files_map['train_summary'] = filename
    elif "val" in filename and "steps" in filename: files_map['val_steps'] = filename
    elif "val" in filename and "summary" in filename: files_map['val_summary'] = filename

# 파일 누락 확인
if not all(files_map.values()):
    print("Error: 4개의 파일 중 일부가 누락되었습니다. 파일명을 확인해주세요.")
    print(f"현재 인식된 파일: {files_map}")
else:
    print("모든 파일이 준비되었습니다.")

# ==========================================
# 2. 데이터셋 클래스 (Train/Val 지원)
# ==========================================
class BlackIceDataset(Dataset):
    def __init__(self, step_file_content, summary_file_content, feature_columns=None, forced_max_len=None, scaler=None):
        """
        Args:
            forced_max_len: 검증 데이터셋을 만들 때 학습 데이터셋의 최대 길이를 강제로 적용하기 위함
            scaler: 학습 데이터셋의 스케일러를 검증 데이터셋에도 똑같이 적용하기 위함
        """
        # CSV 읽기
        df_steps = pd.read_csv(io.BytesIO(step_file_content))
        df_summary = pd.read_csv(io.BytesIO(summary_file_content))
        
        # 질량 병합 (summary -> steps)
        if 'mass' not in df_steps.columns and 'mass_kg' in df_summary.columns:
            mass_info = df_summary[['episode', 'mass_kg']].rename(columns={'mass_kg': 'mass'})
            df = pd.merge(df_steps, mass_info, on='episode', how='left')
        else:
            df = df_steps

        # 사용할 변수
        if feature_columns is None:
            self.feature_cols = ['mass', 'speed', 'friction', 'drag_force_N', 'trigger_dist_m', 'time']
        else:
            self.feature_cols = feature_columns

        self.label_col = 'result_is_failure'

        # 정규화 (Scaler) - Train의 분포로 Val도 정규화해야 함
        if scaler is None:
            self.scaler = StandardScaler()
            df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        else:
            self.scaler = scaler
            df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])

        # 데이터 처리
        self.episodes = []
        self.labels = []
        temp_lengths = []

        grouped = df.groupby('episode')
        
        for ep_id, group in grouped:
            features = group[self.feature_cols].values
            label_val = group[self.label_col].iloc[0]
            
            if label_val == -1: continue 
            
            # [다운샘플링] 240Hz -> 10Hz (메모리 절약)
            if len(features) > 1000:
                features = features[::24] 
            
            self.episodes.append(features)
            self.labels.append(1.0 if label_val == 1 else 0.0)
            temp_lengths.append(len(features))

        # 최대 길이 결정 (Train 기준)
        if forced_max_len is not None:
            self.max_seq_len = forced_max_len
        else:
            self.max_seq_len = max(temp_lengths) if temp_lengths else 0
            print(f"Dataset Max Length set to: {self.max_seq_len}")

        # 패딩 및 자르기 (Padding / Truncating)
        final_episodes = []
        for feat in self.episodes:
            L, C = feat.shape
            
            # 1. 길이가 짧으면 패딩
            if L < self.max_seq_len:
                pad = np.zeros((self.max_seq_len - L, C))
                feat = np.vstack((feat, pad))
            
            # 2. 길이가 길면 자르기 (Validation 데이터가 Train보다 길 수도 있음)
            feat = feat[:self.max_seq_len, :]
            
            final_episodes.append(feat)

        self.X = torch.FloatTensor(np.array(final_episodes)) 
        self.y = torch.FloatTensor(np.array(self.labels)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. 모델 정의 (PatchTST 구조)
# ==========================================
class BlackIceTransformer(nn.Module):
    def __init__(self, num_vars=6, max_seq_len=200, patch_len=8, stride=4, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(BlackIceTransformer, self).__init__()
        self.num_vars = num_vars
        
        self.patch_embedding = nn.Conv1d(
            in_channels=1, out_channels=d_model, kernel_size=patch_len, stride=stride
        )
        
        # Positional Encoding
        max_num_patches = (max_seq_len - patch_len) // stride + 10
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, max_num_patches))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_vars * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # (B, L, 6) -> (B*6, 1, L)
        x = x.permute(0, 2, 1).contiguous().view(batch_size * self.num_vars, 1, -1) 
        
        x = self.patch_embedding(x) 
        
        # PE Slicing
        curr_num_patches = x.shape[-1]
        if curr_num_patches > self.pos_embedding.shape[-1]:
             x = x[:, :, :self.pos_embedding.shape[-1]]
             curr_num_patches = x.shape[-1]

        x = x + self.pos_embedding[:, :, :curr_num_patches]
        
        x = x.permute(0, 2, 1) 
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)
        x = x.view(batch_size, -1)
        return self.head(x)

# ==========================================
# 4. 학습 및 검증 실행
# ==========================================
def main():
    if not all(files_map.values()): return

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # 1. 학습 데이터셋 생성
    print("\n[Loading Train Data]")
    train_dataset = BlackIceDataset(
        uploaded[files_map['train_steps']], 
        uploaded[files_map['train_summary']]
    )

    # 2. 검증 데이터셋 생성
    # 중요: Train의 max_len과 scaler를 Val에 적용해야 차원과 분포가 맞음
    print("\n[Loading Validation Data]")
    val_dataset = BlackIceDataset(
        uploaded[files_map['val_steps']], 
        uploaded[files_map['val_summary']],
        forced_max_len=train_dataset.max_seq_len, # 길이 동기화
        scaler=train_dataset.scaler               # 스케일러 동기화
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"\nTrain Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}")

    # 모델 초기화
    model = BlackIceTransformer(
        num_vars=6, 
        max_seq_len=train_dataset.max_seq_len + 20, 
        d_model=64
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    EPOCHS = 50
    best_val_acc = 0.0

    print("\n--- Training Start ---")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                out = model(X_batch)
                val_loss += criterion(out, y_batch).item()
                
                # Accuracy Calc
                pred = (torch.sigmoid(out) > 0.5).float()
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
        
        acc = 100 * correct / total if total > 0 else 0
        
        # Best Model Check
        if acc > best_val_acc:
            best_val_acc = acc
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | "
                  f"Val Acc: {acc:.2f}%")

    print(f"\n[Final Result] Best Validation Accuracy: {best_val_acc:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), "black_ice_transformer_final.pth")
    print("모델이 저장되었습니다: black_ice_transformer_final.pth")
    files.download("black_ice_transformer_final.pth")

if __name__ == "__main__":
    main()
