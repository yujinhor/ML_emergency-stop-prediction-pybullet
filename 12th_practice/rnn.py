import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# [핵심] torchtext 대신 Keras를 사용해 데이터를 불러옵니다 (에러 해결!)
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------
# 하이퍼파라미터 (수정 O)
# ----------------------------------------------------
epochs = 60
batch_size = 256
learning_rate = 0.001
MAX_VOCAB_SIZE = 10000  # 단어장 크기 (빈도수 상위 1만개만 사용) 영화를 평론하는 문징
MAX_LEN = 200       # 문장 길이 (Time Step) -> 10 단어로 고정  문장이 길어도 10개만 보겠당
EMBED_DIM = 256          # 임베딩 차원 (Feature)
HIDDEN_SIZE = 256       # RNN 기억 용량

# GPU 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ----------------------------------------------------
# 1. 데이터 준비 (수정 X)
# ----------------------------------------------------
print("IMDb 데이터 다운로드 및 전처리 중...")

# (1) 데이터 로드 (이미 정수로 변환되어 있어 토크나이저 불필요)
# num_words: 빈도수 높은 상위 10,000개 단어만 가져옴
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_VOCAB_SIZE)

# (2) 길이 맞추기 (Padding)
# 짧으면 0을 채우고, 길면 자름
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post')
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post')

# (3) PyTorch 텐서로 변환 (LongTensor)
x_train_tensor = torch.tensor(x_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# (4) DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------------------------------
# 2. 신경망 모델 정의 (PyTorch) (수정 O)
# ----------------------------------------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(SimpleRNN, self).__init__()

        # Embedding: 정수(단어ID) -> 벡터(Feature)  단어 아이디에서 벡터값을 뽑아옴
        # padding_idx=0: 숫자 0은 학습하지 않음 (패딩의 의미)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # 수정 안하시는 걸 추천드려요

        # RNN 층
        self.rnn = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, batch_first = True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x input shape: (Batch, Time Step)

        # Parctice
        x = self.embedding(x)

        output,(h_n, c_n)  = self.rnn(x)
        # 마지막 히든 스테이트만 가져오기
        last_hidden = h_n[0] # 수정 안하시는 걸 추천드려요

        return self.fc(last_hidden)

# 모델 생성
model = SimpleRNN(vocab_size=MAX_VOCAB_SIZE,
                  embed_dim=EMBED_DIM,
                  hidden_size=HIDDEN_SIZE,
                  output_size=2).to(device) # 0(부정), 1(긍정) -> 2개 클래스

# ----------------------------------------------------
# 3. 학습 설정 및 루프 (수정 O)
# ----------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


# ----------------------------------------------------
# 4. 평가 (수정 X)
# ----------------------------------------------------
print("\nStarting Evaluation...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

if accuracy > 85:
    print("✅ 분류 성공!")
else:
    print(f"❌ 분류 실패!")