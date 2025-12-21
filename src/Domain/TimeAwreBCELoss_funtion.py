import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 손실함수 정의
class TimeAwareBCELoss(nn.Module):
    def __init__(self, time_penalty_weight=1.0):

        #time_penalty_weight 이건 하이퍼파라메터로 값 조정해서 정보에 중요시 할건지, 시간에 중요시 할건지를 판단해야함.

        """
        Args:
            time_penalty_weight (float): 시간 흐름에 따른 가중치 강도
                - 0.0 : 시간 가중치 없음 (기존 BCE)
                - 1.0 : 종료 시점 오차가 시작 시점보다 약 2배 더 중요
                - 숫자가 클수록 '후반부 오차'를 더 엄격하게 패널티 줌
        """

        super(TimeAwareBCELoss, self).__init__()
        self.time_penalty_weight = time_penalty_weight

    def forward(self, predictions, targets, inputs):

        """
        Args:
            predictions : (Batch, Seq_Len, 1) - 모델 예측 확률 (Sigmoid 후)
            targets     : (Batch, 1)          - 정답 레이블 (0 or 1)
            inputs      : (Batch, Seq_Len, Features) - 원본 입력 데이터 (마스킹용)
            
        Returns:
            loss        : scalar값
        """

        # 1. 정답(Target) 차원 확장 
        # (Batch, 1) -> (Batch, Seq_Len, 1)로 모든 시점의 정답을 만든다. 
        if targets.dim() == 1:
            targets = targets.view(-1, 1)
        targets_expanded = targets.unsqueeze(1).expand_as(predictions)
        
        # 2. 시점별 BCE Loss 계산 (), Reduction (손실값을 줄이는 인수) 없이
        # 결과: (Batch, Seq_Len, 1)
        bce_loss = F.binary_cross_entropy(predictions, targets_expanded, reduction='none')
        
        # 3. 시간 가중치 (Time Weighting) 적용
        batch_size, seq_len, _ = predictions.shape
        device = predictions.device
        
        # 0 ~ 1 사이로 정규화된 시간축 (0, ..., 1)
        time_steps = torch.linspace(0, 1, seq_len, device=device).view(1, -1, 1)
        
        # 시간이 지날수록 가중치 증가 (초반:1.0 , 후반: (1.0 + weight) 배로 증가)
        time_weights = 1.0 + (self.time_penalty_weight * time_steps)
        
        weighted_loss = bce_loss * time_weights
        
        # 4. 패딩 마스킹 (Masking) - 각 에피소드에서 뒤쪽 0(패딩) 무시하기
        # 입력 feature들의 합이 0이 아니면 실제 데이터(1), 0이면 패딩(0)
        mask = (torch.abs(inputs).sum(dim=-1, keepdim=True) > 0).float()
        
        # 마스크 적용 (패딩 구간 Loss = 0)
        masked_loss = weighted_loss * mask
        
        # 5. 최종 평균 (패딩 개수는 분모에서 제외)
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
            
        return loss
    

#학습할 때 아래와 유사하게 사용하시면 돼요!

'''
# 2. 모델과 손실함수 정의
model = MyGRUModel(...) # return_sequences=True 설정 필수!
criterion = TimeAwareBCELoss(time_penalty_weight=1.0) # 가중치는 하이퍼 파마티러로 조절 가능 (논문에선 0.8 사용)

# 3. 학습 루프
for inputs, targets in train_loader:
    # inputs: (Batch, 2400, Features)
    # targets: (Batch, 1)
    
    # 모델 예측
    # outputs: (Batch, 2400, 1) -> 매 순간의 확률
    outputs = model(inputs) 
    
    # 손실 계산 (inputs를 같이 넘겨줘야 마스킹 작동!)
    loss = criterion(outputs, targets, inputs)
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    '''