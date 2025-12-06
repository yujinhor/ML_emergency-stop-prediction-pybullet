class BasePredictionModel(nn.Module):
    """
    공통 인터페이스를 위한 추상 모델
    - forward(x): 로짓(logits) 반환
    - predict_proba(x): softmax 확률
    - predict(x): argmax 클래스
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, num_features)
        return: (B, num_classes)
        """
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, num_features)
        return: (B,)  # 예측 클래스 인덱스
        """
        proba = self.predict_proba(x)