import torch
import torch.nn as nn
import numpy as np
from tcn.tcn import TemporalConvNet

class TCNClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=2):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, [64, 64, 64, 64, 64])
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        y = self.tcn(x)  # [B, C, T]
        return self.linear(y.transpose(1, 2))  # [B, T, num_classes]

class TCNOccClassifier:
    def __init__(self, model_path="best_model.pt", window_size=32, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.model = TCNClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.feature_buffer = []

    def add_features(self, score, iou, area_ratio):
        self.feature_buffer.append([score, iou, area_ratio])
        if len(self.feature_buffer) > self.window_size:
            self.feature_buffer.pop(0)

    def predict(self):
        if len(self.feature_buffer) < self.window_size:
            return None  # Not enough data
        x = np.array(self.feature_buffer).astype(np.float32).T  # [3, T]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 3, T]
        with torch.no_grad():
            logits = self.model(x)  # [1, T, 2]
            pred = logits[0, -1].argmax().item()  # 取最后一帧的预测
        return pred  # 0 or 1