import torch
import numpy as np
from tcn.model import TCN

class TCNOccPredictor:
    def __init__(self, model_path, device='cuda', seq_len=72, pred_steps=36,
                 nhid=128, levels=8, ksize=5, dropout=0.2):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.buffer = []

        # 缓存逻辑
        self.cached_preds = []
        self.cache_idx = 0

        # 构建和训练时一致的模型
        channel_sizes = [nhid] * levels
        self.model = TCN(
            input_size=4,
            output_size=4 * pred_steps,
            num_channels=channel_sizes,
            kernel_size=ksize,
            dropout=dropout
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def add_frame_feature(self, bbox4d):
        """bbox4d: (x,y,w,h)，逐帧加入"""
        self.buffer.append(bbox4d)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def _predict_once(self, buffer=None):
        """
        用最近的 seq_len 帧预测未来 pred_steps 帧
        """
        if buffer is None:
            buffer = self.buffer
        if len(buffer) < self.seq_len:
            return []

        seq = np.array(buffer[-self.seq_len:], dtype=np.float32)  # [seq_len, 4]
        x = torch.tensor(seq, dtype=torch.float32, device=self.device)  # [seq_len, 4]
        x = x.unsqueeze(0).transpose(1, 2)  # [1, 4, seq_len]

        with torch.no_grad():
            out = self.model(x)  # [1, seq_len, 4*pred_steps]
            out_last = out[:, -1, :]  # [1, 4*pred_steps]
            preds = out_last.view(self.pred_steps, 4).cpu().numpy()  # [pred_steps, 4]

        return preds.tolist()

    def predict_future_recursive(self, steps=72):
        """
        一次性预测未来 steps 帧（递归拼接多段 pred_steps）
        """
        if len(self.buffer) < self.seq_len:
            return []

        preds_all = []
        buffer = list(self.buffer)  # 拷贝，避免污染原始 buffer

        while len(preds_all) < steps:
            preds = self._predict_once(buffer)
            if len(preds) == 0:
                break
            preds_all.extend(preds)
            buffer.extend(preds)  # 用预测值继续预测

        return np.array(preds_all[:steps])

    def get_next_cached(self):
        """
        自动递归预测 + 缓存逐帧取结果
        """
        # 如果缓存耗尽，递归预测下一批
        if self.cache_idx >= len(self.cached_preds):
            new_preds = self._predict_once()
            if len(new_preds) == 0:
                return None
            self.cached_preds = new_preds
            self.cache_idx = 0
            # 把预测结果同步进 buffer，保证能继续递归
            for bbox in new_preds:
                self.buffer.append(bbox)
                if len(self.buffer) > self.seq_len:
                    self.buffer.pop(0)

        pred = self.cached_preds[self.cache_idx]
        self.cache_idx += 1
        return pred

    def clear_cache(self):
        """清空TOP的预测缓存与历史输入"""
        self.cached_preds = []  # 清空预测缓存
        self.cache_idx = 0  # 重置索引
        self.buffer = []  # 清空输入序列缓存（重新开始）

