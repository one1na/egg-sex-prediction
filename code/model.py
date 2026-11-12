"""单帧编码器 + 时序模块的多任务模型。

输入 (B, T, C, H, W)，输出三组结果：
- frame_sex:  (B, T, 2) 只看当天这一帧的性别判断（单帧鉴定）
- cum_sex:    (B, T, 2) 综合第 0 天到当天所有帧的性别判断（时序鉴定）
- dev:        (B, T)    当天的发育指数预测（发育预测）
"""
import torch
import torch.nn as nn

FEAT_DIM = 32
HIDDEN_DIM = 48
MAX_DAY = 10.0
CHANNELS_PER_GROUP = 4      # GroupNorm 每组的通道数


def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm 只用这一帧自己的统计量：训练时不同蛋、不同天的帧不会互相影响（BatchNorm 会）。"""
    return nn.GroupNorm(channels // CHANNELS_PER_GROUP, channels)


class FrameEncoder(nn.Module):
    """先在每个像素上做波段混合（1x1 卷积），再做空间卷积，最后全局池化。每帧单独编码。"""

    def __init__(self, n_bands: int, feat_dim: int = FEAT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bands, 16, kernel_size=1), _norm(16), nn.ReLU(),
            nn.Conv2d(16, feat_dim, kernel_size=3, padding=1), _norm(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class EggSeqModel(nn.Module):
    def __init__(self, n_bands: int) -> None:
        super().__init__()
        self.encoder = FrameEncoder(n_bands)
        self.frame_sex_head = nn.Linear(FEAT_DIM, 2)
        # 输入 = 帧特征 + 孵化日 + 是否有数据
        self.gru = nn.GRU(FEAT_DIM + 2, HIDDEN_DIM, batch_first=True)
        self.cum_sex_head = nn.Linear(HIDDEN_DIM, 2)
        self.dev_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                days: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = x.shape[:2]
        feats = self.encoder(x.flatten(0, 1)).view(b, t, -1)
        feats = feats * mask.unsqueeze(-1)  # 缺失的天置零

        day_feat = (days / MAX_DAY).expand(b, t).unsqueeze(-1)
        gru_in = torch.cat([feats, day_feat, mask.unsqueeze(-1).float()], dim=-1)
        hidden, _ = self.gru(gru_in)  # 单向 GRU：第 t 天只用到 t 及之前的数据

        return {
            "frame_sex": self.frame_sex_head(feats),
            "cum_sex": self.cum_sex_head(hidden),
            "dev": self.dev_head(hidden).squeeze(-1),
        }
