"""深度模型（model.py）的训练：标准化、带掩码的多任务损失、早停。交叉验证和最终模型都调用这里。"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import EggSeqModel
from synth_data import EggDataset

BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEV_LOSS_WEIGHT = 5.0       # 发育回归损失的权重 λ；没有实测发育指标时，训练流程会把它设成 0
PRIMARY_DEEP = "深度-累计"   # 拿去和基线比、也是部署时用的深度模型输出：用到当天为止的全部图像


@dataclass(frozen=True)
class Normalizer:
    """每个波段的均值和标准差，只用训练部分计算，预测时原样套用。"""
    mean: np.ndarray        # (1, C, 1, 1)
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, mask: np.ndarray) -> "Normalizer":
        frames = x[mask]                                    # 只用有数据的帧
        return cls(frames.mean(axis=(0, 2, 3), keepdims=True),
                   frames.std(axis=(0, 2, 3), keepdims=True) + 1e-6)

    def apply(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """标准化，缺失的帧置零。"""
        return ((x - self.mean) * mask[:, :, None, None, None] / self.std).astype(np.float32)


@dataclass(frozen=True)
class TrainedModel:
    model: EggSeqModel
    norm: Normalizer
    epochs: int             # 训练了几轮（早停时是最好的那一轮）


def _tensors(data: EggDataset, idx: np.ndarray, norm: Normalizer) -> TensorDataset:
    return TensorDataset(torch.from_numpy(norm.apply(data.x[idx], data.mask[idx])),
                         torch.from_numpy(data.mask[idx]),
                         torch.from_numpy(data.sex[idx].astype(np.int64)),
                         torch.from_numpy(data.dev[idx].astype(np.float32)),
                         torch.from_numpy(data.dev_mask[idx] & data.mask[idx]))


def _loader(data: EggDataset, idx: np.ndarray, norm: Normalizer, seed: int | None = None) -> DataLoader:
    """给 seed 就打乱顺序（训练用），不给就按原顺序（验证用）。"""
    ds = _tensors(data, idx, norm)
    if seed is None:
        return DataLoader(ds, BATCH_SIZE)
    return DataLoader(ds, BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))


def cum_valid(mask: torch.Tensor) -> torch.Tensor:
    """累计输出在哪些位置有定义：当天或之前至少拍过一帧。训练的损失和预测的 nan 都用这一条规则。

    没拍的那天，单向 GRU 靠之前的隐藏状态（只含当天及以前的数据）给出判断，不会看到后面的帧。
    """
    return torch.cumsum(mask.long(), dim=1) > 0


def _masked_ce(logits: torch.Tensor, sex: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    t = logits.shape[1]
    target = sex.unsqueeze(1).expand(-1, t)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none")
    m = mask.flatten().float()
    return (loss * m).sum() / m.sum().clamp(min=1)


def _loss(out: dict[str, torch.Tensor], mask: torch.Tensor, sex: torch.Tensor, dev: torch.Tensor,
          dev_mask: torch.Tensor, dev_weight: float) -> torch.Tensor:
    # 单帧和发育只在有图像的位置监督；累计在 cum_valid 的位置监督（和预测时的规则一致）
    loss = _masked_ce(out["frame_sex"], sex, mask) + _masked_ce(out["cum_sex"], sex, cum_valid(mask))
    if dev_weight > 0 and dev_mask.any():
        # 先把缺失值换成 0 再选：NaN × 0 仍是 NaN，反向传播也会带出 NaN
        target = torch.nan_to_num(dev)
        sq = torch.where(dev_mask, (out["dev"] - target) ** 2, torch.zeros_like(target))
        loss = loss + dev_weight * sq.sum() / dev_mask.sum()
    return loss


def _run_epoch(model: EggSeqModel, loader: DataLoader, days: torch.Tensor, dev_weight: float,
               opt: torch.optim.Optimizer | None = None) -> float:
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, mask, sex, dev, dev_mask in loader:
            loss = _loss(model(x, mask, days), mask, sex, dev, dev_mask, dev_weight)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def _new_model(data: EggDataset) -> tuple[EggSeqModel, torch.optim.Optimizer]:
    model = EggSeqModel(n_bands=data.x.shape[2])
    return model, torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def train_early_stop(data: EggDataset, fit_idx: np.ndarray, val_idx: np.ndarray,
                     dev_weight: float, seed: int) -> TrainedModel:
    """在 fit_idx 上训练；val_idx 上的损失连续 PATIENCE 轮不降就停，取最好的那一轮。"""
    torch.manual_seed(seed)
    norm = Normalizer.fit(data.x[fit_idx], data.mask[fit_idx])
    model, opt = _new_model(data)
    days = torch.from_numpy(data.days)
    train_dl, val_dl = _loader(data, fit_idx, norm, seed), _loader(data, val_idx, norm)
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        _run_epoch(model, train_dl, days, dev_weight, opt)
        val_loss = _run_epoch(model, val_dl, days, dev_weight)
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("验证损失一直算不出来（nan），训练失败")
    model.load_state_dict(best_state)
    return TrainedModel(model, norm, best_epoch)


def train_fixed_epochs(data: EggDataset, idx: np.ndarray, dev_weight: float, seed: int,
                       epochs: int) -> TrainedModel:
    """最终模型：用全部数据训练固定轮数（轮数由交叉验证决定），不再留验证集。"""
    torch.manual_seed(seed)
    norm = Normalizer.fit(data.x[idx], data.mask[idx])
    model, opt = _new_model(data)
    days = torch.from_numpy(data.days)
    train_dl = _loader(data, idx, norm, seed)
    for _ in range(epochs):
        _run_epoch(model, train_dl, days, dev_weight, opt)
    return TrainedModel(model, norm, epochs)


def predict_scores(trained: TrainedModel, x: np.ndarray, mask: np.ndarray,
                   days: np.ndarray) -> dict[str, np.ndarray]:
    """返回 frame（单帧雄概率）、cum（累计雄概率）、dev（发育预测），形状都是 (N, T)。

    判断规则（验证和预测共用）：单帧和发育只在当天有图像时给出；
    累计只要当天或之前拍过就给出（cum_valid，当天没拍就用之前几天的图像）。给不出的位置是 nan。
    """
    mask = np.asarray(mask, dtype=bool)
    trained.model.eval()
    with torch.no_grad():
        out = trained.model(torch.from_numpy(trained.norm.apply(x, mask)), torch.from_numpy(mask),
                            torch.from_numpy(np.asarray(days, dtype=np.float32)))
    seen = cum_valid(torch.from_numpy(mask)).numpy()
    return {"frame": np.where(mask, out["frame_sex"].softmax(-1)[..., 1].numpy(), np.nan),
            "cum": np.where(seen, out["cum_sex"].softmax(-1)[..., 1].numpy(), np.nan),
            "dev": np.where(mask, out["dev"].numpy(), np.nan)}
