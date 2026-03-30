"""
train_stage2.py — Stage 2 STN/TPS 亚像素对齐训练脚本
=====================================================

数据源 : train/stage2_dataset/images/*.jpg   (512×512 失真 QR)
         train/stage2_dataset/labels/*.npy   ((53,2) float32, [-1,1])
模型   : ColorQRStage2 (MobileNetV3-Small + TPS)
损失   : 加权 Wing Loss（预测关键点坐标 vs GT 关键点坐标）
优化器 : Adam, lr=1e-4
可视化 : 每 vis_every 步保存对比图：
           左图 = 原始失真输入 + GT 关键点(绿) + 预测关键点(红)
           右图 = TPS 矫正后图像 + canonical 关键点(蓝)
"""

import os
import math
import glob
import sys
import csv
import time
import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# 同目录导入 Stage 2 模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_stage2 import ColorQRStage2

NUM_KPT = 40
OUT_SIZE = 512

# Version-30 标签布局（3 finder + 33 alignment + 4 corners）
QR_MODULE_COUNT = 137
QR_BOX_SIZE = 4
QR_BORDER = 1
QR_ALIGN_POS = [6, 26, 52, 78, 104, 130]
_FINDER_OVERLAP = {(6, 6), (6, 130), (130, 6)}
IDX_TL, IDX_TR, IDX_BR, IDX_BL = 36, 37, 38, 39

# 关键点加权：3个 Finder 中心 + 4个角点权重更高，减少全局漂移
FINDER_INDICES = [0, 1, 2]
CORNER_INDICES = [36, 37, 38, 39]
FINDER_WEIGHT = 4.5
CORNER_WEIGHT = 7.0

# Wing Loss 参数（关键点高精度定位）
WING_W = 10.0
WING_EPSILON = 2.0


def _module_center_px(row: int, col: int):
    x = (col + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    y = (row + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    return float(x), float(y)


def _get_base_kpts_v30() -> np.ndarray:
    kpts = []
    kpts.append(_module_center_px(3, 3))
    kpts.append(_module_center_px(3, QR_MODULE_COUNT - 4))
    kpts.append(_module_center_px(QR_MODULE_COUNT - 4, 3))
    mc = QR_MODULE_COUNT
    for i in QR_ALIGN_POS:
        for j in QR_ALIGN_POS:
            if (i, j) in _FINDER_OVERLAP:
                continue
            if i <= 8 and j <= 8:
                continue
            if i <= 8 and j >= mc - 8:
                continue
            if i >= mc - 8 and j <= 8:
                continue
            kpts.append(_module_center_px(i, j))

    outer_start = 0.0
    outer_end = float((QR_MODULE_COUNT + 2 * QR_BORDER) * QR_BOX_SIZE - 1)
    kpts.append((outer_start, outer_start))
    kpts.append((outer_end, outer_start))
    kpts.append((outer_end, outer_end))
    kpts.append((outer_start, outer_end))
    assert len(kpts) == NUM_KPT
    return np.array(kpts, dtype=np.float32)


def _compute_canon_kpts_norm_v30(out_size: int = OUT_SIZE) -> np.ndarray:
    base_kpts = _get_base_kpts_v30()
    corners_src = base_kpts[[IDX_TL, IDX_TR, IDX_BR, IDX_BL]]
    s = float(out_size - 1)
    corners_dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
    h_canon = cv2.getPerspectiveTransform(corners_src, corners_dst)

    pts = base_kpts.reshape(-1, 1, 2)
    canon = cv2.perspectiveTransform(pts, h_canon).reshape(NUM_KPT, 2)
    half = (out_size - 1) / 2.0
    return (canon / half - 1.0).astype(np.float32)


CANON_KPTS_NORM = _compute_canon_kpts_norm_v30()


def _safe_torch_save(obj, path: str, retries: int = 3, retry_delay: float = 0.6):
    """
    原子保存并带重试，降低 Windows 上文件被瞬时占用导致的保存失败概率。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    last_err = None

    for i in range(retries):
        tmp_path = f"{path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if i < retries - 1:
                time.sleep(retry_delay * (i + 1))

    raise RuntimeError(f"failed to save checkpoint: {path}") from last_err


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class Stage2Dataset(Dataset):
    """
    每个样本：
      img   : (3, 512, 512) float32  [0, 1]
      kpts  : (53, 2)       float32  [-1, 1]
    """

    def __init__(self, img_dir: str, label_dir: str):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.label_dir = label_dir
        if not self.img_files:
            raise RuntimeError(
                f"No images found in {img_dir}. "
                "Run train/generate_stage2_dataset.py first."
            )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.label_dir, f"{stem}.npy")

        img = Image.open(img_path).convert("RGB")
        img_t = TF.to_tensor(img)  # (3,512,512) float [0,1]
        kpts = np.load(lbl_path).astype(np.float32)
        if kpts.shape != (NUM_KPT, 2):
            raise RuntimeError(
                f"Label shape mismatch for {lbl_path}: expected ({NUM_KPT},2), got {kpts.shape}. "
                "Please regenerate stage2_dataset with Version-30 format."
            )
        return img_t, torch.from_numpy(kpts)


# ─────────────────────────────────────────────────────────────────────────────
# CSV + 曲线图日志工具
# ─────────────────────────────────────────────────────────────────────────────


class TrainingLogger:
    """
    记录每个 epoch 的训练指标到 results.csv，并实时绘制 loss 曲线图。
    格式与 Ultralytics YOLO 的 results.csv 风格保持一致，方便对比。
    """

    FIELDS = [
        "epoch",
        "train/wing_loss",
        "train/wing_small_ratio",
        "train/wing_large_ratio",
        "lr",
    ]

    def __init__(self, save_dir: str):
        self.csv_path = os.path.join(save_dir, "results.csv")
        self.plot_path = os.path.join(save_dir, "results.png")
        self.breakdown_plot_path = os.path.join(save_dir, "wing_piecewise_ratio.png")
        self.rows: list = []
        # 写表头
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.FIELDS)

    def log(
        self,
        epoch: int,
        loss: float,
        wing_small_ratio: float,
        wing_large_ratio: float,
        lr: float,
    ):
        row = {
            "epoch": epoch + 1,
            "train/wing_loss": f"{loss:.8f}",
            "train/wing_small_ratio": f"{wing_small_ratio:.8f}",
            "train/wing_large_ratio": f"{wing_large_ratio:.8f}",
            "lr": f"{lr:.2e}",
        }
        self.rows.append(row)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
        if _HAS_MPL:
            self._plot()

    def _plot(self):
        epochs = [r["epoch"] for r in self.rows]
        losses = [float(r["train/wing_loss"]) for r in self.rows]
        small_ratios = [float(r["train/wing_small_ratio"]) for r in self.rows]
        large_ratios = [float(r["train/wing_large_ratio"]) for r in self.rows]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, losses, linewidth=1.5, color="#2196F3")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Wing Loss")
        ax.set_title("Stage2 STN — Training Wing Loss")
        ax.grid(True, alpha=0.3)
        # 标注最低点
        min_loss = min(losses)
        min_ep = epochs[losses.index(min_loss)]
        ax.annotate(
            f"best={min_loss:.6f}\n@ep{min_ep}",
            xy=(min_ep, min_loss),
            xytext=(min_ep, min_loss * 1.05),
            fontsize=8,
            color="#E53935",
        )
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=100)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(
            epochs,
            [r * 100.0 for r in small_ratios],
            linewidth=1.8,
            color="#2E7D32",
            label="small-error branch (<w)",
        )
        ax2.plot(
            epochs,
            [r * 100.0 for r in large_ratios],
            linewidth=1.8,
            color="#E53935",
            label="large-error branch (>=w)",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Contribution (%)")
        ax2.set_ylim(0.0, 100.0)
        ax2.set_title("Wing Piecewise Contribution Ratio")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        fig2.tight_layout()
        fig2.savefig(self.breakdown_plot_path, dpi=100)
        plt.close(fig2)


def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] tensor → HWC uint8 BGR ndarray"""
    arr = t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _draw_kpts(
    bgr: np.ndarray,
    kpts_norm: np.ndarray,
    color: tuple = (0, 255, 0),
    radius: int = 3,
    out_size: int = OUT_SIZE,
) -> np.ndarray:
    half = (out_size - 1) / 2.0
    img = bgr.copy()
    for xn, yn in kpts_norm:
        x = int(round((float(xn) + 1.0) * half))
        y = int(round((float(yn) + 1.0) * half))
        if 0 <= x < out_size and 0 <= y < out_size:
            cv2.circle(img, (x, y), radius, color, -1)
    return img


def save_comparison(
    img_t: torch.Tensor,  # (3,H,W) 原始失真输入
    rect_t: torch.Tensor,  # (3,H,W) TPS 矫正后
    gt_kpts: torch.Tensor,  # (53,2)  GT 关键点
    pred_kpts: torch.Tensor,  # (53,2)  预测关键点
    save_path: str,
):
    """
    左图：输入图 + GT(绿) + 预测(红)
    右图：TPS 矫正图 + canonical 标准点(蓝橙)
    保存为横拼 JPEG。
    """
    before = _tensor_to_bgr(img_t)
    after = _tensor_to_bgr(rect_t)

    gt_np = gt_kpts.detach().cpu().numpy()
    pred_np = pred_kpts.detach().cpu().numpy()

    vis_before = _draw_kpts(before, gt_np, color=(0, 200, 0))  # GT    = 绿
    vis_before = _draw_kpts(vis_before, pred_np, color=(0, 0, 220))  # pred  = 红

    # 矫正图叠加 canonical 标准点（蓝橙色）
    vis_after = _draw_kpts(after, CANON_KPTS_NORM, color=(255, 128, 0))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, np.hstack([vis_before, vis_after]))


def _build_kpt_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """构建 (53,) 权重向量。"""
    w = torch.ones(NUM_KPT, device=device, dtype=dtype)
    w[FINDER_INDICES] = FINDER_WEIGHT
    w[CORNER_INDICES] = CORNER_WEIGHT
    return w


def _weighted_kpt_wing(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    kpt_weights: torch.Tensor,
    w: float = WING_W,
    epsilon: float = WING_EPSILON,
) -> torch.Tensor:
    """
    加权 Wing Loss。
    对小误差区间使用对数形态，提升亚像素级别回归敏感性；
    对大误差区间保持线性，抑制异常点的梯度爆炸。

    x = |pred - target|
    c = w * (1 - log(1 + w / epsilon))
    if x < w:   wing = w * log(1 + x / epsilon)
    else:       wing = x - c
    """
    x = torch.abs(pred_kpts - gt_kpts)  # (B,53,2)
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    wing = torch.where(
        x < w,
        w * torch.log(1.0 + x / epsilon),
        x - c,
    )

    # 分段贡献，用于可视化统计
    wing_small = torch.where(x < w, wing, torch.zeros_like(wing))
    wing_large = torch.where(x < w, torch.zeros_like(wing), wing)

    # (B,53)
    per_kpt_err = wing.mean(dim=-1)
    per_kpt_small = wing_small.mean(dim=-1)
    per_kpt_large = wing_large.mean(dim=-1)

    # (B,) -> scalar
    denom = kpt_weights.sum() + 1e-12
    weighted_err = (per_kpt_err * kpt_weights.unsqueeze(0)).sum(dim=1) / denom
    weighted_small = (per_kpt_small * kpt_weights.unsqueeze(0)).sum(dim=1) / denom
    weighted_large = (per_kpt_large * kpt_weights.unsqueeze(0)).sum(dim=1) / denom

    loss = weighted_err.mean()
    small_contrib = weighted_small.mean()
    large_contrib = weighted_large.mean()
    return loss, small_contrib, large_contrib


# ─────────────────────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────────────────────
def train_stage2(
    dataset_dir: str = "train/stage2_dataset",
    save_dir: str = "train/runs/stage2_stn",
    epochs: int = 1500,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_workers: int = 0,
    vis_every: int = 100,  # 每隔多少 step 保存可视化
    save_every: int = 10,  # 每隔多少 epoch 保存 checkpoint
    device_str: str = "auto",
):
    # ── Device ───────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Stage2-STN] device={device}")

    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "vis")
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = TrainingLogger(save_dir)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")
    ds = Stage2Dataset(img_dir, lbl_dir)
    print(f"[Stage2-STN] {len(ds)} training samples")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ColorQRStage2().to(device)
    print(
        f"[Stage2-STN] model params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 优化器与调度器需在 resume 之前创建，避免引用未定义变量
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    best_loss = float("inf")

    kpt_weights = _build_kpt_weights(device=device, dtype=torch.float32)
    print(
        "[Stage2-STN] weighted loss enabled: "
        f"finder={FINDER_WEIGHT:.1f}, corners={CORNER_WEIGHT:.1f}, others=1.0"
    )
    print(
        "[Stage2-STN] strict loss enabled: "
        f"type=Weighted-Wing, w={WING_W:.1f}, eps={WING_EPSILON:.1f}"
    )

    # Resume from latest checkpoint if available
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    start_epoch = 0
    if ckpt_files:
        try:
            state = torch.load(ckpt_files[-1], map_location=device)
            model.load_state_dict(state["model"])
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            best_loss = state.get("best_loss", float("inf"))
            start_epoch = state["epoch"] + 1
            print(
                f"[Stage2-STN] resumed from {ckpt_files[-1]}, "
                f"start_epoch={start_epoch}  best_loss={best_loss:.6f}"
            )
        except RuntimeError as e:
            print(
                "[Stage2-STN] checkpoint incompatible with current setup, "
                f"starting fresh: {e}"
            )

    # ── Training Loop ─────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_small_contrib = 0.0
        epoch_large_contrib = 0.0
        n_samples = 0

        for batch_idx, (imgs, gt_kpts) in enumerate(loader):
            imgs = imgs.to(device)  # (B, 3, 512, 512)
            gt_kpts = gt_kpts.to(device)  # (B, NUM_KPT, 2)

            # ── Forward ──────────────────────────────────────────────────────
            pred_kpts, rectified = model(imgs)
            if pred_kpts.shape[1] < NUM_KPT:
                raise RuntimeError(
                    f"Model output keypoints ({pred_kpts.shape[1]}) < required NUM_KPT ({NUM_KPT})"
                )
            if pred_kpts.shape[1] != NUM_KPT:
                pred_kpts = pred_kpts[:, :NUM_KPT, :]

            # ── Loss = 严格加权 Wing Loss ────────────────────────────────────
            loss, small_contrib, large_contrib = _weighted_kpt_wing(
                pred_kpts, gt_kpts, kpt_weights
            )

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            bs = imgs.shape[0]
            n_samples += bs
            epoch_small_contrib += float(small_contrib.item()) * bs
            epoch_large_contrib += float(large_contrib.item()) * bs
            global_step += 1

            # ── 可视化：变换前 vs TPS 矫正后 ─────────────────────────────────
            if global_step % vis_every == 0:
                model.eval()
                with torch.no_grad():
                    p_kpts_vis, rect_vis = model(imgs[:1])
                    if p_kpts_vis.shape[1] != NUM_KPT:
                        p_kpts_vis = p_kpts_vis[:, :NUM_KPT, :]
                vis_path = os.path.join(vis_dir, f"step_{global_step:07d}.jpg")
                save_comparison(
                    imgs[0],
                    rect_vis[0],
                    gt_kpts[0],
                    p_kpts_vis[0],
                    vis_path,
                )
                model.train()
                print(
                    f"  [vis] step={global_step:7d}  "
                    f"loss={loss.item():.6f}  → {vis_path}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_small_contrib = epoch_small_contrib / max(n_samples, 1)
        avg_large_contrib = epoch_large_contrib / max(n_samples, 1)
        contrib_sum = max(avg_small_contrib + avg_large_contrib, 1e-12)
        small_ratio = avg_small_contrib / contrib_sum
        large_ratio = avg_large_contrib / contrib_sum
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:4d}/{epochs}]  avg_loss={avg_loss:.6f}  "
            f"small={small_ratio*100.0:.1f}% large={large_ratio*100.0:.1f}%  "
            f"lr={cur_lr:.2e}"
        )

        # ── CSV + 曲线图 ──────────────────────────────────────────────────────
        logger.log(epoch, avg_loss, small_ratio, large_ratio, cur_lr)

        # ── LR Scheduler ─────────────────────────────────────────────────────
        scheduler.step()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            _safe_torch_save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                },
                ckpt_path,
            )
            print(f"  [ckpt] saved → {ckpt_path}")

        # ── Best 模型追踪 ─────────────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best.pt")
            _safe_torch_save(model.state_dict(), best_path)
            print(f"  [best] loss={best_loss:.6f} → {best_path}")

    # ── Final export ─────────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "stage2_stn_final.pt")
    _safe_torch_save(model.state_dict(), final_path)
    print(f"[Stage2-STN] Training complete → {final_path}")


if __name__ == "__main__":
    train_stage2()
