"""
model_stage2_new.py — STN/TPS 亚像素对齐模型（增强版）
=====================================================

架构：
    ColorQRStage2New
    ├── LocalizationNet  (MobileNetV3-Small backbone → 53×2 关键点预测)
    └── TPSSpatialTransformer  (TPS 薄板样条空间变换，输入图 → 标准化对齐图)

TPS 原理：
  控制点 ctrl_pts 固定为标准 V40 量子坐标（canonical space，[-1,1]）。
  LocalizationNet 预测 pred_pts：这些控制点在输入失真图中的对应位置（[-1,1]）。
  TPS 求解 F(ctrl_pts[i]) = pred_pts[i] 的薄板样条函数 F: R²→R²。
  对输出规则网格中每个像素查询 F，得到在输入图中的采样坐标，送入 grid_sample。

性能优化：
  L_inv（TPS 系数矩阵逆）和 Q_grid（网格基函数矩阵）均在 __init__ 中预计算，
  推理时仅做矩阵乘法，无需在线求逆，批次效率高。
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

NUM_KPT = 53
OUT_SIZE = 800
HEATMAP_UPSCALE = 4.0

# ─────────────────────────────────────────────────────────────────────────────
# Version-40 关键点布局（与 generate_stage2_dataset.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────
QR_MODULE_COUNT = 177
QR_BOX_SIZE = 4
QR_BORDER = 1
QR_ALIGN_POS = [6, 30, 58, 86, 114, 142, 170]
_FINDER_OVERLAP = {(6, 6), (6, 170), (170, 6)}
IDX_TL, IDX_TR, IDX_BR, IDX_BL = 49, 50, 51, 52


def _module_center_px(row: int, col: int):
    x = (col + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    y = (row + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    return float(x), float(y)


def _get_base_kpts() -> np.ndarray:
    """返回 716×716 base 图中 53 个关键点的像素坐标 (53,2)。"""
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
    # 四角点定义为白色外边框四角（整张 base 图边界）
    outer_start = 0.0
    outer_end = float((QR_MODULE_COUNT + 2 * QR_BORDER) * QR_BOX_SIZE - 1)  # 715
    kpts.append((outer_start, outer_start))
    kpts.append((outer_end, outer_start))
    kpts.append((outer_end, outer_end))
    kpts.append((outer_start, outer_end))
    assert len(kpts) == NUM_KPT
    return np.array(kpts, dtype=np.float32)


def _compute_canon_kpts_norm(out_size: int = OUT_SIZE) -> np.ndarray:
    """
    计算 53 个关键点在 out_size×out_size 标准图中的归一化坐标 ([-1,1])。

    方法：将 base 图中的四角点作为透视变换源，映射到 out_size×out_size 四角，
    用同一透视矩阵变换全部 53 个点，然后按 align_corners=True 规范归一化。
    """
    base_kpts = _get_base_kpts()  # (53,2)
    corners_src = base_kpts[[IDX_TL, IDX_TR, IDX_BR, IDX_BL]]  # (4,2)
    s = float(out_size - 1)
    corners_dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
    H_canon = cv2.getPerspectiveTransform(corners_src, corners_dst)

    pts = base_kpts.reshape(-1, 1, 2)
    canon_512 = cv2.perspectiveTransform(pts, H_canon).reshape(NUM_KPT, 2)

    half = (out_size - 1) / 2.0  # 255.5 for 512px
    canon_norm = (canon_512 / half) - 1.0
    return canon_norm.astype(np.float32)  # (53,2), ~ [-1,1]


# 模块级预计算（所有实例共享）
CANON_KPTS_NORM: np.ndarray = _compute_canon_kpts_norm()


# ─────────────────────────────────────────────────────────────────────────────
# TPS Spatial Transformer
# ─────────────────────────────────────────────────────────────────────────────
class TPSSpatialTransformer(nn.Module):
    """
    薄板样条（Thin-Plate Spline）空间变换器。

    预计算（__init__，仅一次）：
      L_inv   (N+3, N+3)  TPS 系数矩阵的逆
      Q_grid  (H*W, N+3)  输出规则网格的 TPS 基函数矩阵

    推理时（forward）：
      theta  = L_inv @ [pred_pts; 0]   ——  TPS 系数  (B, N+3, 2)
      grid   = Q_grid @ theta           ——  采样网格  (B, H, W, 2)
      output = grid_sample(img, grid)   ——  矫正图像  (B, C, H, W)
    """

    def __init__(
        self,
        ctrl_pts: np.ndarray,  # (N,2)  canonical 控制点，固定，[-1,1]
        grid_h: int = OUT_SIZE,
        grid_w: int = OUT_SIZE,
        reg: float = 1e-6,
    ):
        super().__init__()
        N = ctrl_pts.shape[0]
        self.N = N
        self.grid_h = grid_h
        self.grid_w = grid_w

        # ── 构建 TPS 系数矩阵 L 并存其逆 ────────────────────────────────────
        # K_ij = U(||c_i - c_j||)，U(r) = r² log(r²)
        diff_c = ctrl_pts[:, None, :] - ctrl_pts[None, :, :]  # (N,N,2)
        r2_c = (diff_c**2).sum(-1).astype(np.float64)  # (N,N)
        K = r2_c * np.log(r2_c + 1e-12)
        np.fill_diagonal(K, 0.0)

        P = np.hstack(
            [np.ones((N, 1), dtype=np.float64), ctrl_pts.astype(np.float64)]
        )  # (N,3)

        L = np.zeros((N + 3, N + 3), dtype=np.float64)
        L[:N, :N] = K + reg * np.eye(N, dtype=np.float64)
        L[:N, N:] = P
        L[N:, :N] = P.T

        L_inv = np.linalg.inv(L).astype(np.float32)  # (N+3, N+3)
        self.register_buffer("L_inv", torch.from_numpy(L_inv))

        # ── 预建 dense grid 的 TPS 基函数矩阵 Q_grid ────────────────────────
        ys = np.linspace(-1.0, 1.0, grid_h, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, grid_w, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)  # (H,W)
        flat = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (H*W,2)

        diff_g = flat[:, None, :] - ctrl_pts[None, :, :]  # (H*W,N,2)
        r2_g = (diff_g**2).sum(-1).astype(np.float64)  # (H*W,N)
        phi = (r2_g * np.log(r2_g + 1e-12)).astype(np.float32)

        ones = np.ones((flat.shape[0], 1), dtype=np.float32)
        Q_grid = np.hstack([phi, ones, flat])  # (H*W, N+3)
        self.register_buffer("Q_grid", torch.from_numpy(Q_grid))

    def forward(self, img: torch.Tensor, pred_pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img      (B, C, H, W)  输入失真图像，float [0,1]
            pred_pts (B, N, 2)     预测关键点在输入图中的坐标，[-1,1]
        Returns:
            (B, C, H, W)  TPS 矫正后的图像
        """
        B = pred_pts.shape[0]
        device = pred_pts.device
        dtype = pred_pts.dtype

        # rhs: (B, N+3, 2)
        zeros3 = torch.zeros(B, 3, 2, device=device, dtype=dtype)
        rhs = torch.cat([pred_pts, zeros3], dim=1)

        # TPS 系数: theta = L_inv @ rhs → (B, N+3, 2)
        L_inv = self.L_inv.to(device=device, dtype=dtype)
        theta = torch.einsum("ij,bjk->bik", L_inv, rhs)

        # 采样网格: grid_flat = Q_grid @ theta → (B, H*W, 2)
        Q_grid = self.Q_grid.to(device=device, dtype=dtype)
        grid = torch.einsum("ij,bjk->bik", Q_grid, theta)  # (B, H*W, 2)
        grid = grid.view(B, self.grid_h, self.grid_w, 2)

        return F.grid_sample(
            img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class SpatialSoftArgmax2D(nn.Module):
    """
    二维可微积分回归（Spatial Soft-Argmax）。

    - 支持可学习温度系数，默认初始化为 0.1，使概率分布更尖锐。
    - 支持动态网格大小（与上采样后的热力图分辨率一致）。
    """

    def __init__(self, grid_h: int = 400, grid_w: int = 400, init_temp: float = 0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(init_temp))
        self.register_buffer("grid", self._make_grid(grid_h, grid_w))

    @staticmethod
    def _make_grid(grid_h: int, grid_w: int) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, steps=grid_h, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, steps=grid_w, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=0)  # (2, H*W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)

        # 约束温度为正，避免除零；温度越低，分布越尖锐。
        temp = torch.clamp(self.temperature, min=1e-4)
        logits = x_flat / temp
        logits = logits - logits.amax(dim=-1, keepdim=True)
        prob = F.softmax(logits, dim=-1)

        if self.grid.shape[1] != h * w or self.grid.device != x.device:
            grid = self._make_grid(h, w).to(device=x.device, dtype=x.dtype)
        else:
            grid = self.grid.to(dtype=x.dtype)

        expected_coords = torch.einsum("bcn,dn->bcd", prob, grid)
        return expected_coords


# ─────────────────────────────────────────────────────────────────────────────
# Localization Network — MobileNetV3-Small Backbone
# ─────────────────────────────────────────────────────────────────────────────
class LocalizationNet(nn.Module):
    """
    输入：(B, 3, H, W) 失真 QR 图，float [0,1]。
    输出：(B, 53, 2) 预测关键点坐标，范围 [-1,1]。

        使用 ImageNet 预训练的 MobileNetV3-Small 作为特征提取 backbone，
        采用“高分辨率积分回归（Integral Regression）”：
            features(1/32) -> deconv neck(上采样到1/8) -> heatmap head
            -> bilinear 无参上采样(×4) -> Spatial Soft-Argmax。

    相比 GAP+FC：
      1) 保留空间结构信息，避免退化为绝对坐标记忆；
      2) 新增参数极少（仅 1x1 conv）；
      3) 输出为连续坐标，端到端可微，梯度可回传至 backbone。
    """

    FEATURE_DIM = 576  # MobileNetV3-Small 最后特征图通道
    NECK_DIM1 = 256
    NECK_DIM2 = 128

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.mobilenet_v3_small(weights=weights)

        self.features = backbone.features  # (B, 576, H/32, W/32)

        # 轻量上采样 neck：1/32 -> 1/16 -> 1/8
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(
                self.FEATURE_DIM,
                self.NECK_DIM1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.NECK_DIM1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                self.NECK_DIM1,
                self.NECK_DIM2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.NECK_DIM2),
            nn.ReLU(inplace=True),
        )

        # 热力图 head：输出 NUM_KPT 张热力图
        self.heatmap_head = nn.Conv2d(self.NECK_DIM2, NUM_KPT, kernel_size=1, bias=True)
        self.soft_argmax = SpatialSoftArgmax2D(grid_h=400, grid_w=400, init_temp=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)  # (B, 576, H/32, W/32)
        feat = self.neck(feat)  # (B, 128, H/8, W/8)
        heatmaps = self.heatmap_head(feat)  # (B, 53, H/8, W/8)
        heatmaps_up = F.interpolate(
            heatmaps,
            scale_factor=HEATMAP_UPSCALE,
            mode="bilinear",
            align_corners=False,
        )
        coords = self.soft_argmax(heatmaps_up)  # (B, 53, 2)
        return coords

    def forward_with_heatmaps(self, x: torch.Tensor):
        """返回坐标与上采样后的 heatmaps，便于调试可视化。"""
        feat = self.features(x)  # (B, 576, H/32, W/32)
        feat = self.neck(feat)  # (B, 128, H/8, W/8)
        heatmaps = self.heatmap_head(feat)  # (B, 53, H/8, W/8)
        heatmaps_up = F.interpolate(
            heatmaps,
            scale_factor=HEATMAP_UPSCALE,
            mode="bilinear",
            align_corners=False,
        )
        coords = self.soft_argmax(heatmaps_up)  # (B, 53, 2)
        return coords, heatmaps_up


# ─────────────────────────────────────────────────────────────────────────────
# 完整 Stage 2 模型
# ─────────────────────────────────────────────────────────────────────────────
class ColorQRStage2New(nn.Module):
    """
    Stage 2 STN/TPS 亚像素对齐模型。

    Forward 输入：
    img  (B, 3, H, W) — Stage 1 裁剪的失真 QR 图，float [0,1]

    Forward 输出（tuple）：
      pred_pts   (B, 53, 2) — 预测关键点坐标（输入图空间，[-1,1]）
    rectified  (B, 3, OUT_SIZE, OUT_SIZE) — TPS 矫正后的标准对齐图，float [0,1]
    """

    def __init__(
        self,
        pretrained_locnet: bool = True,
        out_size: int = OUT_SIZE,
    ):
        super().__init__()
        self.locnet = LocalizationNet(pretrained=pretrained_locnet)
        self.tps = TPSSpatialTransformer(
            ctrl_pts=CANON_KPTS_NORM,  # (53,2) fixed
            grid_h=out_size,
            grid_w=out_size,
        )

    def forward(self, img: torch.Tensor, return_heatmaps: bool = False):
        if return_heatmaps:
            pred_pts, heatmaps = self.locnet.forward_with_heatmaps(img)
            rectified = self.tps(img, pred_pts)  # (B, 3, out_size, out_size)
            return pred_pts, rectified, heatmaps

        pred_pts = self.locnet(img)  # (B, 53, 2)
        rectified = self.tps(img, pred_pts)  # (B, 3, out_size, out_size)
        return pred_pts, rectified
