"""
generate_stage2_dataset.py — Stage 2 训练数据生成（自包含，不依赖已训练模型）
==============================================================================

完整流程（每张 base QR 图生成 N 个样本）：

  1. 从 base_images/ 读取原始 716×716 二维码图像（纯色彩 QR，黑底白底均可）
     2. 填充黑色边框 → 随机旋转（默认 ±15°）→ 仍保持旋转后 QR 完整可见
     3. 从 53 个 GT 关键点中取 4 个角点 [49,50,51,52]，注入默认 ±12% 图像尺寸随机扰动
     （模拟 Stage 1 预测的定位误差）
  4. cv2.getPerspectiveTransform + warpPerspective → 512×512 矩形裁剪，
      同时对目标角点引入随机全局平移与缩放（二维码占屏约 70%~95%，不再固定居中）
  5. 在 512×512 裁剪中用同一透视矩阵计算所有 53 个精确 GT 关键点坐标，
     归一化到 [-1, 1]（align_corners=True 约定，与 PyTorch grid_sample 一致）
  6. 叠加轻微色彩偏移（Color Jitter）+ 屏幕反光模拟（Glare）
  7. 保存 JPEG 图像 + .npy 关键点文件（shape=(53,2), dtype=float32, range=[-1,1]）

输出目录：train/stage2_dataset/images/  &  train/stage2_dataset/labels/
  - images/  : *.jpg（512×512）
  - labels/  : *.npy（(53,2) float32，关键点坐标归一化到 [-1,1]）
"""

import argparse
import os
import glob
import math
import random
from collections import deque

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Version-40 关键点布局常量（与 process_dataset.py / gen_base.rs 保持一致）
# ─────────────────────────────────────────────────────────────────────────────
QR_MODULE_COUNT = 177
QR_BOX_SIZE = 4
QR_BORDER = 1
QR_IMG_SIZE = (QR_MODULE_COUNT + 2 * QR_BORDER) * QR_BOX_SIZE  # 716

QR_ALIGN_POS = [6, 30, 58, 86, 114, 142, 170]
_FINDER_OVERLAP = {(6, 6), (6, 170), (170, 6)}

# 53 个关键点索引定义
IDX_F0, IDX_F1, IDX_F2 = 0, 1, 2  # Finder 中心（3 个）
IDX_ALIGN_START, IDX_ALIGN_END = 3, 49  # Alignment 中心（[3..48]，46 个）
IDX_TL, IDX_TR, IDX_BR, IDX_BL = 49, 50, 51, 52  # 4 角点
NUM_KPT = 53

# Stage 2 模型预测的关键点数（全量 53 个）
NUM_KPT_STAGE2 = 53


def _module_center_px(row: int, col: int) -> tuple:
    """返回 base 图坐标系中模块 (row, col) 的像素中心 (x, y)。"""
    x = (col + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    y = (row + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    return float(x), float(y)


def get_base_keypoints() -> np.ndarray:
    """
    返回 716×716 base 图坐标系中 53 个关键点的 (x, y) 数组，形状 (53, 2)。
    顺序：[Red Finder, Green Finder, Blue Finder,
           46× Alignment (行优先), TL, TR, BR, BL]
    """
    kpts = []

    # ── Finder Pattern 中心（3 个）────────────────────────────────────────────
    kpts.append(_module_center_px(3, 3))  # Red   [0]
    kpts.append(_module_center_px(3, QR_MODULE_COUNT - 4))  # Green [1]
    kpts.append(_module_center_px(QR_MODULE_COUNT - 4, 3))  # Blue  [2]

    # ── Alignment Pattern 中心（46 个，行优先排列）────────────────────────────
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

    assert len(kpts) == 49, f"Expected 49 kpts before corners, got {len(kpts)}"

    # ── 四个角点（4 个）：白色外边框四角（整张 base 图边界）──────────────────────
    outer_start = 0.0
    outer_end = float(QR_IMG_SIZE - 1)  # 715.0
    kpts.append((outer_start, outer_start))  # TL [49]
    kpts.append((outer_end, outer_start))  # TR [50]
    kpts.append((outer_end, outer_end))  # BR [51]
    kpts.append((outer_start, outer_end))  # BL [52]

    assert len(kpts) == NUM_KPT, f"Expected {NUM_KPT} kpts, got {len(kpts)}"
    return np.array(kpts, dtype=np.float32)  # (53, 2)


# 在模块加载时计算，所有样本共享，避免重复计算
BASE_KPTS: np.ndarray = get_base_keypoints()  # (53, 2)，base 图坐标系


# ─────────────────────────────────────────────────────────────────────────────
# 几何变换辅助
# ─────────────────────────────────────────────────────────────────────────────


def _pad_to_fit_rotation(img: np.ndarray, max_angle_deg: float):
    """
    在图像四周填充黑色边框，使旋转后 QR 内容仍完整可见。
    返回 (padded_img, pad_x, pad_y)，pad_x/pad_y 为水平/垂直填充量（像素）。
    """
    h, w = img.shape[:2]
    rad = math.radians(max_angle_deg)
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    new_w = int(math.ceil(c * w + s * h))
    new_h = int(math.ceil(s * w + c * h))
    pad_x = max(0, (new_w - w + 1) // 2 + 2)
    pad_y = max(0, (new_h - h + 1) // 2 + 2)
    padded = cv2.copyMakeBorder(
        img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded, pad_x, pad_y


def _apply_rotation(img: np.ndarray, kpts: np.ndarray, angle_deg: float) -> tuple:
    """
    以图像中心为轴旋转图像，背景填充黑色。
    同步用仿射矩阵变换关键点坐标。
    返回 (rotated_img, new_kpts)。
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)  # 2×3
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # 变换关键点：[x', y'] = M @ [x, y, 1]^T
    ones = np.ones((kpts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([kpts, ones])  # (N, 3)
    new_kpts = (M @ pts_h.T).T.astype(np.float32)  # (N, 2)
    return rotated, new_kpts


def _apply_perspective(kpts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """将点集经 3×3 透视矩阵 H 变换，返回 (N, 2)。"""
    pts = kpts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 图像增强
# ─────────────────────────────────────────────────────────────────────────────


def _color_jitter(
    img: np.ndarray,
    alpha_range: tuple = (0.82, 1.18),
    beta_range: tuple = (-20, 20),
) -> np.ndarray:
    """
    对比度 + 亮度随机扰动，各通道独立，模拟色彩偏移。
    alpha: 乘性增益；beta: 偏置（加性）。
    """
    out = img.astype(np.float32)
    for c in range(3):
        a = random.uniform(*alpha_range)
        b = random.uniform(*beta_range)
        out[:, :, c] = out[:, :, c] * a + b
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_glare(img: np.ndarray, max_strength: float = 0.18) -> np.ndarray:
    """模拟屏幕局部高光反光（高斯亮斑）。"""
    h, w = img.shape[:2]
    cx = random.randint(w // 6, w * 5 // 6)
    cy = random.randint(h // 6, h * 5 // 6)
    r = random.randint(min(w, h) // 8, min(w, h) // 3)
    strength = random.uniform(0.04, max_strength)

    Y, X = np.ogrid[:h, :w]
    sigma = max(r / 2.0, 1.0)
    mask = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
    mask = (mask * strength * 255.0).astype(np.float32)

    out = img.astype(np.float32) + mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_noise(img: np.ndarray, sigma_range: tuple = (1, 5)) -> np.ndarray:
    """高斯噪声（模拟传感器/量化噪声）。"""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _simulate_screenshot_artifacts(img: np.ndarray) -> np.ndarray:
    """
    模拟真实截图/拍照常见轻微劣化：
      1) 低强度随机噪声（始终存在，但很弱）
      2) 轻 JPEG 重编码伪影（部分样本）
      3) 极轻微锐化或软化（部分样本）
    目标：增加域随机性，同时不破坏二维码可读结构。
    """
    out = img

    # A) 常驻轻噪声：比 _add_noise 默认更弱
    out = _add_noise(out, sigma_range=(0.6, 2.0))

    # B) 轻 JPEG 压缩伪影（模拟截图二次保存/转发）
    if random.random() < 0.55:
        q = random.randint(88, 98)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec

    # C) 轻微锐化或软化
    r = random.random()
    if r < 0.25:
        # soft blur
        out = cv2.GaussianBlur(out, (3, 3), 0)
    elif r < 0.50:
        # mild sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)

    return out


def _build_screen_like_background(
    out_size: int,
    qr_top: int,
    qr_bottom: int,
) -> np.ndarray:
    """
    生成黑底背景，只在与二维码上下接触的位置放两条浅色带：
      - 整体背景：黑色
      - 上下接触条：浅灰 / 浅蓝（随机）
      - 左右区域保持黑色
    """
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # 条带颜色：浅灰或浅蓝（偏柔和）
    if random.random() < 0.55:
        v = random.randint(188, 220)  # 浅灰
        band_color = (v, v, v)
    else:
        band_color = (  # 浅蓝 BGR
            random.randint(198, 232),
            random.randint(186, 220),
            random.randint(170, 206),
        )

    # 条带厚度：约 6%~10%
    band_h = int(round(out_size * random.uniform(0.06, 0.10)))
    band_h = max(6, band_h)

    # 仅画在二维码上边和下边的“外侧接触区”
    top_y1 = max(0, qr_top - band_h)
    top_y2 = max(0, qr_top)
    bot_y1 = min(out_size, qr_bottom)
    bot_y2 = min(out_size, qr_bottom + band_h)

    if top_y2 > top_y1:
        canvas[top_y1:top_y2, :] = band_color
    if bot_y2 > bot_y1:
        canvas[bot_y1:bot_y2, :] = band_color

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 标签生成
# ─────────────────────────────────────────────────────────────────────────────


def _norm_kpts_to_grid(kpts53: np.ndarray, crop_size: int) -> np.ndarray:
    """
    将 512×512 坐标系中 53 个关键点归一化到 [-1, 1]。
    使用 align_corners=True 约定：x_norm = x / (W/2 - 0.5) - 1
    等价于 x_norm = x * 2 / (crop_size - 1) - 1。

    Args:
        kpts53   : (53, 2) 绝对像素坐标，range [0, crop_size-1]
        crop_size: 512
    Returns:
        (53, 2) float32，range 约为 [-1, 1]
    """
    half = (crop_size - 1) / 2.0  # = 255.5 for 512
    return (kpts53.astype(np.float32) / half) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 单样本生成
# ─────────────────────────────────────────────────────────────────────────────


def generate_one(
    base_img: np.ndarray,
    out_size: int = 512,
    noise_pct: float = 0.13,
    max_rot_deg: float = 28.0,
    margin_pct: float = 0.02,  # QR 四边留出的黑色边距，占 out_size 的比例
    qr_scale_min: float = 0.72,
    qr_scale_max: float = 0.95,
    safe_border_pct: float = 0.015,
) -> tuple:
    """
    对一张 base QR 图生成一个 stage2 训练样本。

    流程：
      1. 黑色边框填充（防止旋转裁剪内容）
    2. 随机旋转 ±max_rot_deg°
        3. GT 4 角点 + ±noise_pct 扰动 → 模拟 Stage1 预测误差
        4. 目标角点加入随机缩放+随机平移（不固定居中）
            getPerspectiveTransform + warpPerspective → out_size×out_size
        5. 若精确 GT 四角点投影后超出安全屏幕范围，则丢弃重采样
        6. 精确 GT 关键点（无噪声）经同一 H 变换 → out_size×out_size 坐标
        7. Color Jitter + Glare（可选）+ 噪声（可选）

    返回 (crop_img, kpts_norm) 或 (None, None)（变换异常）。
      crop_img  : (out_size, out_size, 3) uint8
      kpts_norm : (53, 2) float32，坐标归一化到 [-1,1]
    """
    # ── 1. 黑色边框，保证旋转后 QR 完整 ─────────────────────────────────────
    padded_img, pad_x, pad_y = _pad_to_fit_rotation(base_img, max_rot_deg)
    ph, pw = padded_img.shape[:2]

    # 关键点偏移至 padded 坐标系
    kpts_padded = BASE_KPTS.copy() + np.array([pad_x, pad_y], dtype=np.float32)

    # ── 2. 随机旋转 ──────────────────────────────────────────────────────────
    angle = random.uniform(-max_rot_deg, max_rot_deg)
    rotated_img, kpts_rot = _apply_rotation(padded_img, kpts_padded, angle)

    # ── 3. 注入 Stage1 角点误差（默认 ±1.5% 图像尺寸）──────────────────────
    corner_gt = kpts_rot[[IDX_TL, IDX_TR, IDX_BR, IDX_BL], :].copy()  # (4,2)
    corner_noisy = corner_gt.copy()
    nw, nh = pw * noise_pct, ph * noise_pct
    for i in range(4):
        corner_noisy[i, 0] += random.uniform(-nw, nw)
        corner_noisy[i, 1] += random.uniform(-nh, nh)

    # ── 4. 透视矫正 → out_size×out_size（随机占屏比例 + 全局平移）─────────────────
    # 不再强制二维码 95% 且居中，显式扩大域随机性，避免位置记忆。
    s_f = float(out_size - 1)
    qr_scale = random.uniform(qr_scale_min, qr_scale_max)  # 70%~95%
    side = qr_scale * s_f
    half_side = side / 2.0

    # 给真实投影留安全边界，避免 noisy corners 反求 H 后真实二维码溢出画面。
    safe_border = max(2.0, float(out_size) * safe_border_pct)

    # 保证目标四角始终落在输出画幅内；中心在可行范围内随机平移。
    max_shift = max(0.0, (s_f - side) / 2.0 - safe_border)
    shift_x = random.uniform(-max_shift, max_shift)
    shift_y = random.uniform(-max_shift, max_shift)

    cx = s_f * 0.5 + shift_x
    cy = s_f * 0.5 + shift_y
    x0, x1 = cx - half_side, cx + half_side
    y0, y1 = cy - half_side, cy + half_side
    dst_corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )
    try:
        H = cv2.getPerspectiveTransform(corner_noisy, dst_corners)
    except cv2.error:
        return None, None

    # 用真实四角点检查 warp 后是否仍处于“屏幕内”。
    gt_corners_crop = _apply_perspective(corner_gt, H)  # (4,2)
    min_xy = gt_corners_crop.min(axis=0)
    max_xy = gt_corners_crop.max(axis=0)

    # 硬约束：真实二维码四角必须完整落在安全边界内。
    # 若使用 noisy corners 拟合的 H 让真实角点溢出画面，则丢弃重采样。
    in_border = min(2.0, safe_border)
    if (
        float(min_xy[0]) < in_border
        or float(min_xy[1]) < in_border
        or float(max_xy[0]) > (s_f - in_border)
        or float(max_xy[1]) > (s_f - in_border)
    ):
        return None, None

    # 使用掩码将 QR 合成到“屏幕样式”背景上，避免默认黑底
    # src_mask 使用无噪声 corner_gt（覆盖真实二维码区域）
    src_mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.fillConvexPoly(src_mask, corner_gt.astype(np.int32), 255)

    warped_qr = cv2.warpPerspective(
        rotated_img,
        H,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpPerspective(
        src_mask,
        H,
        (out_size, out_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    qr_top = int(round(float(np.clip(min_xy[1], 0.0, s_f))))
    qr_bottom = int(round(float(np.clip(max_xy[1] + 1.0, 0.0, s_f + 1.0))))
    bg_canvas = _build_screen_like_background(
        out_size, qr_top=qr_top, qr_bottom=qr_bottom
    )
    mask_3d = (
        cv2.merge([warped_mask, warped_mask, warped_mask]).astype(np.float32) / 255.0
    )
    crop = (
        warped_qr.astype(np.float32) * mask_3d
        + bg_canvas.astype(np.float32) * (1.0 - mask_3d)
    ).astype(np.uint8)

    # ── 5. 精确 GT 关键点经 H 变换到 512×512 坐标系 ───────────────────────
    # 使用 kpts_rot（精确 GT），而非带噪声的 corner_noisy
    # 这确保标签始终是精确真值，与 Stage1 模拟误差无关
    kpts_crop = _apply_perspective(kpts_rot, H)  # (53, 2)

    # 额外保险：所有关键点应处于画幅附近，避免极端透视导致的异常标签。
    if (
        np.any(kpts_crop[:, 0] < -8.0)
        or np.any(kpts_crop[:, 1] < -8.0)
        or np.any(kpts_crop[:, 0] > s_f + 8.0)
        or np.any(kpts_crop[:, 1] > s_f + 8.0)
    ):
        return None, None

    # ── 6. 图像增强 ──────────────────────────────────────────────────────────
    crop = _color_jitter(crop)  # 必做：色彩偏移

    if random.random() < 0.6:  # 60%：屏幕反光
        crop = _add_glare(crop)

    # 真实截图/拍照噪声模拟（默认启用，强度较弱）
    crop = _simulate_screenshot_artifacts(crop)

    # ── 7. 额外离散旋转增强（0/90/180/270°）─────────────────────────────────
    # 打散“固定角点相对位置”先验，避免模型仅记住绝对布局。
    k = random.randint(0, 3)
    if k > 0:
        crop = np.ascontiguousarray(np.rot90(crop, k))
        s = float(out_size - 1)
        x = kpts_crop[:, 0].copy()
        y = kpts_crop[:, 1].copy()
        if k == 1:  # 90° CCW
            kpts_crop[:, 0] = y
            kpts_crop[:, 1] = s - x
        elif k == 2:  # 180°
            kpts_crop[:, 0] = s - x
            kpts_crop[:, 1] = s - y
        else:  # k == 3, 270° CCW
            kpts_crop[:, 0] = s - y
            kpts_crop[:, 1] = x

    # ── 8. 归一化关键点到 [-1,1] ─────────────────────────────────────────────
    kpts_norm = _norm_kpts_to_grid(kpts_crop, out_size)  # (53,2) float32
    return crop, kpts_norm


# ─────────────────────────────────────────────────────────────────────────────
# 主生成函数
# ─────────────────────────────────────────────────────────────────────────────


def generate_stage2_dataset(
    base_dir: str = "train/base_images",
    dst_dir: str = "train/stage2_dataset",
    out_size: int = 512,
    samples_per_image: int = 50,
    noise_pct: float = 0.13,
    max_rot_deg: float = 28.0,
    margin_pct: float = 0.02,
    jpeg_quality: int = 95,
):
    """
    批量生成 Stage 2 训练数据集（共 100 base × 50 = 5000 样本）。

    参数：
      base_dir          base QR 图像目录（*.png / *.jpg）
      dst_dir           输出目录（子目录 images/ & labels/ 自动创建）
      out_size          矫正裁剪尺寸（默认 512×512）
    samples_per_image 每张 base 图生成的样本数（默认 50，共 5000）
            noise_pct         Stage1 角点误差比例（±noise_pct × 图像尺寸，默认 13%）
            max_rot_deg       最大旋转角度（°），对称均匀分布（默认 ±28°）
            margin_pct        兼容旧参数，保留但不再用于限制二维码占屏比例
      jpeg_quality      输出 JPEG 质量（1-100）

    输出格式：
      images/*.jpg      512×512 失真 QR 图像
      labels/*.npy      (53,2) float32，关键点坐标归一化到 [-1,1]
    """
    dst_images = os.path.join(dst_dir, "images")
    dst_labels = os.path.join(dst_dir, "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    base_files = sorted(
        glob.glob(os.path.join(base_dir, "*.png"))
        + glob.glob(os.path.join(base_dir, "*.jpg"))
    )
    if not base_files:
        raise RuntimeError(f"[Stage2-Gen] No base images found in: {base_dir}")

    total = len(base_files) * samples_per_image

    # 构建待补齐列表：已存在(图片+标签)的样本直接跳过
    pending = []
    existing_complete = 0
    for bf in base_files:
        stem = os.path.splitext(os.path.basename(bf))[0]
        for idx in range(samples_per_image):
            name = f"{stem}_{idx:04d}"
            img_path = os.path.join(dst_images, f"{name}.jpg")
            lbl_path = os.path.join(dst_labels, f"{name}.npy")
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                existing_complete += 1
            else:
                pending.append((bf, stem, idx))

    print(
        f"[Stage2-Gen] {len(base_files)} base images × {samples_per_image} samples "
        f"= {total} total\n"
        f"             noise±{noise_pct*100:.0f}%  "
        f"rot±{max_rot_deg}°  qr_scale=70%-95%  crop={out_size}×{out_size}  "
        f"→ {os.path.abspath(dst_dir)}"
    )
    print(
        f"[Stage2-Gen] existing_complete={existing_complete}, need_fill={len(pending)}"
    )

    ok = fail = 0

    # 缓存 base 图，避免重试时反复读盘
    base_img_cache = {}
    queue = deque(pending)
    attempts = 0

    while queue:
        bf, stem, idx = queue.popleft()
        name = f"{stem}_{idx:04d}"
        img_path = os.path.join(dst_images, f"{name}.jpg")
        lbl_path = os.path.join(dst_labels, f"{name}.npy")

        # 双文件都已存在则直接跳过（支持重复运行）
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            continue

        if bf not in base_img_cache:
            base_img_cache[bf] = cv2.imread(bf)
        base_img = base_img_cache[bf]
        if base_img is None:
            raise RuntimeError(f"[Stage2-Gen] Failed to read base image: {bf}")

        local_retry = 0
        while True:
            crop, kpts_norm = generate_one(
                base_img,
                out_size=out_size,
                noise_pct=noise_pct,
                max_rot_deg=max_rot_deg,
                margin_pct=margin_pct,
            )
            attempts += 1

            if crop is not None:
                break

            fail += 1
            local_retry += 1
            if local_retry % 100 == 0:
                print(
                    f"[Stage2-Gen] retrying {name}: local_retry={local_retry}, total_retry_fail={fail}",
                    flush=True,
                )

        cv2.imwrite(
            img_path,
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        np.save(lbl_path, kpts_norm)

        ok += 1
        done_fill = ok + fail
        if done_fill % 500 == 0:
            pct = (existing_complete + ok) / total * 100.0
            print(
                f"\r[Stage2-Gen] complete={existing_complete + ok}/{total} ({pct:.1f}%)  "
                f"new_ok={ok} retry_fail={fail} pending={len(queue)}",
                end="",
                flush=True,
            )

    print(
        f"\r[Stage2-Gen] done. newly_filled={ok}, total_complete={existing_complete + ok}/{total}, "
        f"retry_fail={fail}, attempts={attempts}"
    )


def generate_stage1_source_dataset(
    base_dir: str = "train/base_images",
    bg_dir: str = "train/bg2",
    stage1_src_dir: str = "train/dataset",
    multiplier: int = 10,
    num_workers=None,
):
    """
    复用原 process_dataset.py 的能力，生成 Stage1 上游 53 点数据：
      stage1_src_dir/images/*.jpg
      stage1_src_dir/labels/*.txt
    """
    from process_dataset import process_dataset

    print(
        f"[Stage1-Source] process_dataset -> {stage1_src_dir} "
        f"(multiplier={multiplier})"
    )
    process_dataset(
        base_dir=base_dir,
        bg_dir=bg_dir,
        output_dir=stage1_src_dir,
        multiplier=multiplier,
        num_workers=num_workers,
    )


def generate_stage1_dataset_from_source(
    stage1_src_dir: str = "train/dataset",
    stage1_dst_dir: str = "train/stage1_dataset",
):
    """
    将 53 点标签提取为 Stage1 4角点标签（复用 generate_stage1_dataset.py）。
    """
    from generate_stage1_dataset import prepare_stage1_dataset

    print(f"[Stage1-Prep] prepare_stage1_dataset: {stage1_src_dir} -> {stage1_dst_dir}")
    prepare_stage1_dataset(src_dir=stage1_src_dir, dst_dir=stage1_dst_dir)


def run_pipeline(
    mode: str,
    base_dir: str,
    bg_dir: str,
    stage1_src_dir: str,
    stage1_dst_dir: str,
    stage2_dst_dir: str,
    stage1_multiplier: int,
    stage2_samples_per_image: int,
    stage2_out_size: int,
    stage2_noise_pct: float,
    stage2_max_rot_deg: float,
    stage2_margin_pct: float,
    stage2_jpeg_quality: int,
):
    """
    mode:
      - stage1_source : 仅生成 train/dataset（原 process_dataset 功能）并提取 stage1_dataset
      - stage2_only   : 仅生成 stage2_dataset
      - full          : stage1_source + stage2_only
    """
    if mode in ("stage1_source", "full"):
        generate_stage1_source_dataset(
            base_dir=base_dir,
            bg_dir=bg_dir,
            stage1_src_dir=stage1_src_dir,
            multiplier=stage1_multiplier,
        )
        generate_stage1_dataset_from_source(
            stage1_src_dir=stage1_src_dir,
            stage1_dst_dir=stage1_dst_dir,
        )

    if mode in ("stage2_only", "full"):
        generate_stage2_dataset(
            base_dir=base_dir,
            dst_dir=stage2_dst_dir,
            out_size=stage2_out_size,
            samples_per_image=stage2_samples_per_image,
            noise_pct=stage2_noise_pct,
            max_rot_deg=stage2_max_rot_deg,
            margin_pct=stage2_margin_pct,
            jpeg_quality=stage2_jpeg_quality,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="stage2_only",
        choices=["stage1_source", "stage2_only", "full"],
        help="stage1_source=原process_dataset+stage1提取, stage2_only=仅stage2, full=全部",
    )
    parser.add_argument("--base-dir", default="train/base_images")
    parser.add_argument("--bg-dir", default="train/bg2")
    parser.add_argument(
        "--stage1-src-dir",
        default="train/dataset",
        help="53-point source dir for stage1_source/full mode",
    )
    parser.add_argument("--stage1-dst-dir", default="train/stage1_dataset")
    parser.add_argument("--stage2-dst-dir", default="train/stage2_dataset")

    parser.add_argument("--stage1-multiplier", type=int, default=50)

    parser.add_argument("--stage2-out-size", type=int, default=512)
    parser.add_argument("--stage2-samples-per-image", type=int, default=100)
    parser.add_argument("--stage2-noise-pct", type=float, default=0.13)
    parser.add_argument("--stage2-max-rot-deg", type=float, default=28.0)
    parser.add_argument("--stage2-margin-pct", type=float, default=0.02)
    parser.add_argument("--stage2-jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        base_dir=args.base_dir,
        bg_dir=args.bg_dir,
        stage1_src_dir=args.stage1_src_dir,
        stage1_dst_dir=args.stage1_dst_dir,
        stage2_dst_dir=args.stage2_dst_dir,
        stage1_multiplier=args.stage1_multiplier,
        stage2_samples_per_image=args.stage2_samples_per_image,
        stage2_out_size=args.stage2_out_size,
        stage2_noise_pct=args.stage2_noise_pct,
        stage2_max_rot_deg=args.stage2_max_rot_deg,
        stage2_margin_pct=args.stage2_margin_pct,
        stage2_jpeg_quality=args.stage2_jpeg_quality,
    )
