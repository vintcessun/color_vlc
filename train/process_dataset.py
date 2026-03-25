import cv2
import numpy as np
import os
import random
import glob
import multiprocessing as _mp

import torch
import torch.nn.functional as _F

# ── GPU 检测（用 PyTorch）──────────────────────────────────────────────────
USE_CUDA = torch.cuda.is_available()
_DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
if USE_CUDA:
    print(
        f"[GPU] PyTorch CUDA 可用: {torch.cuda.get_device_name(0)}，将使用 GPU 加速。"
    )
else:
    print("[GPU] 未检测到 CUDA，使用 CPU 模式。")


# ── PyTorch GPU 加速工具函数 ────────────────────────────────────────────────
def _np_to_tensor(arr):
    """HWC uint8 ndarray → (1, C, H, W) float32 CUDA tensor [0,1]"""
    t = torch.from_numpy(arr).to(_DEVICE, dtype=torch.float32)  # H W C
    return t.permute(2, 0, 1).unsqueeze(0) / 255.0  # 1 C H W


def _tensor_to_np(t):
    """(1, C, H, W) float32 tensor → HWC uint8 ndarray"""
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()


def _build_warp_grid(M, src_hw, dst_hw):
    """
    根据 3×3 透视矩阵 M（src->dst）和目标尺寸生成 grid_sample 用的采样网格。
    返回形状 (1, dst_h, dst_w, 2) 的标准化坐标 [-1,1]。
    """
    dst_h, dst_w = dst_hw
    src_h, src_w = src_hw
    # 构造目标像素坐标网格 (dst空间)
    ys = torch.arange(dst_h, device=_DEVICE, dtype=torch.float32)
    xs = torch.arange(dst_w, device=_DEVICE, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (dst_h, dst_w)
    ones = torch.ones_like(grid_x)
    coords = torch.stack([grid_x, grid_y, ones], dim=0)  # (3, dst_h, dst_w)
    coords = coords.view(3, -1)  # (3, N)

    # M_inv: dst → src
    M_inv = torch.from_numpy(np.linalg.inv(M)).to(_DEVICE, dtype=torch.float32)
    src_coords = M_inv @ coords  # (3, N)
    src_x = src_coords[0] / src_coords[2]
    src_y = src_coords[1] / src_coords[2]

    # 归一化到 [-1, 1]
    src_x_norm = (src_x / (src_w - 1)) * 2 - 1
    src_y_norm = (src_y / (src_h - 1)) * 2 - 1
    grid = torch.stack([src_x_norm, src_y_norm], dim=-1)  # (N, 2)
    return grid.view(1, dst_h, dst_w, 2)


def _gpu_warp_perspective(img_np, M, dst_wh, mode="bilinear"):
    """用 PyTorch grid_sample 实现 warpPerspective。img_np: HWC uint8"""
    dst_w, dst_h = dst_wh
    src_h, src_w = img_np.shape[:2]
    t = _np_to_tensor(img_np)  # 1CHW
    grid = _build_warp_grid(M, (src_h, src_w), (dst_h, dst_w))  # 1HW2
    align = True if mode == "bilinear" else False
    out = _F.grid_sample(t, grid, mode=mode, padding_mode="zeros", align_corners=True)
    return _tensor_to_np(out)


def _gpu_gaussian_blur(img_np, ksize):
    """用 PyTorch separable conv 实现 GaussianBlur。"""
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # OpenCV 默认 sigma 公式
    t = _np_to_tensor(img_np)  # 1CHW
    # 1D 高斯核
    half = ksize // 2
    xs = torch.arange(-half, half + 1, device=_DEVICE, dtype=torch.float32)
    kernel_1d = torch.exp(-(xs**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    # 外积得 2D 核
    kernel_2d = kernel_1d.outer(kernel_1d)  # (ksize, ksize)
    k = kernel_2d.view(1, 1, ksize, ksize).expand(3, 1, ksize, ksize)
    pad = ksize // 2
    out = _F.conv2d(t, k, padding=pad, groups=3)
    return _tensor_to_np(out)


def _gpu_resize(img_np, dst_wh):
    """用 PyTorch interpolate 实现 resize。"""
    dst_w, dst_h = dst_wh
    t = _np_to_tensor(img_np)
    out = _F.interpolate(t, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    return _tensor_to_np(out)


# ─────────────────────────────────────────────────────────────────────────────
# Version-40 QR 关键点布局（固定，与 gen_base.rs 保持一致）
#   module_count = 177, box_size = 4, border = 1, img_size = 716
#
# 关键点顺序（共 53 个）：
#   [0]   Red   Finder 中心   (top-left)
#   [1]   Green Finder 中心   (top-right)
#   [2]   Blue  Finder 中心   (bottom-left)
#   [3..48] Alignment Pattern 中心  (46 个，从左到右、从上到下排列)
#   [49]  TL 角点  (0,       0      )
#   [50]  TR 角点  (img_w-1, 0      )
#   [51]  BR 角点  (img_w-1, img_h-1)
#   [52]  BL 角点  (0,       img_h-1)
# ─────────────────────────────────────────────────────────────────────────────
QR_MODULE_COUNT = 177
QR_BOX_SIZE = 4
QR_BORDER = 1
QR_IMG_SIZE = (QR_MODULE_COUNT + 2 * QR_BORDER) * QR_BOX_SIZE  # 716

# Version-40 Alignment pattern 位置坐标列表
QR_ALIGN_POS = [6, 30, 58, 86, 114, 142, 170]

# 与 Finder Pattern 重叠需跳过的格子（行, 列）
_FINDER_OVERLAP = {(6, 6), (6, 170), (170, 6)}

NUM_KPT = 53  # 3 finders + 46 alignments + 4 corners


def _module_center_px(row, col):
    """返回模块 (row, col) 在 base 图中的像素中心 (x, y)"""
    x = (col + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    y = (row + QR_BORDER) * QR_BOX_SIZE + QR_BOX_SIZE // 2
    return float(x), float(y)


def get_base_keypoints():
    """
    返回 716×716 base 图中所有 53 个关键点的 (x, y) 坐标列表。
    顺序：[Red Finder, Green Finder, Blue Finder,
           46× Alignment (row-major), TL, TR, BR, BL]
    """
    kpts = []

    # 1. Finder Pattern 中心（3 个）
    # Red:   setup_position_probe_pattern(0,               0,               Red)
    #         → 7×7 块中心在 module (3, 3)
    kpts.append(_module_center_px(3, 3))
    # Green: setup_position_probe_pattern(0,               module_count-7,  Green)
    #         → 中心在 module (3, module_count-4)
    kpts.append(_module_center_px(3, QR_MODULE_COUNT - 4))
    # Blue:  setup_position_probe_pattern(module_count-7,  0,               Blue)
    #         → 中心在 module (module_count-4, 3)
    kpts.append(_module_center_px(QR_MODULE_COUNT - 4, 3))

    # 2. Alignment Pattern 中心（46 个，行优先排列）
    mc = QR_MODULE_COUNT
    for i in QR_ALIGN_POS:
        for j in QR_ALIGN_POS:
            if (i, j) in _FINDER_OVERLAP:
                continue
            # 与 Finder 的其余重叠保护（代码镜像 qr_code.rs 逻辑）
            if i <= 8 and j <= 8:
                continue
            if i <= 8 and j >= mc - 8:
                continue
            if i >= mc - 8 and j <= 8:
                continue
            kpts.append(_module_center_px(i, j))

    assert len(kpts) == 49, f"Expected 49 kpts so far, got {len(kpts)}"

    # 3. 四个角点（4 个）：白色外边框四角（整张 base 图边界）
    #   你当前实拍定义是“带白边框像素”的外角，因此这里用 [0, w-1]×[0, h-1]
    outer_start = 0.0
    outer_end = float(QR_IMG_SIZE - 1)  # 715.0
    kpts.append((outer_start, outer_start))  # TL
    kpts.append((outer_end, outer_start))  # TR
    kpts.append((outer_end, outer_end))  # BR
    kpts.append((outer_start, outer_end))  # BL

    assert len(kpts) == NUM_KPT, f"Expected {NUM_KPT} kpts, got {len(kpts)}"
    return kpts  # list of (x, y) in base image pixel coords


def _perspective_point(M, x, y):
    """用透视矩阵 M 变换单个点 (x, y)，返回变换后的 (x', y')"""
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = M @ p
    return q[0] / q[2], q[1] / q[2]


# ─────────────────────────────────────────────────────────────────────────────


def augment_and_label(img, bg_files=None, target_w=3840, target_h=2160):
    """
    进行模拟手机拍摄场景的增强：
    - 输入模拟为 4K 横向分辨率 (3840x2160)
    - 确保二维码完整在图像范围内
    - 场景约束为：大占比、小旋转、轻微光照波动
    - 严禁镜像翻转（保持彩色方向一致）
    """
    h_src, w_src = img.shape[:2]

    # 原始二维码在 base 图中的坐标 (包含 1 单元 border)
    # base 图像大小为 716x716
    img_h, img_w = img.shape[:2]

    # 将四个角点定义为 base 图的四个顶点（用于透视变换矩阵的求解）
    pts = np.array(
        [
            [0, 0],
            [img_w, 0],
            [img_w, img_h],
            [0, img_h],
        ],
        dtype=np.float32,
    )

    # 预先获取全部 53 个 base 关键点（固定坐标）
    base_kpts = get_base_keypoints()

    q_size = img_w

    # 1. 使用 bg2 真实背景
    if not bg_files:
        return None, None

    bg_path = random.choice(bg_files)
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        return None, None

    bg_h, bg_w = bg_img.shape[:2]
    scale_w = target_w / bg_w
    scale_h = target_h / bg_h
    scale_bg = max(scale_w, scale_h)
    new_w = int(bg_w * scale_bg)
    new_h = int(bg_h * scale_bg)
    bg_img_resized = cv2.resize(bg_img, (new_w, new_h))

    start_x = random.randint(0, new_w - target_w)
    start_y = random.randint(0, new_h - target_h)
    canvas = bg_img_resized[start_y : start_y + target_h, start_x : start_x + target_w]
    bg_color = [int(x) for x in cv2.mean(canvas)[:3]]

    # 2. 检测背景中的黑色屏幕区域（符合当前实际）
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray_canvas, 95, 255, cv2.THRESH_BINARY_INV)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, morph_kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, morph_kernel)
    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    sx, sy, sw, sh = cv2.boundingRect(largest_contour)
    if sw * sh < target_w * target_h * 0.10:
        return None, None

    # 对黑屏区域轻微内缩，避免贴边
    inset_x = int(sw * 0.008)
    inset_y = int(sh * 0.008)
    screen_x1 = sx + inset_x
    screen_y1 = sy + inset_y
    screen_x2 = sx + sw - inset_x
    screen_y2 = sy + sh - inset_y
    if screen_x2 <= screen_x1 or screen_y2 <= screen_y1:
        return None, None

    # 3. 随机缩放和旋转（屏幕中央 + 稍小二维码 + 更小角度）
    safe_h = screen_y2 - screen_y1
    safe_w = screen_x2 - screen_x1
    min_qr_h_on_screen = safe_h * 0.82
    max_qr_h_on_screen = safe_h * 0.92
    if safe_h < target_h * 0.45:
        return None, None
    target_qr_h = None

    # 尝试多次找到合法的旋转和平移，使二维码完全在黑屏区域内。
    # 假设使用者会手动把屏幕基本摆正：
    # - 大多数样本旋转仅 ±1.2°
    # - 少量长尾样本可接近 ±2.5°
    # - 平移围绕黑屏中心，仅保留极小“肉眼没完全对准”的误差
    for _ in range(50):
        if random.random() < 0.88:
            angle = random.uniform(-1.2, 1.2)
        else:
            angle = random.uniform(-2.5, 2.5)

        # 先按当前角度估计单位缩放下的旋转包围盒，再用最大可容纳尺度反推实际 scale
        theta = np.deg2rad(angle)
        c = abs(np.cos(theta))
        s = abs(np.sin(theta))
        box_w_unit = c * w_src + s * h_src
        box_h_unit = s * w_src + c * h_src
        if box_w_unit <= 1e-6 or box_h_unit <= 1e-6:
            continue

        max_scale_w = safe_w / box_w_unit
        max_scale_h = safe_h / box_h_unit
        max_scale = min(max_scale_w, max_scale_h)
        if max_scale <= 0:
            continue

        # 控制为“较大但不贴满”：黑屏高度约 82%~92%
        fill_ratio = random.uniform(0.84, 0.90)
        if random.random() < 0.30:
            fill_ratio = random.uniform(0.90, 0.93)
        scale = max_scale * fill_ratio

        M_rot = cv2.getRotationMatrix2D((w_src / 2, h_src / 2), angle, scale)

        # 计算旋转缩放后的矩形边界
        rect_pts_original = np.array(
            [[0, 0, 1], [w_src, 0, 1], [w_src, h_src, 1], [0, h_src, 1]],
            dtype=np.float32,
        ).T
        rotated_rect = (M_rot @ rect_pts_original).T

        min_x, min_y = np.min(rotated_rect, axis=0)
        max_x, max_y = np.max(rotated_rect, axis=0)
        curr_w = max_x - min_x
        curr_h = max_y - min_y

        # 检查旋转后的 QR 是否能放进黑屏安全区域，且高度在目标区间
        if (
            curr_w <= safe_w
            and curr_h <= safe_h
            and curr_h >= min_qr_h_on_screen
            and curr_h <= max_qr_h_on_screen
        ):
            # 约束平移范围：QR 必须在黑屏区域内
            tx_min = screen_x1 - min_x
            tx_max = screen_x2 - max_x
            ty_min = screen_y1 - min_y
            ty_max = screen_y2 - max_y
            if tx_min > tx_max or ty_min > ty_max:
                continue

            # 以黑屏区域中心为目标，仅保留极小对屏误差
            screen_cx = 0.5 * (screen_x1 + screen_x2)
            screen_cy = 0.5 * (screen_y1 + screen_y2)
            rect_cx = 0.5 * (min_x + max_x)
            rect_cy = 0.5 * (min_y + max_y)
            tx_center = screen_cx - rect_cx
            ty_center = screen_cy - rect_cy

            slack_x = max(0.0, tx_max - tx_min)
            slack_y = max(0.0, ty_max - ty_min)
            jitter_x = min(slack_x * 0.06, target_w * 0.004)
            jitter_y = min(slack_y * 0.06, target_h * 0.004)

            tx = np.clip(
                tx_center + random.uniform(-jitter_x, jitter_x), tx_min, tx_max
            )
            ty = np.clip(
                ty_center + random.uniform(-jitter_y, jitter_y), ty_min, ty_max
            )
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty
            target_qr_h = q_size * scale
            break
    else:
        # 如果重试多次无法完整显示，则返回 None
        return None, None

    # 执行变换
    warped_img = cv2.warpAffine(img, M_rot, (target_w, target_h), borderValue=bg_color)

    # 转换关键点
    new_pts = []
    for pt in pts:
        p = np.array([pt[0], pt[1], 1.0])
        new_p = M_rot @ p
        new_pts.append(new_p)
    new_pts = np.array(new_pts)

    # 4. 透视变换 (模拟手机倾斜)
    # 极小透视扰动，保持几何稳定（接近正视屏幕）
    p_offset = target_qr_h * 0.0035

    # 获取旋转后的四个角点
    final_pts = new_pts + np.random.uniform(-p_offset, p_offset, (4, 2)).astype(
        np.float32
    )

    # 确保 final_pts 都在黑屏安全区域内
    for pt in final_pts:
        if (
            pt[0] < screen_x1
            or pt[0] > screen_x2
            or pt[1] < screen_y1
            or pt[1] > screen_y2
        ):
            return None, None

    # 计算从原图到最终点的透视变换矩阵
    M_total = cv2.getPerspectiveTransform(
        pts.astype(np.float32), final_pts.astype(np.float32)
    )

    # 执行总的透视变换
    # 先在画布上覆盖背景
    warped_img = canvas.copy()
    # 将二维码透视变换后贴在背景上
    # 创建掩码
    qr_mask = np.zeros((h_src, w_src), dtype=np.uint8)
    cv2.fillConvexPoly(qr_mask, pts.astype(np.int32), 255)

    if USE_CUDA:
        M_f64 = M_total.astype(np.float64)
        warped_qr = _gpu_warp_perspective(
            img, M_f64, (target_w, target_h), mode="bilinear"
        )
        warped_mask_f = _gpu_warp_perspective(
            np.stack([qr_mask, qr_mask, qr_mask], axis=2),  # 探为 3通道
            M_f64,
            (target_w, target_h),
            mode="nearest",
        )[
            :, :, 0
        ]  # 取回单通道
        warped_mask = warped_mask_f
    else:
        warped_qr = cv2.warpPerspective(
            img, M_total, (target_w, target_h), flags=cv2.INTER_LINEAR
        )
        warped_mask = cv2.warpPerspective(
            qr_mask, M_total, (target_w, target_h), flags=cv2.INTER_NEAREST
        )

    # 混合
    mask_3d = cv2.merge([warped_mask, warped_mask, warped_mask]) / 255.0
    warped_img = (warped_qr * mask_3d + warped_img * (1 - mask_3d)).astype(np.uint8)

    # 5. 模拟真实摄像头成像退化
    # A. 轻失焦（较常见）
    if random.random() < 0.45:
        k = random.choice([3, 5])
        if USE_CUDA:
            warped_img = _gpu_gaussian_blur(warped_img, k)
        else:
            warped_img = cv2.GaussianBlur(warped_img, (k, k), 0)

    # B. 轻运动模糊（手抖/快门）
    if random.random() < 0.28:
        k = random.choice([3, 5, 7, 9])
        kernel = np.zeros((k, k), dtype=np.float32)
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        kernel /= kernel.sum()
        warped_img = cv2.filter2D(warped_img, -1, kernel)

    # C. 屏幕反光（Glare）：轻微高亮斑
    glare_cx = int(random.uniform(target_w * 0.25, target_w * 0.75))
    glare_cy = int(random.uniform(target_h * 0.05, target_h * 0.35))
    glare_strength = random.uniform(0.04, 0.14)
    glare_radius = random.uniform(target_w * 0.35, target_w * 0.55)

    y, x = np.ogrid[:target_h, :target_w]
    dist2 = (x - glare_cx) ** 2 + (y - glare_cy) ** 2
    glare = np.exp(-dist2 / (2 * glare_radius**2)).astype(np.float32)
    warped_img = np.clip(
        warped_img.astype(np.float32) + glare[:, :, None] * (255.0 * glare_strength),
        0,
        255,
    ).astype(np.uint8)

    # D. 轻微亮度/对比度抖动
    alpha = random.uniform(0.94, 1.06)
    beta = random.uniform(-10.0, 10.0)
    warped_img = np.clip(warped_img.astype(np.float32) * alpha + beta, 0, 255).astype(
        np.uint8
    )

    # E. 传感器噪声（始终有弱噪，部分样本更强）
    noise_sigma = random.uniform(0.6, 2.2)
    if random.random() < 0.35:
        noise_sigma = random.uniform(2.2, 4.2)
    noise = np.random.normal(0, noise_sigma, warped_img.shape).astype(np.int16)
    warped_img = np.clip(warped_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # F. JPEG 重编码伪影（截图保存/转发）
    if random.random() < 0.55:
        q = random.randint(86, 98)
        ok, enc = cv2.imencode(".jpg", warped_img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                warped_img = dec

    # 6. 生成 YOLO 标注 (保持 16:9 比例)
    # 注意：YOLO 训练通常输入 640，但我们生成的是 4K 效果。
    # 我们将图像等比例缩放，减少磁盘占用，同时保持 4K 细节经过下采样后的观感
    final_w = 1280
    final_h = 720
    if USE_CUDA:
        warped_img_small = _gpu_resize(warped_img, (final_w, final_h))
    else:
        warped_img_small = cv2.resize(warped_img, (final_w, final_h))

    # ── 变换全部 53 个关键点 ────────────────────────────────────────────────
    M_total_np = M_total.astype(np.float64)
    transformed_kpts = []
    for bx, by in base_kpts:
        tx, ty = _perspective_point(M_total_np, bx, by)
        transformed_kpts.append((tx, ty))

    # ── 计算边界框（基于 4 个角点，即最后 4 个关键点）────────────────────────
    corner_pts = np.array(transformed_kpts[-4:], dtype=np.float32)
    x_min, y_min = np.min(corner_pts, axis=0)
    x_max, y_max = np.max(corner_pts, axis=0)
    bw = x_max - x_min
    bh = y_max - y_min
    cx = x_min + bw / 2
    cy = y_min + bh / 2

    # ── 归一化 (基于原始 target_w/target_h) ─────────────────────────────────
    label_str = (
        f"0 {cx/target_w:.6f} {cy/target_h:.6f} {bw/target_w:.6f} {bh/target_h:.6f}"
    )
    for tx, ty in transformed_kpts:
        # 判断关键点是否在图像内，超出范围标记为不可见 (v=0)
        if 0.0 <= tx <= target_w and 0.0 <= ty <= target_h:
            vis = 2
        else:
            vis = 0
            tx, ty = 0.0, 0.0  # YOLO 约定：不可见点坐标置 0
        label_str += f" {tx/target_w:.6f} {ty/target_h:.6f} {vis}"

    return warped_img_small, label_str


# ── 多进程工作函数（必须为顶层函数以供 pickle）──────────────────────────────
def _worker(args):
    """生成单个样本，返回 (idx, warped_bytes, label_str) 或 None（失败时）。"""
    idx, base_files, bg_files = args
    for _ in range(20):  # 最多重试 20 次
        base_file = random.choice(base_files)
        img = cv2.imread(base_file)
        if img is None:
            continue
        warped, label = augment_and_label(img, bg_files=bg_files)
        if warped is not None:
            # 编码为 JPEG bytes，避免在共享内存中传递大数组
            ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return idx, buf.tobytes(), label
    return None


def process_dataset(
    base_dir="train/base_images",
    bg_dir="train/bg2",
    output_dir="train/dataset",
    multiplier=10,
    num_workers=None,
):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    base_files = glob.glob(os.path.join(base_dir, "*.png"))
    bg_files = glob.glob(os.path.join(bg_dir, "*.png")) + glob.glob(
        os.path.join(bg_dir, "*.jpg")
    )
    total_needed = len(base_files) * multiplier
    print(
        f"Found {len(base_files)} base images, {len(bg_files)} background images."
        f" Target samples: {total_needed}"
    )

    # GPU 模式：每个 worker 都能调用 CUDA，进程数取决于 GPU 吞吐；
    # CPU 模式：用逻辑核心数并行。
    if num_workers is None:
        num_workers = 1 if USE_CUDA else max(1, _mp.cpu_count() - 1)
    print(f"[并行] workers={num_workers}  GPU={USE_CUDA}")

    tasks = [(i, base_files, bg_files) for i in range(total_needed)]

    count = 0
    if num_workers == 1:
        # 单进程（GPU 模式推荐，避免多进程争抢 CUDA context）
        for args in tasks:
            result = _worker(args)
            if result is None:
                continue
            idx, buf, label = result
            with open(os.path.join(images_dir, f"sample_{idx:05d}.jpg"), "wb") as f:
                f.write(buf)
            with open(os.path.join(labels_dir, f"sample_{idx:05d}.txt"), "w") as f:
                f.write(label + "\n")
            count += 1
            if count % 50 == 0:
                print(f"Generated {count}/{total_needed} samples...")
    else:
        # 多进程（CPU 模式）
        with _mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_worker, tasks, chunksize=4):
                if result is None:
                    continue
                idx, buf, label = result
                with open(os.path.join(images_dir, f"sample_{idx:05d}.jpg"), "wb") as f:
                    f.write(buf)
                with open(os.path.join(labels_dir, f"sample_{idx:05d}.txt"), "w") as f:
                    f.write(label + "\n")
                count += 1
                if count % 50 == 0:
                    print(f"Generated {count}/{total_needed} samples...")

    print(f"Done. Generated {count} samples.")


if __name__ == "__main__":
    process_dataset()
