import cv2
import numpy as np
import os
import random
import glob


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

NUM_KPT = 53   # 3 finders + 46 alignments + 4 corners


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

    # 3. 四个角点（4 个）：base 图像素角点
    s = float(QR_IMG_SIZE - 1)  # 715.0
    kpts.append((0.0,   0.0))   # TL
    kpts.append((s,     0.0))   # TR
    kpts.append((s,     s  ))   # BR
    kpts.append((0.0,   s  ))   # BL

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
    - 包含旋转、歪曲、缩放、模糊、光照不均、噪声等
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

    # 1. 创建 4K 画布并填充随机背景或噪声纹理
    if bg_files:
        bg_path = random.choice(bg_files)
        bg_img = cv2.imread(bg_path)
        if bg_img is not None:
            # 随机裁剪/缩放到 target_w, target_h
            bg_h, bg_w = bg_img.shape[:2]
            # 先缩放，保证能覆盖画布
            scale_w = target_w / bg_w
            scale_h = target_h / bg_h
            scale = max(scale_w, scale_h)

            new_w = int(bg_w * scale)
            new_h = int(bg_h * scale)
            bg_img_resized = cv2.resize(bg_img, (new_w, new_h))

            # 随机裁剪
            start_x = random.randint(0, new_w - target_w)
            start_y = random.randint(0, new_h - target_h)
            canvas = bg_img_resized[
                start_y : start_y + target_h, start_x : start_x + target_w
            ]
            bg_color = [int(x) for x in cv2.mean(canvas)[:3]]
        else:
            bg_color = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200),
            )
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * np.array(
                bg_color, dtype=np.uint8
            )
    else:
        bg_color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200),
        )
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * np.array(
            bg_color, dtype=np.uint8
        )
        # 添加一些随机纹理噪声模拟背景
        noise_bg = np.random.randint(0, 30, (target_h, target_w, 3), dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise_bg, 0, 255).astype(np.uint8)

    # 2. 检测背景中的白色/浅色安全区域
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray_canvas, 150, 255, cv2.THRESH_BINARY)
    # 形态学闭运算，填补小间隙
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, morph_kernel)
    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    # 取最大亮色区域的边界矩形作为安全区域
    largest_contour = max(contours, key=cv2.contourArea)
    wx, wy, ww, wh = cv2.boundingRect(largest_contour)
    # 安全区域需要足够大 (至少占画布 30%)
    if ww * wh < target_w * target_h * 0.3:
        return None, None
    white_x1, white_y1 = wx, wy
    white_x2, white_y2 = wx + ww, wy + wh

    # 3. 随机缩放和旋转
    # 二维码大小为白色区域高度的 85%~98%
    safe_h = white_y2 - white_y1
    safe_w = white_x2 - white_x1
    target_qr_h = random.uniform(safe_h * 0.85, safe_h * 0.98)
    scale = target_qr_h / q_size

    # 尝试多次找到合法的旋转和平移，使二维码完全在白色区域内
    for _ in range(50):
        angle = random.uniform(-180, 180)
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

        # 检查旋转后的 QR 是否能放进白色安全区域
        if curr_w <= safe_w and curr_h <= safe_h:
            # 约束平移范围：QR 必须在白色区域内
            tx_min = white_x1 - min_x
            tx_max = white_x2 - max_x
            ty_min = white_y1 - min_y
            ty_max = white_y2 - max_y
            if tx_min > tx_max or ty_min > ty_max:
                continue
            tx = random.uniform(tx_min, tx_max)
            ty = random.uniform(ty_min, ty_max)
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty
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

    # 3. 透视变换 (模拟手机倾斜)
    # 改进透视逻辑：直接对 new_pts 角点进行微小扰动
    p_offset = target_qr_h * 0.05  # 减小透视扰动，确保不超出边界

    # 获取旋转后的四个角点
    final_pts = new_pts + np.random.uniform(-p_offset, p_offset, (4, 2)).astype(
        np.float32
    )

    # 确保 final_pts 都在白色安全区域内
    for pt in final_pts:
        if pt[0] < white_x1 or pt[0] > white_x2 or pt[1] < white_y1 or pt[1] > white_y2:
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

    warped_qr = cv2.warpPerspective(
        img, M_total, (target_w, target_h), flags=cv2.INTER_LINEAR
    )
    warped_mask = cv2.warpPerspective(
        qr_mask, M_total, (target_w, target_h), flags=cv2.INTER_NEAREST
    )

    # 混合
    mask_3d = cv2.merge([warped_mask, warped_mask, warped_mask]) / 255.0
    warped_img = (warped_qr * mask_3d + warped_img * (1 - mask_3d)).astype(np.uint8)

    # 4. 模拟手机拍摄退化
    # A. 模糊 (Defocus + Motion)
    if random.random() > 0.3:
        k = random.choice([3, 5, 7, 9, 11])
        warped_img = cv2.GaussianBlur(warped_img, (k, k), 0)

    # B. 亮度对比度与光照不均 (Vignetting & Spot Light)
    # 模拟渐变光照
    mask = np.ones((target_h, target_w), dtype=np.float32)
    center_light = (random.randint(0, target_w), random.randint(0, target_h))
    strength = random.uniform(0.3, 0.8)
    radius = random.uniform(target_w * 0.5, target_w * 1.5)

    y, x = np.ogrid[:target_h, :target_w]
    dist_from_center = np.sqrt((x - center_light[0]) ** 2 + (y - center_light[1]) ** 2)
    mask = np.exp(-(dist_from_center**2) / (2 * radius**2))
    mask = strength * mask + (1 - strength)

    warped_img = (warped_img.astype(np.float32) * mask[:, :, np.newaxis]).astype(
        np.uint8
    )

    # C. 噪声
    if random.random() > 0.3:
        noise = np.random.normal(0, random.uniform(2, 12), warped_img.shape).astype(
            np.int16
        )
        warped_img = np.clip(warped_img.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )

    # 5. 生成 YOLO 标注 (保持 16:9 比例)
    # 注意：YOLO 训练通常输入 640，但我们生成的是 4K 效果。
    # 我们将图像等比例缩放，减少磁盘占用，同时保持 4K 细节经过下采样后的观感
    final_w = 1280
    final_h = 720
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


def process_dataset(
    base_dir="train/base_images",
    bg_dir="train/bg2",
    output_dir="train/dataset",
    multiplier=100,
):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    base_files = glob.glob(os.path.join(base_dir, "*.png"))
    bg_files = glob.glob(os.path.join(bg_dir, "*.png")) + glob.glob(
        os.path.join(bg_dir, "*.jpg")
    )

    print(
        f"Found {len(base_files)} base images, {len(bg_files)} background images. Target samples: {len(base_files) * multiplier}"
    )

    count = 0
    total_needed = len(base_files) * multiplier

    while count < total_needed:
        base_file = random.choice(base_files)
        img = cv2.imread(base_file)
        if img is None:
            continue

        warped, label = augment_and_label(img, bg_files=bg_files)
        if warped is None:
            continue

        cv2.imwrite(
            os.path.join(images_dir, f"sample_{count:05d}.jpg"),
            warped,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        with open(os.path.join(labels_dir, f"sample_{count:05d}.txt"), "w") as f:
            f.write(label + "\n")

        count += 1
        if count % 50 == 0:
            print(f"Generated {count}/{total_needed} samples...")


if __name__ == "__main__":
    process_dataset()
