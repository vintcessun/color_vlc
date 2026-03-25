import argparse
import os
import glob
import shutil
from collections import deque
import random

import cv2
import numpy as np


CORNER_INNER_MARGIN = 0.015


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _is_valid_pose_label_file(path: str) -> bool:
    """检测标签是否为 YOLO-Pose 格式：class + 4 个 box 浮点 + 12 个关键点浮点（共 17 列）。"""
    if not os.path.exists(path):
        return False
    try:
        line = open(path, "r", encoding="utf-8").read().strip()
    except OSError:
        return False
    if not line:
        return False
    parts = line.split()
    if len(parts) != 17:
        return False
    try:
        _ = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return False
    return True


def _bbox_to_corners(cx: float, cy: float, bw: float, bh: float):
    x_min = _clip01(cx - bw * 0.5)
    y_min = _clip01(cy - bh * 0.5)
    x_max = _clip01(cx + bw * 0.5)
    y_max = _clip01(cy + bh * 0.5)
    return [
        (x_min, y_min),  # TL
        (x_max, y_min),  # TR
        (x_max, y_max),  # BR
        (x_min, y_max),  # BL
    ]


def _extract_4corners_from_53(label_line: str):
    """从 53 点标签中提取四角点（归一化坐标，顺序 TL/TR/BR/BL）。"""
    arr = label_line.strip().split()
    if len(arr) != 164:
        raise ValueError(f"invalid label length {len(arr)}, expected 164")

    cls_id = arr[0]
    cx, cy, bw, bh = map(float, arr[1:5])

    kpts = arr[5:]
    corners = []
    for idx in [49, 50, 51, 52]:
        base = idx * 3
        x = float(kpts[base + 0])
        y = float(kpts[base + 1])
        v = int(float(kpts[base + 2]))
        if v > 0:
            corners.append((_clip01(x), _clip01(y)))

    if len(corners) != 4:
        corners = _bbox_to_corners(cx, cy, bw, bh)

    return cls_id, corners


def _format_pose_label_from_corners(
    cls_id: str, corners_norm, noisy_bbox: bool = False
):
    """输出 YOLO-Pose: class + bbox(cx,cy,w,h) + 12个角点参数(x,y,v)"""
    pts = np.array(corners_norm, dtype=np.float32)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    x_min_exact, y_min_exact = float(x_min), float(y_min)
    x_max_exact, y_max_exact = float(x_max), float(y_max)

    bw_exact = float(x_max - x_min)
    bh_exact = float(y_max - y_min)

    if noisy_bbox and random.random() < 0.8:
        # 仅允许“内缩+轻平移”，不允许外扩到二维码之外。
        shrink_x = random.uniform(0.00, 0.06)
        shrink_y = random.uniform(0.00, 0.06)
        shift_x = random.uniform(-0.02, 0.02) * bw_exact
        shift_y = random.uniform(-0.02, 0.02) * bh_exact

        nx_min = float(x_min) + bw_exact * shrink_x + shift_x
        nx_max = float(x_max) - bw_exact * shrink_x + shift_x
        ny_min = float(y_min) + bh_exact * shrink_y + shift_y
        ny_max = float(y_max) - bh_exact * shrink_y + shift_y

        # 防止收缩过度导致反转
        if nx_max - nx_min >= 0.5 * bw_exact and ny_max - ny_min >= 0.5 * bh_exact:
            x_min, x_max, y_min, y_max = nx_min, nx_max, ny_min, ny_max

        # 硬钳制：绝不允许超出真实二维码包围框。
        x_min = max(float(x_min), x_min_exact)
        x_max = min(float(x_max), x_max_exact)
        y_min = max(float(y_min), y_min_exact)
        y_max = min(float(y_max), y_max_exact)

    x_min = _clip01(float(x_min))
    y_min = _clip01(float(y_min))
    x_max = _clip01(float(x_max))
    y_max = _clip01(float(y_max))

    bw = max(1e-6, x_max - x_min)
    bh = max(1e-6, y_max - y_min)
    cx = _clip01(x_min + bw * 0.5)
    cy = _clip01(y_min + bh * 0.5)

    base_label = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    kpt_parts = []
    for x, y in pts:
        px = _clip01(float(x))
        py = _clip01(float(y))
        # v=2表示可见并且已经标注
        kpt_parts.append(f"{px:.6f} {py:.6f} 2")

    return base_label + " " + " ".join(kpt_parts)


def _extract_stage1_pose_label_from_53(label_line: str) -> str:
    """
    原始格式: class + box(4) + 53*(x y v)
    目标格式: class + box(4) + 4*(x y v)
    Stage1 适配 YOLO-Pose。
    """
    cls_id, corners = _extract_4corners_from_53(label_line)
    return _format_pose_label_from_corners(cls_id, corners)


def _corners_all_inside_screen(
    corners_norm, margin: float = CORNER_INNER_MARGIN
) -> bool:
    """四角必须严格在屏幕内，不允许贴边或越界。"""
    pts = np.asarray(corners_norm, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return False
    return bool(
        np.all(pts[:, 0] >= margin)
        and np.all(pts[:, 0] <= 1.0 - margin)
        and np.all(pts[:, 1] >= margin)
        and np.all(pts[:, 1] <= 1.0 - margin)
    )


def _pull_corners_inside(corners_norm, margin: float = CORNER_INNER_MARGIN):
    """将四角点拉回屏幕内部边距内。"""
    pts = np.asarray(corners_norm, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], margin, 1.0 - margin)
    pts[:, 1] = np.clip(pts[:, 1], margin, 1.0 - margin)
    return pts.tolist()


def _is_reasonable_corners(corners_norm) -> bool:
    """
    过滤掉几何扭曲过强的样本，避免二维码被扭到不可学习。
    """
    pts = np.asarray(corners_norm, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return False

    # 要求四角都严格在屏幕内（不贴边）。
    if not _corners_all_inside_screen(pts):
        return False

    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    bw = float(x_max - x_min)
    bh = float(y_max - y_min)
    if bw < 0.20 or bh < 0.20:
        return False
    if bw > 0.90 or bh > 0.90:
        return False

    ratio = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
    if ratio > 1.80:
        return False

    # 多边形面积占比过小或过大都视为异常。
    area_ratio = abs(float(cv2.contourArea(pts.astype(np.float32))))
    if area_ratio < 0.06 or area_ratio > 0.93:
        return False

    return True


def get_random_quad(img_size, margin=50):
    """
    随机生成 4 个目标顶点，确保在背景图内部且不退化。
    """
    w, h = img_size
    # 先生成一个基础矩形，然后进行强力随机扰动
    # 动态缩放：20% 到 90%
    scale = random.uniform(0.20, 0.90)
    target_w = w * scale
    target_h = h * scale

    # 随机中心点，确保不超出边界
    cx = random.uniform(target_w / 2 + margin, w - target_w / 2 - margin)
    cy = random.uniform(target_h / 2 + margin, h - target_h / 2 - margin)

    # 基础四个角点
    x1, y1 = cx - target_w / 2, cy - target_h / 2
    x2, y2 = cx + target_w / 2, cy - target_h / 2
    x3, y3 = cx + target_w / 2, cy + target_h / 2
    x4, y4 = cx - target_w / 2, cy + target_h / 2

    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    # 全向旋转：0-360
    angle = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    pts_homo = np.hstack([pts, np.ones((4, 1))])
    pts_rotated = (M @ pts_homo.T).T

    # 强力透视扰动：对每个角点进行独立随机偏移
    distort_range = min(target_w, target_h) * 0.35
    for i in range(4):
        pts_rotated[i, 0] += random.uniform(-distort_range, distort_range)
        pts_rotated[i, 1] += random.uniform(-distort_range, distort_range)

    # 边界检查
    pts_rotated[:, 0] = np.clip(pts_rotated[:, 0], 0, w - 1)
    pts_rotated[:, 1] = np.clip(pts_rotated[:, 1], 0, h - 1)

    # 显式转换为 float32，避免 OpenCV 函数报错
    return pts_rotated.astype(np.float32)


def apply_extreme_augmentations(img: np.ndarray, pts: list):
    """
    升级版：支持全向旋转、强透视变换和动态缩放。
    """
    h, w = img.shape[:2]
    # 原图角点（通常是 0,0 到 w,h）
    src_pts = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )

    # 随机生成目标四边形
    dst_pts = get_random_quad((w, h))

    # 面积检查
    area = cv2.contourArea(dst_pts.astype(np.float32))
    if area < (w * h * 0.04):  # 丢弃面积太小的样本
        return None, None

    # 透视变换
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    out_img = cv2.warpPerspective(
        img.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 更新关键点坐标（归一化）
    out_pts = dst_pts / np.array([w, h], dtype=np.float32)

    # 2. 屏幕翻拍真实感模拟
    # 2.1 环境反光 (Screen Glare)
    if random.random() < 0.8:
        num_glares = random.randint(1, 3)
        glare_mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(num_glares):
            cx, cy = random.uniform(0, w), random.uniform(0, h)
            rx, ry = random.uniform(0.1 * w, 0.4 * w), random.uniform(0.1 * h, 0.4 * h)
            pts_ellipse = cv2.ellipse2Poly(
                (int(cx), int(cy)),
                (int(rx), int(ry)),
                angle=random.randint(0, 360),
                arcStart=0,
                arcEnd=360,
                delta=10,
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts_ellipse, 255)
            alpha = random.uniform(0.1, 0.3)
            k = random.choice([35, 55, 75])
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            glare_mask += (mask.astype(np.float32) / 255.0) * alpha
        glare_mask = np.clip(glare_mask, 0.0, 1.0)
        out_img = (
            out_img * (1.0 - glare_mask[:, :, None]) + 255.0 * glare_mask[:, :, None]
        )

    # 2.2 高级屏幕摩尔纹与色彩串扰 (Advanced Moiré & Color Crosstalk)
    if random.random() < 0.85:
        # a. 色彩通道偏移 (模拟 RGB Sub-pixel 屏幕采样错位)
        if random.random() < 0.6:
            shift_m = np.float32(
                [[1, 0, random.uniform(-1.5, 1.5)], [0, 1, random.uniform(-1.5, 1.5)]]
            )
            b, g, r = cv2.split(out_img)
            b = cv2.warpAffine(b, shift_m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            r = cv2.warpAffine(r, shift_m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            out_img = cv2.merge((b, g, r))

        # b. 物理级高频摩尔纹干扰 (Moiré)
        if random.random() < 0.7:
            y_coord, x_coord = np.mgrid[0:h, 0:w].astype(np.float32)
            # 屏幕发光频率与相机采样频率的差拍
            freq_x = random.uniform(0.5, 2.0)
            freq_y = random.uniform(0.5, 2.0)
            phase_x = random.uniform(0, np.pi * 2)
            phase_y = random.uniform(0, np.pi * 2)

            # 使用低频弯曲来模拟镜头畸变下的非均匀摩尔纹
            distortion = np.sin(x_coord * 0.02 + y_coord * 0.02) * 5.0
            pattern = np.sin((x_coord + distortion) * freq_x + phase_x) * np.cos(
                (y_coord + distortion) * freq_y + phase_y
            )

            # 各通道摩尔响应有细微差异
            amp = random.uniform(15.0, 45.0)
            color_weights = np.array(
                [
                    random.uniform(0.8, 1.2),
                    random.uniform(0.8, 1.2),
                    random.uniform(0.8, 1.2),
                ],
                dtype=np.float32,
            )

            moire_noise = pattern[:, :, None] * amp * color_weights
            out_img += moire_noise

    # 2.25 轻微遮挡（默认很低概率）：避免出现明显“灰色大块”
    if random.random() < 0.08:
        occ_n = random.randint(1, 2)
        for _ in range(occ_n):
            rw = int(round(random.uniform(0.04, 0.10) * w))
            rh = int(round(random.uniform(0.04, 0.10) * h))
            x0 = random.randint(0, max(0, w - rw))
            y0 = random.randint(0, max(0, h - rh))
            color = random.randint(160, 235)
            alpha = random.uniform(0.16, 0.32)
            patch = np.full((rh, rw, 3), color, dtype=np.float32)
            out_img[y0 : y0 + rh, x0 : x0 + rw] = (
                out_img[y0 : y0 + rh, x0 : x0 + rw] * (1.0 - alpha) + patch * alpha
            )

    # 2.3 动态光影 (Local Illumination)
    if random.random() < 0.8:
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        rx, ry = random.uniform(0.3 * w, 0.8 * w), random.uniform(0.3 * h, 0.8 * h)
        y_idx, x_idx = np.ogrid[:h, :w]
        g = np.exp(
            -(((x_idx - cx) ** 2) / (2.0 * rx**2) + ((y_idx - cy) ** 2) / (2.0 * ry**2))
        )
        strength = random.uniform(-60.0, 60.0)
        out_img += g[:, :, None] * strength

    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    out_pts = np.clip(out_pts, 0.0, 1.0)

    return out_img, out_pts.tolist()


def _apply_stage1_camera_blur(img: np.ndarray) -> np.ndarray:
    """
    Stage1 额外相机模糊链：保证样本具备真实拍摄下的轻失焦感。
    """
    out = img

    # 始终执行一次轻失焦
    k = random.choice([3, 5])
    out = cv2.GaussianBlur(out, (k, k), random.uniform(0.6, 1.6))

    # 部分样本附加运动模糊
    if random.random() < 0.45:
        m = random.choice([3, 5, 7])
        kernel = np.zeros((m, m), dtype=np.float32)
        if random.random() < 0.5:
            kernel[m // 2, :] = 1.0
        else:
            kernel[:, m // 2] = 1.0
        kernel /= kernel.sum()
        out = cv2.filter2D(out, -1, kernel)

    # 轻微压缩伪影（拍照保存/转发）
    q = random.randint(82, 95)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            out = dec

    return out


def _safe_mild_geometry(img: np.ndarray, corners: list):
    """
    回退路径：保证有旋转/平移，但几何幅度温和，避免样本过于单调。
    """
    h, w = img.shape[:2]
    base_pts = np.array(corners, dtype=np.float32)
    for _ in range(12):
        angle = random.uniform(-22.0, 22.0)
        scale = random.uniform(0.94, 1.06)
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        M[0, 2] += random.uniform(-0.05 * w, 0.05 * w)
        M[1, 2] += random.uniform(-0.05 * h, 0.05 * h)

        out_img = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        pts_px = np.array([[x * w, y * h] for x, y in base_pts], dtype=np.float32)
        ones = np.ones((pts_px.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pts_px, ones])
        pts_aff = (M @ pts_h.T).T
        out_pts = np.clip(pts_aff / np.array([w, h], dtype=np.float32), 0.0, 1.0)

        if _is_reasonable_corners(out_pts.tolist()):
            return out_img, out_pts.tolist()

    return img, corners


def _augment_stage1_screen_recapture(img: np.ndarray, label53: str):
    """Stage1 屏幕翻拍增强：极致几何干扰和真实屏幕感，输出 YOLO-Pose 标签。"""
    cls_id, corners = _extract_4corners_from_53(label53)

    # 重试若干次，确保增强后几何仍可学习。
    for _ in range(32):  # 增加尝试次数
        out_img, corners_out = apply_extreme_augmentations(img, corners)
        if out_img is not None and _is_reasonable_corners(corners_out):
            out_img = _apply_stage1_camera_blur(out_img)
            det_label = _format_pose_label_from_corners(
                cls_id, corners_out, noisy_bbox=False
            )
            return out_img, det_label

    # 兜底：走一条温和几何分支，避免样本完全不旋转、过于单调。
    out_img, corners_out = _safe_mild_geometry(img, corners)
    if not _corners_all_inside_screen(corners_out):
        corners_out = _pull_corners_inside(corners_out)
    out_img = _apply_stage1_camera_blur(out_img)
    det_label = _format_pose_label_from_corners(cls_id, corners_out, noisy_bbox=False)
    return out_img, det_label


def prepare_stage1_dataset(
    src_dir="train/dataset",
    dst_dir="train/stage1_dataset",
):
    src_images = os.path.join(src_dir, "images")
    src_labels = os.path.join(src_dir, "labels")
    dst_images = os.path.join(dst_dir, "images")
    dst_labels = os.path.join(dst_dir, "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(src_images, "*.jpg")))
    if not image_files:
        raise RuntimeError(f"No images found: {src_images}")

    converted_count = 0
    copied_image_count = 0
    total = len(image_files)

    for idx, img_fp in enumerate(image_files, start=1):
        name = os.path.basename(img_fp)
        stem, _ = os.path.splitext(name)
        src_label_fp = os.path.join(src_labels, f"{stem}.txt")
        if not os.path.exists(src_label_fp):
            print(
                f"\r[Stage1] preparing dataset: {idx}/{total} "
                f"({idx / total * 100:.1f}%) copied={copied_image_count} converted={converted_count}",
                end="",
                flush=True,
            )
            continue

        dst_img_fp = os.path.join(dst_images, name)
        if not os.path.exists(dst_img_fp):
            shutil.copy2(img_fp, dst_img_fp)
            copied_image_count += 1

        with open(src_label_fp, "r", encoding="utf-8") as f:
            line = f.read().strip()
        if not line:
            print(
                f"\r[Stage1] preparing dataset: {idx}/{total} "
                f"({idx / total * 100:.1f}%) copied={copied_image_count} converted={converted_count}",
                end="",
                flush=True,
            )
            continue
        out_line = _extract_stage1_pose_label_from_53(line)
        with open(os.path.join(dst_labels, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(out_line + "\n")
        converted_count += 1

        print(
            f"\r[Stage1] preparing dataset: {idx}/{total} "
            f"({idx / total * 100:.1f}%) copied={copied_image_count} converted={converted_count}",
            end="",
            flush=True,
        )

    print()

    print(
        f"[Stage1] prepared dataset: converted={converted_count}, copied_new_images={copied_image_count} -> {dst_dir}"
    )


def build_stage1_dataset(
    dst_dir="train/stage1_dataset",
    base_dir="train/base_images",
    bg_dir="train/bg2",
    multiplier=100,
    num_workers=None,
    overwrite_existing=False,
):
    """
    一步直出 Stage1 数据集（无中间路径落盘）。
    直接复用 process_dataset 的增强逻辑，从 base_images + bg2 生成图像，
    并在内存中把 53 点标签提取成 4 角点标签写入 stage1_dataset。
    """
    import process_dataset as _pd

    dst_images = os.path.join(dst_dir, "images")
    dst_labels = os.path.join(dst_dir, "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    base_files = glob.glob(os.path.join(base_dir, "*.png")) + glob.glob(
        os.path.join(base_dir, "*.jpg")
    )
    bg_files = glob.glob(os.path.join(bg_dir, "*.png")) + glob.glob(
        os.path.join(bg_dir, "*.jpg")
    )

    if not base_files:
        raise RuntimeError(f"No base images found: {base_dir}")
    if not bg_files:
        raise RuntimeError(f"No background images found: {bg_dir}")

    total_needed = len(base_files) * multiplier
    print(
        f"[Stage1] direct build from base+bg: base={len(base_files)} bg={len(bg_files)} "
        f"target={total_needed} -> {dst_dir}"
    )

    if num_workers is None:
        num_workers = 1 if _pd.USE_CUDA else max(1, _pd._mp.cpu_count() - 1)
    # 为了保证“缺失补齐”和“失败重试”可控，默认使用单 worker 串行补齐。
    if num_workers != 1:
        print("[Stage1] force workers=1 for deterministic fill/retry behavior")
        num_workers = 1
    print(f"[Stage1] workers={num_workers}  GPU={_pd.USE_CUDA}")

    # 统计已存在完整样本，仅补齐缺失文件（默认）
    # 若 overwrite_existing=True，则强制全部重生（用于更新增强策略）
    pending_indices = []
    existing_complete = 0
    discarded_bad = 0
    for idx in range(total_needed):
        img_path = os.path.join(dst_images, f"sample_{idx:05d}.jpg")
        lbl_path = os.path.join(dst_labels, f"sample_{idx:05d}.txt")
        img_exists = os.path.exists(img_path)
        lbl_exists = os.path.exists(lbl_path)
        label_ok = _is_valid_pose_label_file(lbl_path)

        if (not overwrite_existing) and img_exists and lbl_exists and label_ok:
            existing_complete += 1
        else:
            # 错误样本不做原地修复：直接丢弃后重生成。
            if (not overwrite_existing) and (
                (img_exists and (not lbl_exists or not label_ok))
                or (lbl_exists and not img_exists)
            ):
                for p in (img_path, lbl_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass
                discarded_bad += 1
            pending_indices.append(idx)

    print(
        f"[Stage1] existing_complete={existing_complete}, "
        f"discarded_bad={discarded_bad}, need_fill={len(pending_indices)}"
    )

    count = 0

    def _save_result(result):
        nonlocal count
        if result is None:
            return
        idx, img_bytes, label53 = result
        det_label = _extract_stage1_pose_label_from_53(label53)

        # 将 process_dataset 输出再做一次 Stage1 屏幕翻拍增强（图像+关键点同步）。
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_np is not None:
            img_np, det_label = _augment_stage1_screen_recapture(img_np, label53)
            ok, enc = cv2.imencode(".jpg", img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                img_bytes = enc.tobytes()

        # 防御式创建，避免外部清理/并发操作导致目录瞬时缺失。
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        with open(os.path.join(dst_images, f"sample_{idx:05d}.jpg"), "wb") as f:
            f.write(img_bytes)
        with open(
            os.path.join(dst_labels, f"sample_{idx:05d}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(det_label + "\n")

        count += 1
        if count % 50 == 0:
            print(
                f"[Stage1] filled {count}/{len(pending_indices)} "
                f"(total_complete={existing_complete + count}/{total_needed}) ..."
            )

    queue = deque(pending_indices)
    attempts = 0

    while queue:
        idx = queue.popleft()

        # 双文件都已存在则直接跳过（覆盖模式下不会走到这里）
        img_path = os.path.join(dst_images, f"sample_{idx:05d}.jpg")
        lbl_path = os.path.join(dst_labels, f"sample_{idx:05d}.txt")
        label_ok = _is_valid_pose_label_file(lbl_path)
        if (not overwrite_existing) and os.path.exists(img_path) and label_ok:
            continue

        result = _pd._worker((idx, base_files, bg_files))
        attempts += 1
        if result is None:
            queue.append(idx)
            continue
        _save_result(result)

    print(
        f"[Stage1] done. newly_filled={count}, "
        f"total_complete={existing_complete + count}/{total_needed} -> {dst_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst-dir", default="train/stage1_dataset")
    parser.add_argument("--base-dir", default="train/base_images")
    parser.add_argument("--bg-dir", default="train/bg2")
    parser.add_argument("--multiplier", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="regenerate existing samples instead of skipping",
    )
    args = parser.parse_args()

    build_stage1_dataset(
        dst_dir=args.dst_dir,
        base_dir=args.base_dir,
        bg_dir=args.bg_dir,
        multiplier=args.multiplier,
        num_workers=args.num_workers,
        overwrite_existing=args.overwrite_existing,
    )
