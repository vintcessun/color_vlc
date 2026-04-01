import os
import sys

os.environ["SSL_CERT_FILE"] = os.path.join(
    sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(__file__),
    "certifi",
    "cacert.pem",
)


import cv2
import numpy as np
from qreader import QReader
import sys
import warnings
from typing import Any


# 屏蔽 QReader 的编码警告
warnings.filterwarnings("ignore", category=UserWarning)


class ColorDecoder:
    def __init__(self):
        # 提高识别精度，尝试多个频率
        self.qreader = QReader(model_size="s")
        # 预生成的不同尺寸定位块模板
        self.templates = self._generate_templates()

    def _generate_templates(self):
        """生成 7x7 定位块在不同缩放尺度下的彩色模板 (从大到小: 120x120 到 8x8 像素)"""
        templates: list[dict[str, Any]] = []
        # BGR 颜色定义
        colors = {
            "TL": [0, 0, 255],  # 红
            "TR": [0, 255, 0],  # 绿
            "BL": [255, 0, 0],  # 蓝
        }

        for size in range(120, 7, -4):
            scale_templates: dict[str, Any] = {"size": size}
            m = max(1, round(size / 7.0))
            m2 = 2 * m

            for name, color in colors.items():
                # 创建 3 通道白色背景模板
                tpl = np.ones((size, size, 3), dtype=np.uint8) * 255
                # 1. 外层 7x7 (彩色)
                cv2.rectangle(tpl, (0, 0), (size - 1, size - 1), color, -1)
                # 2. 内部 5x5 (白色)
                cv2.rectangle(
                    tpl, (m, m), (size - 1 - m, size - 1 - m), (255, 255, 255), -1
                )
                # 3. 最中心 3x3 (彩色)
                cv2.rectangle(tpl, (m2, m2), (size - 1 - m2, size - 1 - m2), color, -1)
                scale_templates[name] = tpl

            templates.append(scale_templates)
        return templates

    def _draw_finder_pattern(
        self,
        img: np.ndarray,
        r_module: int,
        c_module: int,
        box_size: int = 4,
        margin: int = 4,
        color: Any = 0,
        white: Any = 255,
    ):
        """
        在指定 module 坐标处绘制标准 7x7 定位块。
        支持单通道或多通道图像。
        """
        r = r_module * box_size + margin
        c = c_module * box_size + margin
        s = 7 * box_size

        # 1. 外层 7x7
        cv2.rectangle(img, (c, r), (c + s - 1, r + s - 1), color, -1)
        # 2. 内部 5x5
        r1, c1 = r + box_size, c + box_size
        s1 = 5 * box_size
        cv2.rectangle(img, (c1, r1), (c1 + s1 - 1, r1 + s1 - 1), white, -1)
        # 3. 最中心 3x3
        r2, c2 = r + 2 * box_size, c + 2 * box_size
        s2 = 3 * box_size
        cv2.rectangle(img, (c2, r2), (c2 + s2 - 1, r2 + s2 - 1), color, -1)

    def _order_points(self, pts: np.ndarray):
        """
        对四个顶点进行排序：左上、右上、右下、左下
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # tl
        rect[2] = pts[np.argmax(s)]  # br

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # tr
        rect[3] = pts[np.argmax(diff)]  # bl
        return rect

    def _find_finder_patterns(self, frame: np.ndarray):
        """
        基于彩色模板匹配寻找定位块
        TL (Red), TR (Green), BL (Blue)
        """
        best_overall_val = -1.0
        best_pts = None

        # 遍历尺度
        for stmts in self.templates:
            size = stmts["size"]
            if frame.shape[0] < size or frame.shape[1] < size:
                continue

            # 使用彩色图像直接匹配彩色模板
            res_tl = cv2.matchTemplate(frame, stmts["TL"], cv2.TM_CCOEFF_NORMED)
            res_tr = cv2.matchTemplate(frame, stmts["TR"], cv2.TM_CCOEFF_NORMED)
            res_bl = cv2.matchTemplate(frame, stmts["BL"], cv2.TM_CCOEFF_NORMED)

            _, max_val_tl, _, max_loc_tl = cv2.minMaxLoc(res_tl)
            _, max_val_tr, _, max_loc_tr = cv2.minMaxLoc(res_tr)
            _, max_val_bl, _, max_loc_bl = cv2.minMaxLoc(res_bl)

            # 综合评分
            score = max_val_tl + max_val_tr + max_val_bl

            # 在彩色匹配下，阈值可以稍微放低一点，因为颜色约束很强
            if (
                score > best_overall_val
                and max_val_tl > 0.4
                and max_val_tr > 0.4
                and max_val_bl > 0.4
            ):
                offset = size / 2.0
                pt_tl = (max_loc_tl[0] + offset, max_loc_tl[1] + offset)
                pt_tr = (max_loc_tr[0] + offset, max_loc_tr[1] + offset)
                pt_bl = (max_loc_bl[0] + offset, max_loc_bl[1] + offset)

                best_overall_val = score
                best_pts = (pt_tl, pt_tr, pt_bl)

                # 如果评分非常高，可以提前终止
                if score > 2.7:
                    break

        return best_pts

    def _get_warped_frame(self, frame: np.ndarray):
        """
        执行截取、旋转与缩放：利用定位块位置进行校正
        并在返回前采样定位块颜色，并强制将定位块区域涂黑，确保识别稳定。
        """
        pts = self._find_finder_patterns(frame)
        if pts is None:
            return None, None

        pt_tl, pt_tr, pt_bl = pts

        # 目标参数 (Version 30: modules=137, border=1, box_size=4)
        # 二维码核心大小是 556 像素。我们左右各留 40 像素余量（共 80 像素）。
        margin_px = 40
        qr_size = 556
        target_size = qr_size + 2 * margin_px

        # 定位块中心点相对于 QR 核心左上角的坐标是 18.0 和 538.0
        # 加上外围余量 margin_px
        p1 = 18.0 + margin_px
        p2 = 538.0 + margin_px

        src = np.array([pt_tl, pt_tr, pt_bl], dtype=np.float32)
        dst = np.array([[p1, p1], [p2, p1], [p1, p2]], dtype=np.float32)

        M = cv2.getAffineTransform(src, dst)

        warped = cv2.warpAffine(
            frame,
            M,
            (target_size, target_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # 在涂黑前，从中心采样参考颜色 (BGR)
        # TL 是红色, TR 是绿色, BL 是蓝色
        ip1 = int(p1)
        ip2 = int(p2)
        ref_colors: dict[str, Any] = {
            "r": warped[ip1, ip1].astype(np.float32),
            "g": warped[ip1, ip2].astype(np.float32),
            "b": warped[ip2, ip1].astype(np.float32),
            "w": np.array([255, 255, 255], dtype=np.float32),
        }

        # 强制涂黑定位块 (BGR: 黑[0,0,0], 白[255,255,255])
        # 模块位置 0, 130 (即 137-7)
        # 我们需要传入新的 margin 参数：原 border=1 像素 * box_size=4 + 外部余量 margin_px
        m_start = 0
        m_end = 130
        total_margin = 4 + margin_px

        self._draw_finder_pattern(
            warped,
            m_start,
            m_start,
            margin=total_margin,
            color=(0, 0, 0),
            white=(255, 255, 255),
        )
        self._draw_finder_pattern(
            warped,
            m_start,
            m_end,
            margin=total_margin,
            color=(0, 0, 0),
            white=(255, 255, 255),
        )
        self._draw_finder_pattern(
            warped,
            m_end,
            m_start,
            margin=total_margin,
            color=(0, 0, 0),
            white=(255, 255, 255),
        )

        return warped, ref_colors

    def _extract_qr_bits(self, warped_frame: np.ndarray, ref_colors: dict[str, Any]):
        """
        从校正并修复定位块后的彩色帧中提取 A, B 两个通道。
        """
        h, w = warped_frame.shape[:2]
        float_img = warped_frame.astype(np.float32)

        # 1. 使用传入的实时参考颜色 (BGR 顺序)
        ref_r = ref_colors["r"]
        ref_g = ref_colors["g"]
        ref_b = ref_colors["b"]
        ref_w = ref_colors["w"]

        # 2. 计算距离
        dist_r = np.linalg.norm(float_img - ref_r, axis=2)
        dist_g = np.linalg.norm(float_img - ref_g, axis=2)
        dist_b = np.linalg.norm(float_img - ref_b, axis=2)
        dist_w = np.linalg.norm(float_img - ref_w, axis=2)

        # 判定该像素是否足够“接近”某种颜色，且比白色更近
        # 我们不再单纯依赖 is_colored 阈值，而是直接看竞争结果
        is_r = (dist_r < dist_g) & (dist_r < dist_b) & (dist_r < dist_w)
        is_g = (dist_g < dist_r) & (dist_g < dist_b) & (dist_g < dist_w)
        is_b = (dist_b < dist_r) & (dist_b < dist_g) & (dist_b < dist_w)

        img_a = np.ones((h, w), dtype=np.uint8) * 255
        img_b = np.ones((h, w), dtype=np.uint8) * 255

        # A通道: 红色或蓝色区域为黑 (0)
        img_a[is_r | is_b] = 0
        # B通道: 绿色或蓝色区域为黑 (0)
        img_b[is_g | is_b] = 0

        # 4. 再次强制重绘定位块确保 100% 标准
        m_start, m_end = 0, 130
        margin_px = 40
        total_margin = 4 + margin_px
        for img in [img_a, img_b]:
            self._draw_finder_pattern(img, m_start, m_start, margin=total_margin)
            self._draw_finder_pattern(img, m_start, m_end, margin=total_margin)
            self._draw_finder_pattern(img, m_end, m_start, margin=total_margin)

            # 去噪
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img[:] = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        return img_a, img_b

    def decode(self, video_path: str, out_bin: str, vout_bin: str):
        cap = cv2.VideoCapture(video_path)
        decoded_chunks = {}
        total_chunks = -1
        frame_idx = 0

        print(f"Starting decode: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            warped, ref_colors = self._get_warped_frame(frame)
            if warped is None or ref_colors is None:
                print(f"Frame {frame_idx}: No QR code detected.")
                frame_idx += 1
                continue

            img_a, img_b = self._extract_qr_bits(warped, ref_colors)
            # img_a, img_b = self._extract_qr_bits(frame)

            # 调试输出
            cv2.imwrite(f"debug/{frame_idx}_frame.png", frame)
            cv2.imwrite(f"debug/{frame_idx}_warped.png", warped)
            cv2.imwrite(f"debug/{frame_idx}_a.png", img_a)
            cv2.imwrite(f"debug/{frame_idx}_b.png", img_b)

            for img, channel in [(img_a, "A"), (img_b, "B")]:
                # 转换为 RGB 因为 QReader 需要
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # 使用 detect_and_decode 直接获取内容
                decoded_list = self.qreader.detect_and_decode(image=rgb)

                for res in decoded_list:
                    if res:
                        try:
                            # 尝试 latin-1 恢复原始字节流
                            res_bytes = res.encode("latin-1")
                            if len(res_bytes) >= 10:
                                idx = int.from_bytes(res_bytes[0:4], "big")
                                print(
                                    f"Frame {frame_idx} Channel {channel}: Decoded chunk index {idx}"
                                )
                                total = int.from_bytes(res_bytes[4:8], "big")
                                len_payload = int.from_bytes(res_bytes[8:10], "big")
                                payload = res_bytes[10 : 10 + len_payload]

                                if idx not in decoded_chunks:
                                    decoded_chunks[idx] = payload
                                    total_chunks = total
                        except Exception as e:
                            print(f"Error decoding chunk: {e}")
                            cv2.imwrite(
                                f"error_{frame_idx}_{channel}.png", img
                            )  # 保存错误帧以供分析

            print(f"Frame {frame_idx}: Decoded {len(decoded_chunks)} chunks...")
            frame_idx += 1

        cap.release()

        # 汇总逻辑保持不变...
        if not decoded_chunks:
            print("Failed to decode any data.")
            return

        num_chunks = (
            total_chunks if total_chunks > 0 else max(decoded_chunks.keys()) + 1
        )
        print(f"\nFinal: {len(decoded_chunks)}/{num_chunks} chunks captured.")

        all_data = b""
        validity = []
        CHUNK_SIZE = 2940  # 假设的块大小

        for i in range(num_chunks):
            if i in decoded_chunks:
                data = decoded_chunks[i]
                all_data += data
                validity.extend([0xFF] * len(data))
            else:
                all_data += b"\x00" * CHUNK_SIZE
                validity.extend([0x00] * CHUNK_SIZE)

        with open(out_bin, "wb") as f:
            f.write(all_data)
        with open(vout_bin, "wb") as f:
            f.write(bytes(validity))
        print(f"Saved to {out_bin}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python decoder.py <video> <out.bin> <vout.bin>")
    else:
        decoder = ColorDecoder()
        decoder.decode(sys.argv[1], sys.argv[2], sys.argv[3])

        # --- 在这里添加 ---
        print("\n[Success] All processes finished. Exiting...")
        # 强制退出进程，不触发 Python 的正常对象销毁流程（避免垃圾回收报错）
        os._exit(0)
