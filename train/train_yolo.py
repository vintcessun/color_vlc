from ultralytics import YOLO
import os

# =============================================================================
# 关键点布局（共 53 pts），与 process_dataset.py 的 get_base_keypoints() 对应：
#   [0]      Red   Finder 中心 (top-left)
#   [1]      Green Finder 中心 (top-right)
#   [2]      Blue  Finder 中心 (bottom-left)
#   [3..48]  Alignment Pattern 中心（Version-40 共 46 个，行优先）
#   [49]     TL 角点
#   [50]     TR 角点
#   [51]     BR 角点
#   [52]     BL 角点
# =============================================================================
NUM_KPT = 53


def train_yolo():
    # 1. 创建数据配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "dataset")

    data_yaml = f"""
path: {dataset_path}
train: images
val: images

names:
  0: color_qr

# Keypoints: x, y, visible (v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible)
# 53 keypoints: 3 Finder Pattern centers + 46 Alignment Pattern centers + 4 corners
kpt_shape: [{NUM_KPT}, 3]
"""
    with open("train/data.yaml", "w") as f:
        f.write(data_yaml)

    # 2. 初始化模型
    # 53 个稠密关键点检测任务较复杂，使用 yolov8m-pose（中型）以获得足够容量。
    # 若显存充裕可换用 yolov8l-pose.pt 或 yolov8x-pose.pt 进一步提升精度。
    model = YOLO("yolov8m-pose.pt")

    # 3. 训练模型
    model.train(
        data="train/data.yaml",
        epochs=300,
        imgsz=(384, 640),  # 16:9，对应 4K 下采样后 1280×720 的长边 640
        batch=8,  # m 模型显存占用更高，适当降低 batch
        project="train/runs",
        name="color_qr_pose_53kpt",
        workers=0,  # Windows 下强制主进程加载数据
        # 关键点损失权重：增大 kobj/kpt 权重以强化密集小目标关键点
        pose=12.0,  # keypoint loss gain
        fliplr=0.0,  # 禁用水平翻转（翻转会破坏 Red/Green/Blue Finder 的空间语义）
        mosaic=0.5,
    )

    # 4. 导出为 ONNX 格式，方便 Rust 使用
    success = model.export(format="onnx")
    print(f"Model exported to: {success}")


if __name__ == "__main__":
    train_yolo()
