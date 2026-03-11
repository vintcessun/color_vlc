from ultralytics import YOLO
import cv2
import numpy as np
import os


def test_inference(
    image_path="test.png",
    model_path="runs/pose/train/runs/color_qr_pose3/weights/best.pt",
):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # 1. 加载模型
    model = YOLO(model_path)

    # 2. 读取测试图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found at {image_path}")
        return

    h, w = img.shape[:2]

    # 3. 推理
    results = model.predict(img, imgsz=640, conf=0.25)

    for result in results:
        if result.keypoints is not None:
            # 提取关键点 (x, y)
            # YOLOv8 Pose 输出 keypoints 形状为 [N, 4, 2] 或 [N, 4, 3]
            kpts = result.keypoints.xyn[0].cpu().numpy()  # 归一化坐标

            if len(kpts) < 4:
                continue

            # 关键点映射回原图坐标
            src_pts = kpts[:4] * np.array([w, h])
            src_pts = src_pts.astype(np.float32)

            # 定义目标坐标 (还原为 400x400 的正方形)
            dst_pts = np.array(
                [[0, 0], [399, 0], [399, 399], [0, 399]], dtype=np.float32
            )

            # 4. 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # 5. 执行校正
            warped = cv2.warpPerspective(img, M, (400, 400))

            # 6. 保存结果
            cv2.imwrite("corrected_test.png", warped)
            print("Successfully corrected and saved to corrected_test.png")

            # 在原图上画出检测到的关键点用于调试
            for i, pt in enumerate(src_pts):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    str(i),
                    (int(pt[0]), int(pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3,
                )
            cv2.imwrite("debug_detection.png", img)
            print("Debug detection image saved to debug_detection.png")
            break
        else:
            print("No keypoints detected.")


if __name__ == "__main__":
    test_inference()
