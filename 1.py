import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from unet import UNet

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = UNet(n_channels=3, n_classes=10, bilinear=False)  # 确保类别数正确

# 加载 checkpoint 并忽略 "mask_values" 错误
checkpoint_path = "./checkpoints/checkpoint_epoch5.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

if "mask_values" in checkpoint:
    del checkpoint["mask_values"]  # 移除不属于 UNet 的键

model.load_state_dict(checkpoint, strict=False)  # 忽略额外的 keys
model.to(device)
model.eval()

# 预处理输入图像
def preprocess_image(img_path):
    img = cv2.imread(img_path)  # 读取图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道
    img = cv2.resize(img, (256, 256))  # 调整大小
    img = img / 255.0  # 归一化
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 添加 batch 维度
    return img.to(device)

# 运行预测
def predict(img_path, output_path="output.png"):
    img = preprocess_image(img_path)

    with torch.no_grad():
        output = model(img)  # 预测
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # 获取类别索引

    # 颜色映射（可选）
    color_map = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0]
    ])
    color_mask = color_map[output]

    cv2.imwrite(output_path, color_mask)  # 保存结果
    print(f"Prediction saved to {output_path}")

# 测试新图片
predict("/Users/xiaofuqing/Desktop/outsource/Pytorch-UNet-master/JPEGImages/1.jpg")