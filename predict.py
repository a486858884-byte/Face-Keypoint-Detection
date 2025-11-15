import cv2
import torch
import numpy as np

# --- [关键] 从我们之前的训练脚本中，导入“模型图纸”和“预处理图纸” ---
# 假设您的训练脚本名为 replay.py，并且 predict.py 和它在同一个文件夹下
from replay import KeypointModel, FacePreprocess

# ===================================================================
# 模块一：配置“诊室”
# ===================================================================

# 1. 配置“医生执照”（模型权重文件）的路径
MODEL_PATH = r'C:\my\dogcat\split_data\model.pth' # 确保这个路径是正确的！

# 2. 配置“病人”（新图片）的路径
IMAGE_PATH = r'C:\my\dogcat\jd_test_281.jpg' # 替换成你自己的新图片路径！

# 3. 配置“手术台”（计算设备）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


# ===================================================================
# 模块二：请“医生”上岗
# ===================================================================

print("正在加载训练好的模型...")
# 1. 实例化一个和训练时结构完全相同的“空”模型
#    这就像为“灵魂”准备一个“身体”
model = KeypointModel(num_keypoints=100)

# 2. 加载我们保存的最佳权重 (载入“灵魂”)
#    map_location=DEVICE 确保了即使当初是用 GPU 训练的，在没有 GPU 的电脑上也能用 CPU 加载
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# 3. 将模型部署到“手术台”上，并切换到“出诊模式”(评估模式)
model.to(DEVICE)
model.eval() # 极其重要！这会关闭 Dropout 和 BatchNorm 的训练行为
print("模型加载成功，医生已就绪！")


# ===================================================================
# 模块三：对“病人”进行“术前检查与处理”
# ===================================================================

print(f"正在处理图片: {IMAGE_PATH}")
# 1. 实例化一个和训练时配置完全一样的“备菜助理”
preprocess = FacePreprocess(target_size=(224, 224))

# 2. 读取原始图片
original_image = cv2.imread(IMAGE_PATH)
if original_image is None:
    raise FileNotFoundError(f"无法找到或读取图片: {IMAGE_PATH}")

# 3. (重要) 预处理需要一个假的 keypoints 参数，因为图纸上是这么设计的。
#    我们传一个空的 NumPy 数组即可，它在内部不会被使用。
image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_tensor, _ = preprocess(image_rgb, np.zeros((100, 2)))

# 4. (重要) 为模型输入增加一个“批次维度”。
#    我们的模型被设计为接收一批 (batch) 图片，即使我们只预测一张。
#    所以需要把形状从 (C, H, W) -> (1, C, H, W)，这里的 1 就是 batch_size。
image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
print("图片预处理完成。")


# ===================================================================
# 模块四：“医生”进行诊断 (模型推理)
# ===================================================================

print("模型正在进行预测...")
# 1. 使用 with torch.no_grad() 关闭梯度计算，以节省资源并加速
with torch.no_grad():
    # 2. 执行前向传播，得到预测结果
    predicted_keypoints_normalized = model(image_tensor)
print("预测完成！")


# ===================================================================
# 模块五：解读“诊断报告”并“可视化”
# ===================================================================

# 1. 将预测结果从 GPU 移回 CPU，并转换为 NumPy 数组，方便后续用 OpenCV 处理
#    .squeeze(0) 用于移除我们之前添加的那个批次维度 (1, 100, 2) -> (100, 2)
predicted_keypoints = predicted_keypoints_normalized.squeeze(0).cpu().numpy()

# 2. (关键) 将预测出的、在 224x224 尺寸下的坐标，“反向缩放”回原始图像的尺寸
h_original, w_original = original_image.shape[:2]
scale_w = w_original / 224.0
scale_h = h_original / 224.0
predicted_keypoints_original_scale = predicted_keypoints * np.array([scale_w, scale_h])

# 3. 在原始图片上，将预测出的关键点画出来
output_image = original_image.copy() # 创建一个副本以保持原始图片不变
for (x, y) in predicted_keypoints_original_scale:
    # cv2.circle 用于画圆点
    # (int(x), int(y)) 是圆心坐标，必须是整数
    # 3 是半径 (像素)
    # (0, 255, 0) 是颜色 (BGR格式的绿色)
    # -1 表示填充整个圆形
    cv2.circle(output_image, (int(x), int(y)), 3, (0, 255, 0), -1)

# 4. 显示最终的结果图片
print("正在显示预测结果... 按下任意键关闭窗口。")
cv2.imshow('Predicted Keypoints', output_image)
cv2.waitKey(0) # 等待用户按下任意按键
cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口

# (可选) 如果你想把结果保存成文件，可以取消下面两行的注释
# cv2.imwrite('result.jpg', output_image)
# print("结果已保存为 result.jpg")