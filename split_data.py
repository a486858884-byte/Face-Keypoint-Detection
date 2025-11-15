import os
import shutil
import random

# --------------------------
# 第一步：配置你的路径（只改这里！其他不用动）
# --------------------------
# 1. 原始图像文件夹路径（存放所有照片的文件夹名）
ORIGINAL_IMAGE_DIR = "images"
# 2. 原始总标签txt文件路径（包含所有标签的txt文件名）
ORIGINAL_LABEL_FILE = "labels.txt"
# 3. 拆分后的数据存放目录名（自动创建，如split_data）
SPLIT_OUTPUT_DIR = "split_data"
# 4. 训练集比例（0.8=80%训练，20%验证）
TRAIN_RATIO = 0.8
# 5. 随机种子（固定为42，确保拆分结果一致）
RANDOM_SEED = 42
# 6. 关键配置：你的标签是100个关键点（对应200个坐标）
NUM_KEYPOINTS = 100  # 已改为100，无需再动

# --------------------------
# 第二步：自动创建目标文件夹（不用改）
# --------------------------
# 训练集路径（图像+标签）
TRAIN_IMAGE_DIR = os.path.join(SPLIT_OUTPUT_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(SPLIT_OUTPUT_DIR, "train", "labels")
# 验证集路径（图像+标签）
VAL_IMAGE_DIR = os.path.join(SPLIT_OUTPUT_DIR, "val", "images")
VAL_LABEL_DIR = os.path.join(SPLIT_OUTPUT_DIR, "val", "labels")

# 创建所有文件夹（不存在则创建，存在不报错）
for dir_path in [TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_IMAGE_DIR, VAL_LABEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)
print(f"✅ 目标文件夹已创建：{SPLIT_OUTPUT_DIR}")

# --------------------------
# 第三步：读取原始图像，确保存在（不用改）
# --------------------------
# 支持的图像格式（jpg/jpeg/png）
image_extensions = (".jpg", ".jpeg", ".png")
# 读取有效图像文件（排除隐藏文件如.DS_Store）
image_names = [
    f for f in os.listdir(ORIGINAL_IMAGE_DIR)
    if f.lower().endswith(image_extensions) and not f.startswith(".")
]

# 检查图像文件夹是否有照片
if len(image_names) == 0:
    raise ValueError(f"❌ 原始图像文件夹 {ORIGINAL_IMAGE_DIR} 中没有找到照片！请检查路径。")
print(f"✅ 从 {ORIGINAL_IMAGE_DIR} 找到 {len(image_names)} 张照片")

# 图像名存为集合，方便快速匹配
image_name_set = set(image_names)

# --------------------------
# 第四步：解析总标签txt，匹配图像（已适配100个关键点）
# --------------------------
valid_image_label_pairs = []  # 存储“图像名-坐标”有效配对

with open(ORIGINAL_LABEL_FILE, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):  # line_idx：行号，方便定位错误
        line = line.strip()  # 去除行首尾空格/换行符

        # 跳过空行
        if not line:
            continue

        # 拆分该行：第一个是图像名，后面是坐标
        line_parts = line.split()
        # 检查：至少包含“1个图像名 + 100×2=200个坐标”
        required_len = 1 + NUM_KEYPOINTS * 2
        if len(line_parts) < required_len:
            print(f"⚠️  跳过第{line_idx}行：内容不足（需1个图像名+{required_len - 1}个坐标，实际{len(line_parts)}个元素）")
            continue

        # 提取图像名和坐标
        img_name_from_label = line_parts[0]  # 标签行中的图像名
        coords_str = line_parts[1:]  # 坐标部分（字符串列表）

        # 1. 检查图像是否在原始图像文件夹中
        if img_name_from_label not in image_name_set:
            print(f"⚠️  跳过第{line_idx}行：图像 {img_name_from_label} 不在 {ORIGINAL_IMAGE_DIR} 中")
            continue

        # 2. 检查坐标数是否为“100×2=200个”
        if len(coords_str) != NUM_KEYPOINTS * 2:
            print(
                f"⚠️  跳过第{line_idx}行：{img_name_from_label} 坐标数错误（需{NUM_KEYPOINTS * 2}个，实际{len(coords_str)}个）")
            continue

        # 3. 检查坐标是否都是数字（避免非数字字符）
        try:
            coords = list(map(float, coords_str))  # 转浮点数列表
        except ValueError:
            print(f"⚠️  跳过第{line_idx}行：{img_name_from_label} 坐标包含非数字字符")
            continue

        # 所有检查通过，加入有效配对
        valid_image_label_pairs.append((img_name_from_label, coords))

# 检查有效配对数量
if len(valid_image_label_pairs) == 0:
    raise ValueError("❌ 没有找到有效匹配的图像-标签对！请检查标签txt格式和图像文件夹。")
print(f"✅ 找到 {len(valid_image_label_pairs)} 个有效图像-标签对")

# --------------------------
# 第五步：打乱并拆分训练集/验证集（不用改）
# --------------------------
random.seed(RANDOM_SEED)  # 固定随机种子
random.shuffle(valid_image_label_pairs)  # 打乱配对列表

# 计算拆分数量
total_valid = len(valid_image_label_pairs)
train_num = int(total_valid * TRAIN_RATIO)
val_num = total_valid - train_num

# 拆分列表
train_pairs = valid_image_label_pairs[:train_num]
val_pairs = valid_image_label_pairs[train_num:]

print(f"✅ 拆分完成：训练集 {train_num} 个，验证集 {val_num} 个")

# --------------------------
# 第六步：复制训练集图像和标签（已适配100个关键点）
# --------------------------
print("\n📥 正在复制训练集...")
train_success = 0
for img_name, coords in train_pairs:
    # 复制图像
    src_img_path = os.path.join(ORIGINAL_IMAGE_DIR, img_name)
    dst_img_path = os.path.join(TRAIN_IMAGE_DIR, img_name)
    try:
        shutil.copy(src_img_path, dst_img_path)
    except Exception as e:
        print(f"⚠️  训练集图像 {img_name} 复制失败：{str(e)}")
        continue

    # 生成标签文件（图像名.jpg → 图像名.txt）
    img_prefix = os.path.splitext(img_name)[0]  # 去除扩展名（如wflw_9997.jpg→wflw_9997）
    dst_label_path = os.path.join(TRAIN_LABEL_DIR, f"{img_prefix}.txt")

    # 写入标签（100行，每行1个关键点：x y）
    with open(dst_label_path, "w", encoding="utf-8") as f:
        for i in range(NUM_KEYPOINTS):  # 循环100次，生成100个关键点
            x = coords[2 * i]
            y = coords[2 * i + 1]
            f.write(f"{x} {y}\n")

    train_success += 1
print(f"✅ 训练集复制完成：成功 {train_success} 个，失败 {len(train_pairs) - train_success} 个")

# --------------------------
# 第七步：复制验证集图像和标签（已适配100个关键点）
# --------------------------
print("\n📥 正在复制验证集...")
val_success = 0
for img_name, coords in val_pairs:
    # 复制图像
    src_img_path = os.path.join(ORIGINAL_IMAGE_DIR, img_name)
    dst_img_path = os.path.join(VAL_IMAGE_DIR, img_name)
    try:
        shutil.copy(src_img_path, dst_img_path)
    except Exception as e:
        print(f"⚠️  验证集图像 {img_name} 复制失败：{str(e)}")
        continue

    # 生成标签文件
    img_prefix = os.path.splitext(img_name)[0]
    dst_label_path = os.path.join(VAL_LABEL_DIR, f"{img_prefix}.txt")

    # 写入标签（100行，每行1个关键点）
    with open(dst_label_path, "w", encoding="utf-8") as f:
        for i in range(NUM_KEYPOINTS):  # 循环100次
            x = coords[2 * i]
            y = coords[2 * i + 1]
            f.write(f"{x} {y}\n")

    val_success += 1
print(f"✅ 验证集复制完成：成功 {val_success} 个，失败 {len(val_pairs) - val_success} 个")

# --------------------------
# 第八步：显示最终结果（不用改）
# --------------------------
print("\n🎉 所有操作完成！最终结果：")
print(f"训练集图像：{len(os.listdir(TRAIN_IMAGE_DIR))} 张 → 路径：{TRAIN_IMAGE_DIR}")
print(f"训练集标签：{len(os.listdir(TRAIN_LABEL_DIR))} 个 → 路径：{TRAIN_LABEL_DIR}")
print(f"验证集图像：{len(os.listdir(VAL_IMAGE_DIR))} 张 → 路径：{VAL_IMAGE_DIR}")
print(f"验证集标签：{len(os.listdir(VAL_LABEL_DIR))} 个 → 路径：{VAL_LABEL_DIR}")

# 检查图像和标签数量是否匹配
if len(os.listdir(TRAIN_IMAGE_DIR)) != len(os.listdir(TRAIN_LABEL_DIR)):
    print("⚠️  警告：训练集图像数和标签数不匹配！可能有复制失败的文件。")
if len(os.listdir(VAL_IMAGE_DIR)) != len(os.listdir(VAL_LABEL_DIR)):
    print("⚠️  警告：验证集图像数和标签数不匹配！可能有复制失败的文件。")