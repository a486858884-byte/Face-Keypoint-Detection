import os
import random
import shutil


def create_debug_subset(source_base, dest_base, train_samples=1600, val_samples=160):
    """
    从源数据集中随机抽取样本，创建一个小型的调试子集。

    :param source_base: 原始数据集的根目录 (例如 'C:/my/dogcat/split_data')
    :param dest_base: 目标调试数据集的根目录 (例如 'C:/my/dogcat/debug_data')
    :param train_samples: 要抽取的训练样本数量
    :param val_samples: 要抽取的验证样本数量
    """

    # 确保目标文件夹存在
    os.makedirs(os.path.join(dest_base, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_base, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dest_base, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_base, 'val', 'labels'), exist_ok=True)

    # 定义要处理的数据集部分
    parts = {
        'train': train_samples,
        'val': val_samples
    }

    for part, num_samples in parts.items():
        print(f"--- 正在处理 {part} 部分 ---")

        source_img_dir = os.path.join(source_base, part, 'images')
        source_lbl_dir = os.path.join(source_base, part, 'labels')
        dest_img_dir = os.path.join(dest_base, part, 'images')
        dest_lbl_dir = os.path.join(dest_base, part, 'labels')

        # 获取所有可用的图像文件名
        available_images = [f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # 如果请求的样本数超过了可用数量，就使用所有可用样本
        if num_samples > len(available_images):
            print(f"警告：请求的样本数 ({num_samples}) 超过了可用数量 ({len(available_images)})。将使用所有可用样本。")
            num_samples = len(available_images)

        # 随机抽取指定数量的图像
        selected_images = random.sample(available_images, num_samples)
        print(f"已随机抽取 {len(selected_images)} 个样本。")

        # 复制图像和对应的标签文件
        count = 0
        for img_name in selected_images:
            # 构建源文件和目标文件的完整路径
            source_img_path = os.path.join(source_img_dir, img_name)
            dest_img_path = os.path.join(dest_img_dir, img_name)

            label_name = os.path.splitext(img_name)[0] + '.txt'
            source_lbl_path = os.path.join(source_lbl_dir, label_name)
            dest_lbl_path = os.path.join(dest_lbl_dir, label_name)

            # 执行复制操作
            if os.path.exists(source_lbl_path):
                shutil.copy(source_img_path, dest_img_path)
                shutil.copy(source_lbl_path, dest_lbl_path)
                count += 1

        print(f"成功复制了 {count} 对图像和标签文件到目标目录。")


if __name__ == '__main__':
    # --- 请在这里配置你的路径 ---
    source_data_path = r'C:\my\dogcat\split_data'
    debug_data_path = r'C:\my\dogcat\debug_data'

    # --- 在这里配置你想要的样本数量 ---
    num_train = 1600  # 你想要的训练样本数
    num_val = 160  # 你想要的验证样本数 (通常是训练的10%)

    create_debug_subset(source_data_path, debug_data_path, num_train, num_val)
    print("\n调试子集创建完成！")