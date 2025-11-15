import os
import cv2 #用于图像读取和处理
import numpy as np #用于数值的计算，数组操作
import torch #Pytorch的核心库
import torch.nn as nn #pytorch神经网路模块，里面有所有的网络层定义
from torch.utils.data import Dataset, DataLoader #构建和加载数据集的
from torchvision import transforms,models #包含了常用的图像变换和预训练模型

# --- 模块一：数据预处理 ---
# 目标：把图片和坐标，处理成模型能吃的 Tensor
# 需要的功能：统一尺寸、同步缩放坐标、转 Tensor、标准化
# TODO: 定义 FacePreprocess 类
class FacePreprocess:
    '''
    一个可以调用的类，负责单张图像和关键点进行完整的预处理
    包括：尺寸缩放、关键点同步缩放、转换为tensor、标准化
    '''

    # --- 类的初始化函数 (构造函数) ---
    # 当我们创建一个 FacePreprocess 对象时，这个函数会自动被调用。
    # 它的任务是提前准备好所有需要的“处理工具”。
    def __init__(self,target_size=(224,224)):
        #将目标的尺寸----宽高保存为对象属性，方便后续使用
        self.target_size=target_size

        #定义一个图像处理的“流水线” transforms.Compose会按顺序执行要处理的步骤
        self.transform = transforms.Compose([
            #先转化为Pytorch Tensor
            transforms.ToTensor(),
            #标准化Tensor 用Imagenet数据集的均值和标准差 即mean与std来标准化图像
            # 这是因为我们后面要用的 ResNet18 是在 ImageNet 上预训练的，使用相同的标准化方法可以获得最佳性能。
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    #类的调用函数
    #__call__方法，让我们可以像调用一个普通函数一样使用这个类的对象
    #例如：preprocess(image, keypoints)
    def __call__(self, image, keypoints):
        # 获取原始图像的高度和宽度，用于计算缩放比例。
        # 这个数组的形状 (shape) 是一个包含三个数字的元组 (高度, 宽度, 通道数)
        h_original, w_original = image.shape[:2]
        # 使用 OpenCV 的 resize 函数，将图像缩放到我们预设的目标尺寸。
        image_resized = cv2.resize(image,self.target_size)
        # 计算宽度和高度方向上的缩放比例。
        scale_w = self.target_size[0]/w_original
        scale_h = self.target_size[1]/h_original

        #根据计算比例缩放关键点xy坐标
        #保证图像缩放后关键点也和图像大小对应
        keypoints_resized = keypoints * np.array([scale_w, scale_h])
        # 使用我们之前定义好的 transform 流水线，对缩放后的图像进行“转 Tensor”和“标准化”。
        image_tensor = self.transform(image_resized)

        # 将处理好的关键点坐标 (NumPy 数组) 也转换为 PyTorch Tensor。
        keypoints_tensor = torch.from_numpy(keypoints_resized.astype(np.float32))

        # 返回一对处理好的、随时可以送入神经网络的 Tensor。
        return image_tensor, keypoints_tensor
# --- 模块二：自定义数据集 ---
# 目标：管理数据，实现按需加载
# 需要的功能：继承 Dataset，实现 __init__, __len__, __getitem__
# TODO: 定义 FaceKeypointDataset 类
class FaceKeypointDataset(Dataset):
# --- 类的初始化函数 (构造函数) ---
# 在这里完成所有一次性的准备工作，比如加载文件列表和数据校验。
    def __init__(self, image_dir,label_dir,preprocess):
        self.image_dir = image_dir # 保存图像文件夹的路径
        self.label_dir = label_dir # 保存标签文件夹的路径
        self.preprocess = preprocess  # 保存从外部传入的预处理对象
        # 获取图像文件夹下所有图片文件的文件名，形成一个“花名册”。
        self.image_names = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ]

        # (数据校验) 确保“花名册”上的每一张图片，都能在标签文件夹里找到对应的标签文件。
        for image_name in self.image_names:
            label_name = os.path.splitext(image_name)[0] + '.txt'
            if not os.path.exists(os.path.join(label_dir,label_name)):
                # 如果找不到，就立刻报错并停止程序，防止后续出现问题。
                raise FileNotFoundError(f'缺失标签文件：{label_name}')
        print(f'成功加载：{len(self.image_names)}个样本')

    # --- 获取数据集总长度的方法 ---
    # 这个方法必须实现，它告诉 PyTorch 这个数据集中总共有多少个样本。
    def __len__(self):
        return len(self.image_names)

    # --- 获取单个样本的方法 (核心) ---
    # 这个方法定义了“如何获取并处理第 idx 个样本”的完整流程。
    # DataLoader 会在后台不断地调用这个方法来取数据。
    def __getitem__(self, idx):
        # 1. 根据索引 idx，从“花名册”中获取对应的文件名。
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir,img_name)
        # 2. 使用 OpenCV 读取图像文件。
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像{img_path}")
        # 3. (重要细节) 将 OpenCV 默认的 BGR 颜色通道顺序转换为 RGB 顺序。
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 4. 根据图像名，构建对应的标签文件名并读取标签数据。
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir,label_name)
        keypoints = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                x, y = line.split()
                keypoints.append([float(x), float(y)])
        keypoints = np.array(keypoints, dtype=np.float32)
        # 5. (数据校验) 确保读取到的关键点数量是正确的 (100个)。
        if keypoints.shape !=(100, 2):
            raise ValueError(f"标签格式错误：{label_path}，实际关键点为{keypoints.shape[0]}个")
        # 6. (协同工作) 调用我们传入的 preprocess 对象，对图像和关键点进行预处理。
        image_tensor ,keypoints_tensor = self.preprocess(image_rgb,keypoints)
        return image_tensor ,keypoints_tensor
# --- 模块三：模型定义 ---
# 目标：构建神经网络
# 思路：使用迁移学习，改造 ResNet18
# TODO: 定义 KeypointModel 类
class KeypointModel(nn.Module):
    """
        人脸关键点检测模型，基于 ResNet18 骨干网络。
    """

    # 在这里定义模型由哪些“神经网络层”组成。
    def __init__(self,num_keypoints=100):
        # 必须调用的父类构造函数
        super().__init__()
        # 加载一个在 ImageNet 上预训练好的 ResNet18 模型。
        # pretrained=True 是迁移学习的关键，它会下载并加载已经训练好的权重。
        self.backbone = models.resnet18(pretrained=True)

        # (迁移学习的核心改造)
        # ResNet18 原始的最后一层 (self.backbone.fc) 是一个用于1000类分类的全连接层。
        # 我们要把它替换成一个符合我们任务需求的新层。
        # nn.Linear 定义了一个全连接层。
        # in_features=512: 它的输入维度必须是 512，因为这是 ResNet18 骨干网络固定的输出特征维度。
        # out_features=num_keypoints * 2: 它的输出维度必须是 100 * 2 = 200，对应100个关键点的 (x, y) 坐标。
        self.backbone.fc = nn.Linear(in_features=512, out_features=num_keypoints * 2)

    # --- 前向传播函数 ---
    # 这个方法定义了数据在模型中是如何流动的。
    def forward(self,x):
        # x 是输入的图像 Tensor，形状为 (batch_size, 3, 224, 224)。
        # 数据流经我们改造后的 ResNet18 模型。
        out = self.backbone(x)
        # 此时，out 的形状是 (batch_size, 200)。
        # 为了方便后续计算损失，我们将这个扁平的 200 维向量，重塑为 (100, 2) 的形状。
        # -1 在这里是占位符，表示自动计算 batch_size 的大小。
        return out.reshape(-1, 100, 2)



# --- 模块四：训练与验证流程 ---
# 目标：定义学习和评估的单轮次流程
# 需要的功能：一个用于训练的函数，一个用于验证的函数
# TODO: 定义 train_one_epoch 函数
# TODO: 定义 validate 函数
def train_one_epoch(model, train_loader, criterion, optimizer,device):
    '''
    对模型进行一整个轮次（epoch）的训练
    '''
    # 开启训练模式。这会启用 Dropout 和 BatchNorm 等只在训练时需要的功能。
    model.train()
    total_loss = 0

    # 从 DataLoader 中批量取出数据进行循环训练。
    for batch_idx, (images, keypoints) in enumerate(train_loader):
        images = images.to(device)
        keypoints = keypoints.to(device)

        # 1. 前向传播：将图像输入模型，得到预测的关键点。
        outputs = model(images)
        # 2. 计算损失：使用损失函数 (criterion) 比较预测值和真实值，计算出差距 (loss)。
        loss = criterion(outputs, keypoints)
        # 3. 反向传播：
        optimizer.zero_grad() # 清空上一批次计算的梯度，防止累加。
        loss.backward()  # 根据当前的损失，自动计算模型中所有参数的梯度。
        optimizer.step() # 命令优化器根据刚刚计算出的梯度，去更新模型的所有参数。
        # 累加损失值，用于后续计算平均损失。
        total_loss += loss.item()
        # 打印训练进度。
        if (batch_idx + 1)%10 ==0:
            print(f'训练批次 [{batch_idx + 1}/{len(train_loader)}], 损失: {loss.item():.6f}')
    # 返回本轮训练的平均损失。
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """
    对模型在验证集上进行评估。
    """
    model.eval()
    total_loss = 0
    # 使用 torch.no_grad() 上下文管理器，临时禁用所有梯度计算。
    # 这能极大地节省内存和计算资源，加速评估过程。
    with torch.no_grad():
        for images, keypoints in val_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            # 只进行前向传播和计算损失，没有反向传播和参数更新。
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            total_loss += loss.item()

    # 返回本轮验证的平均损失。
    return total_loss / len(val_loader)
# --- 模块五：主流程控制 ---
# 目标：组织和调度所有模块，启动训练
# 需要的功能：设置路径、实例化所有对象、编写训练循环
# TODO: 编写 if __name__ == '__main__': 部分
if __name__ == '__main__':
    # --- 1. 配置与实例化 ---
    #train_image_dir = r'C:\my\dogcat\split_data\train\images'
    #train_label_dir = r'C:\my\dogcat\split_data\train\labels'
    #val_image_dir = r'C:\my\dogcat\split_data\val\images'
    #val_label_dir = r'C:\my\dogcat\split_data\val\labels'

    train_image_dir = r'C:\my\dogcat\debug_data\train\images'
    train_label_dir = r'C:\my\dogcat\debug_data\train\labels'
    val_image_dir = r'C:\my\dogcat\debug_data\val\images'
    val_label_dir = r'C:\my\dogcat\debug_data\val\labels'





    # 实例化“备菜助理”。
    preprocess = FacePreprocess(target_size=(224,224))

    # 实例化“数据管理员”，并为他们指派“备菜助理”。
    print("加载训练集...")
    train_dataset = FaceKeypointDataset(train_image_dir,train_label_dir,preprocess)
    print("加载验证集...")
    val_dataset = FaceKeypointDataset(val_image_dir,val_label_dir,preprocess)

    # 实例化“传菜员”，负责从数据集中批量、高效地加载数据。
    train_loader = DataLoader(
        train_dataset,
        batch_size=16, # 每批次加载16个样本。
        shuffle=True,# 在每个 epoch 开始时，打乱训练集的数据顺序。
        num_workers=0,# 在 Windows 上设为 0 以避免多线程问题
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,  # 验证集通常不需要打乱。
        num_workers=0
    )

    # --- 2. 核心组件初始化 ---
    # 自动检测并设置计算设备 (优先使用 GPU)。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}")

    # 实例化“大厨”，并将其移动到指定的计算设备上。
    model = KeypointModel(num_keypoints=100).to(device)

    # 实例化“菜品评分标准”。MSELoss (均方误差损失) 是回归任务的常用选择。
    criterion = nn.MSELoss()

    # 实例化“烹饪导师”。Adam 是一种高效的优化算法。
    # model.parameters() 告诉优化器它需要更新模型中的所有可训练参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # --- 3. 训练循环 ---
    # 设置训练的总轮次。
    num_epochs = 5
    # 初始化一个变量，用于记录历史上最好的 (最低的) 验证损失。
    best_val_loss = float('inf')

    print("\n===== 开始训练 =====")
    # 外层循环，控制总共训练多少个 epoch。
    for epoch in range(num_epochs):
        # 调用训练函数，进行一轮完整的训练。
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,device)

        # 调用验证函数，对当前模型状态进行一次评估。
        val_loss = validate(model, val_loader, criterion, device)

        # 打印本轮的训练和验证结果。
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"训练集平均损失: {train_loss:.6f}")
        print(f"验证集平均损失: {val_loss:.6f}")

        # 检查本轮的验证损失是否是历史最佳。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 如果是，就将当前模型的参数 (state_dict) 保存到文件中。
            torch.save(model.state_dict(), r'C:\my\dogcat\split_data\model.pth')
            print(f"✅ 已保存最佳模型（验证损失: {best_val_loss:.6f}）")
    print("\n===== 训练完成 =====")
    print(f"最佳模型已保存")