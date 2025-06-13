import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import os
from PIL import Image
import random
from sklearn.metrics import precision_score, recall_score, f1_score

# --- 1. 数据加载类 ---
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        folder_path = os.path.join(root_dir, phase)
        real_images = []
        fake_images = []

        # 遍历 'fake' 和 'real' 文件夹加载图片
        for label_name in ['fake', 'real']:
            label_folder = os.path.join(folder_path, label_name)
            # 确保文件夹存在，避免错误
            if not os.path.exists(label_folder):
                print(f"警告: 路径不存在 {label_folder}，跳过此类别。")
                continue
            images = [os.path.join(label_folder, img) for img in os.listdir(label_folder) if img.endswith('.png')]
            if label_name == 'real':
                real_images.extend(images)
            else:
                fake_images.extend(images)
        
        # 应用限制，并确保每个类别的图片数量平衡
        if limit:
            real_images = real_images[:min(len(real_images), limit // 2)]
            fake_images = fake_images[:min(len(fake_images), limit // 2)]
        
        self.image_paths.extend(real_images)
        self.labels.extend([0] * len(real_images)) # 真实图片标签为 0

        self.image_paths.extend(fake_images)
        self.labels.extend([1] * len(fake_images)) # 伪造图片标签为 1

        print(f"Dataset phase: {phase}, Real images: {len(real_images)}, Fake images: {len(fake_images)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- 2. 纯经典对比模型 (Classical_CNN_DD) ---
class Classical_CNN_DD(nn.Module):
    def __init__(self, freeze_cnn=False):
        super(Classical_CNN_DD, self).__init__()
        
        # 经典CNN特征提取器 (ResNet-18)
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 移除ResNet的最后的全连接层，只保留特征提取部分
        self.cnn = nn.Sequential(*(list(self.cnn.children())[:-1])) 

        # 根据freeze_cnn参数决定是否冻结CNN权重
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval() # 评估模式，禁用 dropout/batchnorm 更新

        # 纯经典分类器，替代混合模型中的量子部分
        # 增加复杂度以模拟量子神经网络可能引入的非线性能力
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), # ResNet-18的特征输出是512维
            nn.ReLU(),           # 使用ReLU激活函数
            nn.Dropout(0.5),     # 添加Dropout层防止过拟合
            nn.Linear(128, 2)    # 输出2分类logits (真实/伪造)
        )

    def forward(self, x):
        # CNN特征提取
        x = self.cnn(x)
        # 展平特征，从 (batch_size, 512, 1, 1) 变为 (batch_size, 512)
        x = x.view(x.size(0), -1) 
        # 经过纯经典分类器处理
        x = self.classifier(x)
        return x

# --- 3. 训练函数 ---
def train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs=10, device=None, scheduler=None, model_save_path='best_model.pth'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"训练数据加载器批次数量: {len(train_dataloader)}")
    print(f"验证数据加载器批次数量: {len(validation_dataloader)}")

    best_val_f1 = -1.0 # 用于保存最佳模型
    
    for epoch in range(num_epochs):
        print(f'\nStarting Epoch {epoch + 1}/{num_epochs} for {os.path.basename(model_save_path).replace(".pth", "")}')
        
        # --- 训练阶段 ---
        model.train() # 设置模型为训练模式
        total_train_loss = 0.0
        total_train_samples = 0
        
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)
            optimizer.zero_grad() 
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size
        
        epoch_avg_train_loss = total_train_loss / total_train_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}] Training Average Loss: {epoch_avg_train_loss:.4f}')

        # --- 验证阶段 ---
        model.eval() # 设置模型为评估模式
        total_val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad(): # 在评估阶段不计算梯度
            for images, labels in validation_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels) 
                total_val_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())
        
        epoch_avg_val_loss = total_val_loss / len(validation_dataloader.dataset) 
        val_accuracy = 100 * np.sum(np.array(all_val_predictions) == np.array(all_val_labels)) / len(validation_dataloader.dataset)
        val_precision = precision_score(all_val_labels, all_val_predictions, average='binary', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_predictions, average='binary', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='binary', zero_division=0)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Average Loss: {epoch_avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.4f}')

        if scheduler:
            scheduler.step(epoch_avg_val_loss) 

        # 保存最佳模型：如果当前验证 F1 分数更高，则保存模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f'Saving best model with F1 score: {best_val_f1:.4f} to {model_save_path}')

    print('Finished Training')

# --- 4. 评估函数 ---
def evaluate_model(model, dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval() # 确保模型处于评估模式

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    return accuracy, precision, recall, f1

# --- 5. 主函数 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 图像预处理和数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集根目录
    # 请确保 'faceforensec++\\1000_videos' 路径是正确的，并且其下有 'train', 'validation', 'test' 文件夹
    # 且每个文件夹内有 'fake' 和 'real' 子文件夹，包含 .png 图像文件。
    root_dir = 'faceforensec++\\1000_videos' 
    
    # 实例化数据集
    # limit 参数用于限制每个数据集加载的图片数量，方便快速测试和调试
    train_dataset = DeepfakeDataset(root_dir=root_dir, phase='train', transform=transform, limit=600) # 建议增加训练数据量
    validation_dataset = DeepfakeDataset(root_dir=root_dir, phase='validation', transform=transform, limit=100)
    test_dataset = DeepfakeDataset(root_dir=root_dir, phase='test', transform=transform, limit=100)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"训练数据加载器长度: {len(train_loader)}")
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    print(f"验证数据加载器长度: {len(validation_loader)}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试数据加载器长度: {len(test_loader)}")

    # 定义通用的训练参数
    num_epochs = 30 # 训练轮次
    learning_rate = 0.001 # 初始学习率
    # 类别权重，用于处理类别不平衡问题 (伪造图片权重更高)
    class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights) # 交叉熵损失函数

    # --- 纯经典模型 (Classical_CNN_DD) 的训练和评估 ---
    print("\n" + "="*50 + "\n--- 开始训练和评估 纯经典模型 (Classical_CNN_DD) ---\n" + "="*50)
    classical_model = Classical_CNN_DD(freeze_cnn=False) # 实例化纯经典模型
    classical_optimizer = optim.Adam(classical_model.parameters(), lr=learning_rate) # Adam优化器
    # 学习率调度器
    classical_scheduler = optim.lr_scheduler.ReduceLROnPlateau(classical_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    classical_model_path = 'best_classical_deepfake_model.pth' # 纯经典模型保存路径

    print("\n--- 开始训练 纯经典模型 ---")
    train_model(classical_model, train_loader, validation_loader, criterion, classical_optimizer, 
                num_epochs=num_epochs, device=device, scheduler=classical_scheduler, 
                model_save_path=classical_model_path)

    print(f"\n--- 加载最佳纯经典模型: {classical_model_path} ---")
    classical_model.load_state_dict(torch.load(classical_model_path))
    classical_model.eval() # 评估模式

    print("\n--- 在验证集上评估 纯经典模型 ---")
    evaluate_model(classical_model, validation_loader, device=device)
    
    print("\n--- 在测试集上评估 纯经典模型 ---")
    evaluate_model(classical_model, test_loader, device=device)

if __name__ == '__main__':
    main()