import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pennylane as qml
import numpy as np
import os
from PIL import Image
import random
from sklearn.metrics import precision_score, recall_score, f1_score

# 加载数据
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        folder_path = os.path.join(root_dir, phase)
        real_images = []
        fake_images = []

        for label_name in ['fake', 'real']:
            label_folder = os.path.join(folder_path, label_name)
            images = [os.path.join(label_folder, img) for img in os.listdir(label_folder) if img.endswith('.png')]
            if label_name == 'real':
                real_images.extend(images)
            else:
                fake_images.extend(images)
        
        # 确保每个类别的限制是针对该类别的，并且如果一个类别数量不足，就取全部
        if limit:
            # 确保每个类别的图片数量不超过limit的一半，如果实际数量小于limit/2，则取实际数量
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

# 结合经典CNN（ResNet-18）和量子神经网络
class QE_VTDD(nn.Module):
    def __init__(self, num_qubits=4, quantum_depth=3, freeze_cnn=False):
        super(QE_VTDD, self).__init__()
        
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*(list(self.cnn.children())[:-1])) 

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval() # 评估模式，禁用 dropout/batchnorm 更新

        self.num_qubits = num_qubits
        self.quantum_depth = quantum_depth
        
        # 改进 fc_reduce：增加一个中间层和 Tanh 激活函数
        self.fc_reduce = nn.Sequential(
            nn.Linear(512, 64), # 512维特征 -> 64维
            nn.Tanh(),          # 使用 Tanh 激活函数，输出范围在 [-1, 1]
            nn.Linear(64, num_qubits) # 64维 -> num_qubits 维
        )

        # # 再改进一下网络
        # self.fc_reduce = nn.Sequential(
        #     nn.Linear(512,128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128,64),
        #     nn.Tanh(),
        #     nn.Linear(64,num_qubits)
        # )
        # 改进一下，不能那么激进
        # self.fc_reduce = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
              
        #     nn.Linear(256,64),
        #     nn.Tanh(),
        #     nn.Linear(64,num_qubits) 
        # )
        self.qnode = self.create_quantum_node()
        self.fc = nn.Linear(num_qubits, 2)

    def create_quantum_node(self):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            # AngleEmbedding 期望输入为角度（弧度），Tanh的输出 [-1, 1] 可以直接作为输入
            # 如果需要将其映射到 [0, 2pi]，可以加 (inputs + 1) * np.pi
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits)) 
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        weight_shapes = {"weights": (self.quantum_depth, self.num_qubits, 3)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # 确保 CNN 在训练模式下根据 self.training 状态工作
        # 如果 self.cnn.training 是 False (例如在 model.eval() 模式下)，
        # 那么就使用 torch.no_grad() 来确保不计算梯度
        if not self.training and self.cnn.training: # 如果模型是 eval 模式但 CNN 还在 train 模式
             # CNN 部分在初始化时已经设置为 eval，但保险起见，这里是防止误用
             # 实际训练时，model.train() 和 model.eval() 会自动设置所有子模块
             pass # 不需要额外处理，因为 model.eval() 会设置所有子模块为 eval 模式

        x = self.cnn(x) # CNN 保持其自身的 train/eval 状态
            
        x = x.view(x.size(0), -1)
        x = self.fc_reduce(x) # 经过 fc_reduce (包含 Tanh)
        x = self.qnode(x)
        x = self.fc(x)
        return x


### 改进后的 `train_model` 函数 (关键更新)


def train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs=10, device=None, scheduler=None, model_save_path='best_model.pth'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"训练数据加载器批次数量: {len(train_dataloader)}")
    print(f"验证数据加载器批次数量: {len(validation_dataloader)}")

    # 用于保存最佳模型
    best_val_f1 = -1.0 # 初始 F1 分数设为负数，确保第一次肯定能保存
    
    for epoch in range(num_epochs):
        print(f'\nStarting Epoch {epoch + 1}/{num_epochs}')
        
        # --- 训练阶段 ---
        model.train() # 设置模型为训练模式
        total_train_loss = 0.0
        total_train_samples = 0
        
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)
            optimizer.zero_grad() # 清除之前的梯度
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新模型参数
            
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size
        
        epoch_avg_train_loss = total_train_loss / total_train_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}] Training Average Loss: {epoch_avg_train_loss:.4f}')

        # --- 验证阶段 ---
        model.eval() # 设置模型为评估模式，这将禁用 dropout 等层
        total_val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad(): # 在评估阶段不计算梯度
            for images, labels in validation_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels) # 计算验证集上的损失
                total_val_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())
        
        epoch_avg_val_loss = total_val_loss / len(validation_dataloader.dataset) # 注意这里是 len(dataset)
        val_accuracy = 100 * np.sum(np.array(all_val_predictions) == np.array(all_val_labels)) / len(validation_dataloader.dataset)
        val_precision = precision_score(all_val_labels, all_val_predictions, average='binary', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_predictions, average='binary', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='binary', zero_division=0)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Average Loss: {epoch_avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.4f}')

        # 步进学习率调度器，传入验证损失
        if scheduler:
            scheduler.step(epoch_avg_val_loss) 

        # 保存最佳模型：如果当前验证 F1 分数更高，则保存模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f'Saving best model with F1 score: {best_val_f1:.4f} to {model_save_path}')

    print('Finished Training')

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
    print(f'Precision: {precision:.4f}') # 格式化输出
    print(f'Recall: {recall:.4f}')      # 格式化输出
    print(f'F1 Score: {f1:.4f}')        # 格式化输出
    return accuracy, precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = 'faceforensec++\\1000_videos' 
    # 增加训练数据量，并确保每个类别的图片数量平衡
    # 保持 limit=600 用于训练，limit=100 用于验证和测试，以快速测试
    train_dataset = DeepfakeDataset(root_dir=root_dir, phase='train', transform=transform, limit=600)
    validation_dataset = DeepfakeDataset(root_dir=root_dir, phase='validation', transform=transform, limit=100)
    test_dataset = DeepfakeDataset(root_dir=root_dir, phase='test', transform=transform, limit=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"训练数据加载器长度: {len(train_loader)}")
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    print(f"验证数据加载器长度: {len(validation_loader)}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试数据加载器长度: {len(test_loader)}")

    # 现在将 freeze_cnn 设置为 False，以便微调 CNN
    model = QE_VTDD(num_qubits=6, quantum_depth=5, freeze_cnn=False) # 保持 num_qubits 和 quantum_depth
    
    # 类别权重保持 [1.0, 1.0]，先观察效果
    class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 降低学习率，因为 CNN 正在微调
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 降低学习率
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 定义保存最佳模型的路径
    best_model_path = 'best_deepfake_model.pth'

    print("\n--- 开始训练 ---")
    # 将 validation_loader 和 model_save_path 传递给 train_model
    train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=30, device=device, scheduler=scheduler, model_save_path=best_model_path)
    
    # --- 加载最佳模型进行评估 ---
    print(f"\n--- 加载在训练中表现最佳的模型: {best_model_path} ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval() # 确保模型处于评估模式

    print("\n--- 在验证集上评估 (使用最佳模型) ---")
    evaluate_model(model, validation_loader, device=device)
    
    print("\n--- 在测试集上评估 (使用最佳模型) ---")
    evaluate_model(model, test_loader, device=device)

if __name__ == '__main__':
    main()