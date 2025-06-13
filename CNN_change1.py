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
            self.cnn.eval()

        self.num_qubits = num_qubits
        self.quantum_depth = quantum_depth
        
        # 改进 fc_reduce：增加一个中间层和 Tanh 激活函数
        self.fc_reduce = nn.Sequential(
            nn.Linear(512, 64), # 512维特征 -> 64维
            nn.Tanh(),          # 使用 Tanh 激活函数，输出范围在 [-1, 1]
            nn.Linear(64, num_qubits) # 64维 -> num_qubits 维
        )
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
        if not self.cnn.training:
             with torch.no_grad():
                x = self.cnn(x)
        else:
            x = self.cnn(x)
            
        x = x.view(x.size(0), -1)
        x = self.fc_reduce(x) # 经过 fc_reduce (包含 Tanh)
        # 不再添加额外的 ReLU 或缩放，让 QNN 接收 Tanh 后的输出
        x = self.qnode(x)
        x = self.fc(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device=None, scheduler=None): # 添加 scheduler 参数
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"训练数据加载器批次数量: {len(dataloader)}")
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0.0
        total_samples = 0
        
        model.train() 

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        epoch_avg_loss = total_loss / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {epoch_avg_loss:.4f}')

        # 在每个epoch结束后调用调度器，传入训练损失作为监控指标（如果使用 ReduceLROnPlateau）
        if scheduler:
            scheduler.step(epoch_avg_loss) # 理想情况下这里应该传入验证损失

    print('Finished Training')

def evaluate_model(model, dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval() 

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
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0) # 添加 zero_division
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0) # 添加 zero_division
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0) # 添加 zero_division
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
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
    train_dataset = DeepfakeDataset(root_dir=root_dir, phase='train', transform=transform, limit=600) # 尝试增加到600 (每个类别300)
    validation_dataset = DeepfakeDataset(root_dir=root_dir, phase='validation', transform=transform, limit=100) # 相应增加
    test_dataset = DeepfakeDataset(root_dir=root_dir, phase='test', transform=transform, limit=100) # 相应增加

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"训练数据加载器长度: {len(train_loader)}")
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    print(f"验证数据加载器长度: {len(validation_loader)}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试数据加载器长度: {len(test_loader)}")

    # 冻结 CNN 参数 (freeze_cnn=True) 是一个好的起点，如果性能仍不好，再尝试微调 (freeze_cnn=False)
    # 我在这里把它改回 True，让你先测试冻结 CNN 的性能
    model = QE_VTDD(num_qubits=6, quantum_depth=5, freeze_cnn=False) # 尝试增加 num_qubits 和 quantum_depth
    
    # 根据之前评估结果（高精确率，低召回率），调整损失函数权重
    # 假设标签 0 是真实，标签 1 是伪造。模型倾向于预测 0。
    # 增加标签 1 (伪造) 的权重，以惩罚模型错误地预测真实图片。
    # 需要根据实际数据集的类别比例来调整，这里只是一个示例
    # 假设真实:伪造比例为 1:1，但模型偏向真实，可以给伪造更大的权重
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    print("\n--- 开始训练 ---")
    # 将 scheduler 传递给 train_model
    train_model(model, train_loader, criterion, optimizer, num_epochs=50, device=device, scheduler=scheduler) # 增加 Epoch 数量
    
    print("\n--- 在验证集上评估 ---")
    evaluate_model(model, validation_loader, device=device)
    
    print("\n--- 在测试集上评估 ---")
    evaluate_model(model, test_loader, device=device)

if __name__ == '__main__':
    main()