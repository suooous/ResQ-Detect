import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models # 导入 models
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
        for label in ['fake', 'real']:
            label_folder = os.path.join(folder_path, label)
            images = [os.path.join(label_folder, img) for img in os.listdir(label_folder) if img.endswith('.png')]
            if limit:
                images = images[:limit]  # 每个类别截断一半的limit
            self.image_paths.extend(images)
            self.labels.extend([0 if label == 'real' else 1] * len(images))

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
class QE_VTDD(nn.Module): # 保持类名不变，但内部逻辑已改变
    def __init__(self, num_qubits=4, quantum_depth=3, freeze_cnn=False): # 添加 freeze_cnn 参数
        super(QE_VTDD, self).__init__()
        
        # 替换 ViT 为 ResNet-18
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 移除 ResNet 最后的分类层，保留特征提取部分
        self.cnn = nn.Sequential(*(list(self.cnn.children())[:-1])) 

        if freeze_cnn:
            # 冻结CNN参数
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval() # 设置为评估模式，禁用 dropout/batchnorm 更新

        self.num_qubits = num_qubits
        self.quantum_depth = quantum_depth
        
        # ResNet-18 的输出特征是 512 维（在最后的平均池化之后），所以 fc_reduce 需要从 512 维输入
        self.fc_reduce = nn.Linear(512, num_qubits) 
        self.qnode = self.create_quantum_node()
        self.fc = nn.Linear(num_qubits, 2)  # 二分类

    def create_quantum_node(self):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        weight_shapes = {"weights": (self.quantum_depth, self.num_qubits, 3)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # 冻结时使用 torch.no_grad()
        if not self.cnn.training: # 如果CNN被冻结（处于eval模式）
             with torch.no_grad():
                x = self.cnn(x)
        else: # 如果CNN不冻结
            x = self.cnn(x)
            
        x = x.view(x.size(0), -1) # 展平特征，从 (batch_size, 512, 1, 1) 变为 (batch_size, 512)
        x = self.fc_reduce(x)
        # 移除 ReLU，因为它可能不适合 AngleEmbedding 的输入范围
        # x = nn.ReLU()(x) 
        x = self.qnode(x)
        x = self.fc(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # 将模型移到指定设备

    print(f"训练数据加载器批次数量: {len(dataloader)}")
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0.0
        total_samples = 0
        
        # 设置模型为训练模式
        model.train() 

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device) # 将数据移到指定设备
            labels = labels.to(device) # 将标签移到指定设备

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
    
    print('Finished Training')

def evaluate_model(model, dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # 将模型移到指定设备
    
    # 设置模型为评估模式
    model.eval() 

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device) # 将数据移到指定设备
            labels = labels.to(device) # 将标签移到指定设备

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    f1 = f1_score(all_labels, all_predictions, average='binary') # 确保这里再次计算了F1
    print(f'F1 Score: {f1}')
    return accuracy, precision, recall, f1

def main():
    # 检测并设置设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet 使用 ImageNet 的均值和标准差
    ])

    # 设置数据集路径
    root_dir = 'faceforensec++\\1000_videos' 
    train_dataset = DeepfakeDataset(root_dir=root_dir, phase='train', transform=transform, limit=300)
    validation_dataset = DeepfakeDataset(root_dir=root_dir, phase='validation', transform=transform, limit=100)
    test_dataset = DeepfakeDataset(root_dir=root_dir, phase='test', transform=transform, limit=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"训练数据加载器长度: {len(train_loader)}")
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    print(f"验证数据加载器长度: {len(validation_loader)}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试数据加载器长度: {len(test_loader)}")

    # 默认冻结 CNN 参数 (freeze_cnn=True)
    model = QE_VTDD(freeze_cnn=True) # 你也可以尝试 freeze_cnn=False 来微调CNN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- 开始训练 ---")
    train_model(model, train_loader, criterion, optimizer, device=device)
    
    print("\n--- 在验证集上评估 ---")
    evaluate_model(model, validation_loader, device=device)
    
    print("\n--- 在测试集上评估 ---")
    evaluate_model(model, test_loader, device=device)

if __name__ == '__main__':
    main()