import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
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

# 结合经典ViT和量子神经网络
class QE_VTDD(nn.Module):
    def __init__(self, num_qubits=4, quantum_depth=3):
        super(QE_VTDD, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit.eval()  # Freeze ViT parameters
        self.num_qubits = num_qubits
        self.quantum_depth = quantum_depth
        self.fc_reduce = nn.Linear(768, num_qubits)
        self.qnode = self.create_quantum_node()
        self.fc = nn.Linear(num_qubits, 2)  # Binary classification

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
        with torch.no_grad():
            x = self.vit(x).last_hidden_state[:, 0, :]  # Extract CLS token
        x = self.fc_reduce(x)
        x = nn.ReLU()(x)  # 添加 ReLU 激活函数
        x = self.qnode(x)
        x = self.fc(x)
        return x

# def train_model(model, dataloader, criterion, optimizer, num_epochs=1):
#     print(len(dataloader))
#     for epoch in range(num_epochs):
#         print(f'Starting Epoch {epoch + 1}/{num_epochs}')  # Log start of epoch
#         running_loss = 0.0
#         for i, (images, labels) in enumerate(dataloader):
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#         # 打印每个epoch的平均损失
#         epoch_loss = running_loss / len(dataloader)
#         print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

#     print('Finished Training')

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    print(len(dataloader))
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0.0
        total_samples = 0
        
        for i, (images, labels) in enumerate(dataloader):
            batch_size = labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size  # 累加批次总损失
            total_samples += batch_size  # 累加总样本数
        
        # 计算正确的平均损失：总损失 / 总样本数
        epoch_avg_loss = total_loss / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {epoch_avg_loss:.4f}')
    
    print('Finished Training')

def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in dataloader:
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
    print(f'F1 Score: {f1}')
    return accuracy, precision, recall, f1

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 设置数据集路径
    train_dataset = DeepfakeDataset(root_dir='faceforensec++\\1000_videos', phase='train', transform=transform, limit=200)
    validation_dataset = DeepfakeDataset(root_dir='faceforensec++\\1000_videos', phase='validation', transform=transform, limit=100)
    test_dataset = DeepfakeDataset(root_dir='faceforensec++\\1000_videos', phase='test', transform=transform, limit=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(len(train_loader))
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    print(len(validation_loader))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(len(test_loader))

    model = QE_VTDD()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer)
    evaluate_model(model, validation_loader)
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main() 