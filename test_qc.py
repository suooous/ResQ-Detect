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

        if phase == 'train':
            fake_folder = os.path.join(root_dir, 'fake_png')
            real_folder = os.path.join(root_dir, 'real_png')
            fake_images = [os.path.join(fake_folder, img) for img in os.listdir(fake_folder) if img.endswith('.png')]
            real_images = [os.path.join(real_folder, img) for img in os.listdir(real_folder) if img.endswith('.png')]
            # Select 200 fake and 200 real images
            self.image_paths.extend(fake_images[:500])
            self.labels.extend([1] * 500)
            self.image_paths.extend(real_images[:500])
            self.labels.extend([0] * 500)
        elif phase == 'test':
            fake_folder = os.path.join(root_dir, 'fake_png')
            real_folder = os.path.join(root_dir, 'real_png')
            fake_images = [os.path.join(fake_folder, img) for img in os.listdir(fake_folder) if img.endswith('.png')]
            real_images = [os.path.join(real_folder, img) for img in os.listdir(real_folder) if img.endswith('.png')]
            # Select 50 fake and 50 real images
            self.image_paths.extend(fake_images[500:700])
            self.labels.extend([1] * 200)
            self.image_paths.extend(real_images[500:700])
            self.labels.extend([0] * 200)

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
        x = self.qnode(x)
        x = self.fc(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=4):
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}/{num_epochs}')  # Log start of epoch
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        print(f'Finished Epoch {epoch + 1}/{num_epochs}')  # Log end of epoch

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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 设置数据集路径
    train_dataset = DeepfakeDataset(root_dir='QC_datasets', phase='train', transform=transform)
    test_dataset = DeepfakeDataset(root_dir='QC_datasets', phase='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = QE_VTDD()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer)
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main() 