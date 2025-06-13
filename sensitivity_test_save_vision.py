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
import logging
import matplotlib.pyplot as plt

# Set up logging for better tracking of experiments
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            if not os.path.exists(label_folder):
                logging.warning(f"Folder not found: {label_folder}. Skipping.")
                continue
            images = [os.path.join(label_folder, img) for img in os.listdir(label_folder) if img.endswith('.png')]
            if label_name == 'real':
                real_images.extend(images)
            else:
                fake_images.extend(images)
        
        if limit:
            real_images = real_images[:min(len(real_images), limit // 2)]
            fake_images = fake_images[:min(len(fake_images), limit // 2)]
        
        self.image_paths.extend(real_images)
        self.labels.extend([0] * len(real_images)) # 真实图片标签为 0

        self.image_paths.extend(fake_images)
        self.labels.extend([1] * len(fake_images)) # 伪造图片标签为 1

        logging.info(f"Dataset phase: {phase}, Real images: {len(real_images)}, Fake images: {len(fake_images)}")

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
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits)) 
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        weight_shapes = {"weights": (self.quantum_depth, self.num_qubits, 3)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        x = self.cnn(x) 
            
        x = x.view(x.size(0), -1)
        x = self.fc_reduce(x) # 经过 fc_reduce (包含 Tanh)
        x = self.qnode(x)
        x = self.fc(x)
        return x

def train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs=10, device=None, scheduler=None, model_save_path='best_model.pth'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info(f"训练数据加载器批次数量: {len(train_dataloader)}")
    logging.info(f"验证数据加载器批次数量: {len(validation_dataloader)}")

    best_val_f1 = -1.0 
    
    for epoch in range(num_epochs):
        logging.info(f'\nStarting Epoch {epoch + 1}/{num_epochs}')
        
        # --- 训练阶段 ---
        model.train() 
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
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}] Training Average Loss: {epoch_avg_train_loss:.4f}')

        # --- 验证阶段 ---
        model.eval() 
        total_val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad(): 
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

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}] Validation Average Loss: {epoch_avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.4f}')

        if scheduler:
            scheduler.step(epoch_avg_val_loss) 

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            logging.info(f'Saving best model with F1 score: {best_val_f1:.4f} to {model_save_path}')

    logging.info('Finished Training')

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
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    logging.info(f'Accuracy: {accuracy}%')
    logging.info(f'Precision: {precision:.4f}') 
    logging.info(f'Recall: {recall:.4f}') 
    logging.info(f'F1 Score: {f1:.4f}') 
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def run_experiment(
    exp_name, 
    root_dir, 
    transform, 
    num_qubits, 
    quantum_depth, 
    freeze_cnn, 
    class_weights_val, 
    learning_rate, 
    optimizer_type, 
    num_epochs, 
    device
):
    logging.info(f"\n--- Running Experiment: {exp_name} ---")

    train_dataset = DeepfakeDataset(root_dir=root_dir, phase='train', transform=transform, limit=600)
    validation_dataset = DeepfakeDataset(root_dir=root_dir, phase='validation', transform=transform, limit=100)
    test_dataset = DeepfakeDataset(root_dir=root_dir, phase='test', transform=transform, limit=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = QE_VTDD(num_qubits=num_qubits, quantum_depth=quantum_depth, freeze_cnn=freeze_cnn)
    
    class_weights_tensor = torch.tensor(class_weights_val, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_model_path = f'best_model_{exp_name}.pth'

    train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=num_epochs, device=device, scheduler=scheduler, model_save_path=best_model_path)
    
    logging.info(f"\n--- Evaluating Best Model for {exp_name} ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval() 

    logging.info(f"\n--- Results for {exp_name} on Validation Set ---")
    val_metrics = evaluate_model(model, validation_loader, device=device)
    logging.info(f"\n--- Results for {exp_name} on Test Set ---")
    test_metrics = evaluate_model(model, test_loader, device=device)

    return val_metrics, test_metrics

def save_results_to_txt(all_results, filename="sensitivity_test_results.txt"):
    with open(filename, 'w') as f:
        for exp_type, results in all_results.items():
            f.write(f"=== {exp_type} ===\n")
            for config, metrics in results.items():
                f.write(f"  Configuration: {config}\n")
                f.write(f"    Validation Metrics: Accuracy={metrics['val']['accuracy']:.2f}%, Precision={metrics['val']['precision']:.4f}, Recall={metrics['val']['recall']:.4f}, F1={metrics['val']['f1']:.4f}\n")
                f.write(f"    Test Metrics: Accuracy={metrics['test']['accuracy']:.2f}%, Precision={metrics['test']['precision']:.4f}, Recall={metrics['test']['recall']:.4f}, F1={metrics['test']['f1']:.4f}\n")
            f.write("\n")
    logging.info(f"All results saved to {filename}")

def plot_sensitivity_results(results, title_prefix, x_label, filename_prefix, metric='f1'):
    plt.figure(figsize=(10, 6))
    
    configs = []
    val_scores = []
    test_scores = []

    for config, metrics in results.items():
        configs.append(str(config)) # Convert list/float to string for x-axis
        val_scores.append(metrics['val'][metric])
        test_scores.append(metrics['test'][metric])

    x = np.arange(len(configs))
    width = 0.35

    plt.bar(x - width/2, val_scores, width, label=f'Validation {metric.upper()} Score')
    plt.bar(x + width/2, test_scores, width, label=f'Test {metric.upper()} Score')

    plt.xlabel(x_label)
    plt.ylabel(f'{metric.upper()} Score')
    plt.title(f'{title_prefix} - {metric.upper()} Score Sensitivity')
    plt.xticks(x, configs)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_{metric}_sensitivity.png')
    plt.close()
    logging.info(f"Plot saved to {filename_prefix}_{metric}_sensitivity.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = 'faceforensec++\\1000_videos' 
    num_epochs = 10 # Reduced epochs for quicker testing during development. Increase for final runs.

    all_experiment_results = {
        'Class Weights Sensitivity': {},
        'Learning Rate Sensitivity': {},
        'Optimizer Sensitivity': {}
    }

    # --- Sensitive Test for Class Weights ---
    logging.info("\n--- 开始类别权重敏感性测试 ---")
    class_weights_configs = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    for i, weights in enumerate(class_weights_configs):
        exp_name = f"ClassWeights_Exp_{i+1}_Weights_{weights[0]}_{weights[1]}"
        val_metrics, test_metrics = run_experiment(
            exp_name=exp_name,
            root_dir=root_dir,
            transform=transform,
            num_qubits=6,
            quantum_depth=5,
            freeze_cnn=False,
            class_weights_val=weights,
            learning_rate=0.001, 
            optimizer_type='Adam', 
            num_epochs=num_epochs,
            device=device
        )
        all_experiment_results['Class Weights Sensitivity'][str(weights)] = {'val': val_metrics, 'test': test_metrics}

    # # --- Sensitive Test for Learning Rate ---
    # logging.info("\n--- 开始学习率敏感性测试 ---")
    # learning_rate_configs = [0.05, 0.001, 0.005]
    # for i, lr in enumerate(learning_rate_configs):
    #     exp_name = f"LearningRate_Exp_{i+1}_LR_{lr}"
    #     val_metrics, test_metrics = run_experiment(
    #         exp_name=exp_name,
    #         root_dir=root_dir,
    #         transform=transform,
    #         num_qubits=6,
    #         quantum_depth=5,
    #         freeze_cnn=False,
    #         class_weights_val=[1.0, 1.0], 
    #         learning_rate=lr,
    #         optimizer_type='Adam', 
    #         num_epochs=num_epochs,
    #         device=device
    #     )
    #     all_experiment_results['Learning Rate Sensitivity'][str(lr)] = {'val': val_metrics, 'test': test_metrics}

    # # --- Sensitive Test for Optimizer ---
    # logging.info("\n--- 开始优化器敏感性测试 ---")
    # optimizer_configs = ['SGD', 'Adam', 'RMSprop']
    # common_lr_for_optimizers = 0.001 
    # for i, opt_type in enumerate(optimizer_configs):
    #     exp_name = f"Optimizer_Exp_{i+1}_Type_{opt_type}"
    #     val_metrics, test_metrics = run_experiment(
    #         exp_name=exp_name,
    #         root_dir=root_dir,
    #         transform=transform,
    #         num_qubits=6,
    #         quantum_depth=5,
    #         freeze_cnn=False,
    #         class_weights_val=[1.0, 1.0], 
    #         learning_rate=common_lr_for_optimizers,
    #         optimizer_type=opt_type,
    #         num_epochs=num_epochs,
    #         device=device
    #     )
    #     all_experiment_results['Optimizer Sensitivity'][opt_type] = {'val': val_metrics, 'test': test_metrics}

    logging.info("\n--- 所有敏感性测试完成 ---")

    # Save all results to a text file
    save_results_to_txt(all_experiment_results)

    # Plotting the results
    plot_sensitivity_results(
        all_experiment_results['Class Weights Sensitivity'], 
        "Class Weights", "Class Weight Configuration (Real:Fake)", "class_weights_sensitivity"
    )
    # plot_sensitivity_results(
    #     all_experiment_results['Learning Rate Sensitivity'], 
    #     "Learning Rate", "Learning Rate", "learning_rate_sensitivity"
    # )
    # plot_sensitivity_results(
    #     all_experiment_results['Optimizer Sensitivity'], 
    #     "Optimizer", "Optimizer Type", "optimizer_sensitivity"
    # )
    
    logging.info("\n--- 所有结果已保存并绘图 ---")

if __name__ == '__main__':
    main()