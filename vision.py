import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# 这是一个示例的QE_VTDD模型，只包含与量子电路相关和可视化所需的部分。
# 实际模型中的其他部分（如CNN、数据加载等）已被省略，因为它们不是可视化量子电路所必需的。
class QE_VTDD(nn.Module):
    def __init__(self, num_qubits=4, quantum_depth=3):
        super(QE_VTDD, self).__init__()
        
        self.num_qubits = num_qubits
        self.quantum_depth = quantum_depth
        
        # 这里的fc_reduce只是为了模拟真实模型中的输入维度
        # 实际模型中，它的输入是CNN的输出特征
        self.fc_reduce = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, num_qubits) 
        )
        self.qnode = self.create_quantum_node()
        self.fc = nn.Linear(num_qubits, 2)

    def create_quantum_node(self):
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            """
            量子电路的定义。
            - AngleEmbedding 将经典输入编码为量子态。
            - StronglyEntanglingLayers 是一个常用的参数化量子电路层，用于学习特征。
            """
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits)) 
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # --- 量子电路可视化部分 ---
        print("\n--- 正在生成量子电路图 ---")
        # 为了绘制电路图，我们需要提供一些示例输入和权重。
        # 这些虚拟值只用于描绘电路结构，不影响电路的实际计算。
        dummy_inputs = torch.rand(self.num_qubits)
        dummy_weights = torch.rand(self.quantum_depth, self.num_qubits, 3)

        fig, ax = qml.draw_mpl(circuit)(inputs=dummy_inputs, weights=dummy_weights)
        ax.set_title("量子电路架构示意图") # 设置图表标题
        # plt.show() # 显示电路图
        plt.savefig("quantum_circuit_all.png")
        # 如果你想将图表保存为文件，可以取消注释下面这行：
        # fig.savefig("quantum_circuit.png")
        print("--- 量子电路图已显示 ---")
        # --- 量子电路可视化部分结束 ---

        weight_shapes = {"weights": (self.quantum_depth, self.num_qubits, 3)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # 这里的forward方法简化了，因为我们只关注量子电路部分
        # 实际应用中，x会是CNN的输出
        x = self.fc_reduce(x) 
        x = self.qnode(x)
        x = self.fc(x)
        return x

def main():
    print("初始化 QE_VTDD 模型以展示量子电路图...")
    # 你可以根据需要调整 num_qubits 和 quantum_depth
    # 例如：num_qubits=4, quantum_depth=3 是一个较小的示例
    # 原代码中使用的 num_qubits=6, quantum_depth=5 也会绘制出来
    model = QE_VTDD(num_qubits=6, quantum_depth=5) 
    
    print("\n模型初始化完成。量子电路图已自动生成并显示。")
    print("请关闭弹出的电路图窗口以继续程序。")

    # 为了让程序能够完整运行，这里添加一个简单的forward调用，但它不会实际训练模型
    # model(torch.rand(1, 512)) # 假设fc_reduce的输入维度是512

if __name__ == '__main__':
    main()