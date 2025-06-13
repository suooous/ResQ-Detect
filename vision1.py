import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# 这是一个示例的QE_VTDD模型，只包含与量子电路相关和可视化所需的部分。
class QE_VTDD(nn.Module):
    def __init__(self, num_qubits=4, quantum_depth=3):
        super(QE_VTDD, self).__init__()
        
        self.num_qubits = num_qubits
        self.quantum_depth = quantum_depth
        
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
            量子电路的定义，现在展开为基本门操作。
            - AngleEmbedding 的展开：使用单比特旋转门。
            - StronglyEntanglingLayers 的展开：使用单比特旋转门和CNOT门。
            """
            # --- AngleEmbedding 的展开 ---
            # 默认的AngleEmbedding使用RY门。
            # 如果输入维度与量子比特数匹配，通常每个比特应用一个门。
            # inputs 的形状是 (num_qubits,)，对应每个量子比特的旋转角度
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            
            # --- StronglyEntanglingLayers 的展开 ---
            # StronglyEntanglingLayers 默认包含旋转门和CNOT纠缠门。
            # 它会重复 quantum_depth 次这个模式。
            # weights 的形状是 (quantum_depth, num_qubits, 3)
            # 每个 num_qubits 有3个旋转参数 (Rx, Ry, Rz)
            for d in range(self.quantum_depth):
                # 单比特旋转门
                for i in range(self.num_qubits):
                    # weights[d, i, 0] 是 Rx 的参数
                    # weights[d, i, 1] 是 Ry 的参数
                    # weights[d, i, 2] 是 Rz 的参数
                    qml.Rot(weights[d, i, 0], weights[d, i, 1], weights[d, i, 2], wires=i)
                
                # 线性纠缠（CNOT门）
                # PennyLane的StronglyEntanglingLayers通常采用循环或线性连接
                # 这里我们模拟一个简单的线性连接，你可以根据需要调整
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.num_qubits]) # 循环连接，确保所有比特都参与纠缠

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # --- 量子电路可视化部分 ---
        print("\n--- 正在生成量子电路图 (显示基本门) ---")
        dummy_inputs = torch.rand(self.num_qubits)
        dummy_weights = torch.rand(self.quantum_depth, self.num_qubits, 3)

        fig, ax = qml.draw_mpl(circuit)(inputs=dummy_inputs, weights=dummy_weights)
        ax.set_title("量子电路架构示意图 (展开基本门)") # 设置图表标题
        # plt.show() # 显示电路图
        plt.savefig("quantum_circuit_basic_gates.png")
        # 如果你想将图表保存为文件，可以取消注释下面这行：
        # fig.savefig("quantum_circuit_basic_gates.png")
        print("--- 量子电路图已显示 ---")
        # --- 量子电路可视化部分结束 ---

        # 调整 weight_shapes 以匹配展开后的权重参数
        # AngleEmbedding 的输入由 dummy_inputs 提供，不是权重。
        # StronglyEntanglingLayers 的 weights 结构保持不变。
        weight_shapes = {"weights": (self.quantum_depth, self.num_qubits, 3)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # 这里的forward方法简化了，因为我们只关注量子电路部分
        x = self.fc_reduce(x) 
        x = self.qnode(x)
        x = self.fc(x)
        return x

def main():
    print("初始化 QE_VTDD 模型以展示量子电路图...")
    # 更改为 num_qubits=4, quantum_depth=3 以更好地匹配你提供的第一张图的复杂性
    model = QE_VTDD(num_qubits=6, quantum_depth=5) 
    
    print("\n模型初始化完成。量子电路图已自动生成并显示。")
    print("请关闭弹出的电路图窗口以继续程序。")

if __name__ == '__main__':
    main()