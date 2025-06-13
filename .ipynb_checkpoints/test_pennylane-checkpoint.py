import pennylane as qml
import numpy as np

# 创建一个模拟器设备
dev = qml.device('default.qubit', wires=1)

# 定义一个简单的量子电路
@qml.qnode(dev)
def quantum_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# 测试电路
params = np.array([0.54, 0.12])
print(f"回路输出: {quantum_circuit(params)}")
print("PennyLane 安装成功!")