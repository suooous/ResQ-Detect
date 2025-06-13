# ResQ-Detect
ResQ-Detect：基于ResNet和量子神经网络的深度伪造检测模型

# 环境搭建
> 具体的环境的搭建的步骤，请参照CSDN博客[量子计算导论课程设计 之 PennyLane环境搭建](https://guojiewang.blog.csdn.net/article/details/148519398?spm=1001.2014.3001.5502)

# 项目架构说明
**ResQ-Detect**
* `ResQ-Detect`模型在这个`CNN_change2_savemodule.py`文件
* 运行命令
```bash
python CNN_change2_savemodule.py
```
> 请注意里面的参数设置，可根据自身的算力大小选择合适的`epoch`和`train`、`test`、`validation`的大小

**ResNet-18(用于作为经典的对比)**
* `ResNet-18`位于`CNN_DD.py`文件
```bash
python CNN_DD.py
```
> 对比的时候，注意和这个`ResQ-Detect`保持一致

**量子电路的可视化**
* 有两个文件，`vision.py`和`vision1.py`分别对应这个量子电路的层级和门级的可视化
```bash
python vision.py
```
```bash
python vision1.py
```

**敏感性测试**
* 在这个`sensitivity_test_save_vision.py`，运行之后，会给出`类别权重`、`学习率`、`优化器`的敏感性测试的结果和过程中出现的最好的模型，以及可视化结果
```bash
python sensitivity_test_save_vision.py
```
