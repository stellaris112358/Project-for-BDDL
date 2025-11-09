# Project-for-BDDL项目：conditional generated picture complement based on ddpm


本项目实现了一个基于条件降噪扩散概率模型（Conditional DDPM）的系统，用于对CIFAR-10图像进行图像补全（Inpainting）。

模型的功能是根据图像中心 10x10 像素的已知区域，生成并补全图像外部 32x32 的其余未知部分。


## 项目结构
主要代码文件在conditional_generated_picture_complement文件夹下\
conditional_generated_picture_complement/
├── config/
│ └── config.yaml # 配置文件（模型参数、训练参数等）
├── models/
│ └── unet_cond.py # 条件 U-Net 模型定义
├── scripts/
│ ├── train_conditional_ddpm.py # 模型训练脚本
│ ├── gen_image_dataset.py # 生成训练数据集（CIFAR-10 转 npy）
│ ├── gen_image_testset.py # 生成测试数据集
│ ├── visualize_image_sample.py # 可视化补全结果
│ └── gen_test_sets.py # 生成带掩码的测试集和补全后的测试集
├── data/ # 数据存放目录（自动生成）
├── models_image/ # 模型保存目录（自动生成）
└── outputs_image/ # 可视化结果保存目录（自动生成）



## 使用方法 (Quick Start)

**前提:**
1.  安装所需的 Python 库 (如 `torch`, `numpy`, `pyyaml`, `matplotlib`, `pillow`)。
2.  下载 CIFAR-10 Python 版本 (cifar-10-python.tar.gz)，解压并确保 `cifar-10-batches-py` 目录位于项目根目录。
数据集链接https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz



**步骤:**
1.  **预处理训练数据:**
    ```bash
    python3 scripts/gen_image_dataset.py
    ```
    *(这会根据 `config.yaml` 在 `data/image_batches/` 目录下生成 .npy 文件)*

2.  **开始训练:**
    ```bash
    python3 scripts/train_conditional_ddpm.py
    ```
    *(模型将开始训练，最佳模型将保存到 `models_image/ddpm_cond_best.pth`)*

3.  **可视化生成结果:**
    ```bash
    python3 scripts/visualize_image_sample.py
    ```
    *(这会加载训练好的模型，生成一张补全示例图并保存到 `outputs_image/` 目录)*
