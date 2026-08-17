# 仓库文件说明（3D Reconstruction via Reconstructed Neural Networks with Pigeon-inspired Optimization Algorithm）

本仓库为论文《重组神经网络的三维重建——固定权重与调整输入》的官方代码与数据开源仓库。

## 环境配置与依赖
- Python 3.8+
- 核心依赖库：`numpy`, `torch`, `opencv-python`, `pandas`, `matplotlib`, `scikit-learn`
- 完整依赖见 `requirements.txt`
- 安装命令：`pip install -r requirements.txt`

---

## 1. 本文核心方法程序
**`Three_Dimensional_Reconstruction_2025_09_29_RL.py`**  
对应论文第3.3节及第4章的核心三维重建程序（神经网络重组 + 固定权重 + 优化输入）。  
【权重来源】网络权重直接以数组形式硬编码在脚本中（对应论文正文表1-4），无需额外加载外部权重文件。  
【运行方式】`python Three_Dimensional_Reconstruction_2025_09_29_RL.py`

---

## 2. 对比实验2（基于全局优化策略的智能标定方案）
对应论文第4.2.3节。

**`OnlyBPNN_Global_Predict_2026_02_23_.py`**  
三阶段优化MLP的三维重建程序（端到端前向回归）。  
【依赖文件】需加载预训练权重 `Only_BPNN_model_Global_Optimum.pth`（该文件为仅含权重的 state_dict，加载时需先定义网络结构）。  
【运行方式】`python OnlyBPNN_Global_Predict_2026_02_23_.py`

**`Fine_Tuning_Model_with_BN_Predict_2026_02_23.py`**  
四阶段优化浅层神经网络（含批归一化与Dropout）的三维重建程序。  
【依赖文件】需加载预训练权重 `Deep_Learning_based_3D_Reconstruction_Model.pth`（该文件为仅含权重的 state_dict）。  
【运行方式】`python Fine_Tuning_Model_with_BN_Predict_2026_02_23.py`

---

## 3. 类消融实验程序
对应论文第4.2.4节。

**`Restructuring_NN_GA_2025_11_07.py`**  
消融实验1主程序：去除PIO全局优化模块的重组神经网络三维重建。  
【权重来源】局部最优权重已硬编码在脚本内，无需外部文件。  
【运行方式】`python Restructuring_NN_GA_2025_11_07.py`

**`Restructuring_NN_GA_2025_11_30.py`** 与 **`Restructuring_NN_GA_2025_12_01.py`**  
消融实验1的补充实验（分别对应不同随机初始化种子下的重复验证，以排除偶然性）。  
【权重来源】局部最优权重已硬编码在脚本内。  
【运行方式】同上。

**`OnlyBPNN_2025_11_09_4_36_3_Predict.py`**  
消融实验2：去除全局搜索模块的常规BP神经网络三维重建（普通MLP）。  
【依赖文件】需加载 `only_bpnn_model.pth`（该文件为仅含权重的 state_dict）。  
【运行方式】`python OnlyBPNN_2025_11_09_4_36_3_Predict.py`

---

## 4. 误差分析与可视化脚本
对应论文第4.2节的误差统计与箱线图绘制。

**`RHistogram_of_Errors_Boxplot_of_Errors_0.py`**  
绘制四种方法（本文方法、张正友法、MLP+全局搜索、DLBND+全局搜索）的误差直方图与箱线图（对应论文图8及表5）。  
【运行方式】`python RHistogram_of_Errors_Boxplot_of_Errors_0.py`

**`RHistogram_of_Errors_Boxplot_of_Errors_1.py`**  
绘制类消融实验（去除全局优化后的重组NN与普通MLP）的误差直方图与箱线图（对应论文图9及表6）。  
【运行方式】`python RHistogram_of_Errors_Boxplot_of_Errors_1.py`

---

## 5. 工程应用实验程序
对应论文第4.2.5节。

**`Three_Dimensional_Reconstruction_2025_09_28_PCB.py`**  
PCB芯片引脚间距精密测量（工程应用1）。  
【权重来源】权重硬编码在脚本内。  
【运行方式】`python Three_Dimensional_Reconstruction_2025_09_28_PCB.py`

**`Three_Dimensional_Reconstruction_2025_11_29_Gauge.py`**  
量块深度精密测量（工程应用2）。  
【权重来源】权重硬编码在脚本内。  
【运行方式】`python Three_Dimensional_Reconstruction_2025_11_29_Gauge.py`

---

## 6. 原始数据文件
- **`Coordinates_of_calibration_board.doc`**：标定板49个孔心在多个Z轴位置的原始坐标数据（供查阅，程序内已内嵌关键数据）。  
- **`PCB_pin_coordinates.xlsx`**：PCB芯片引脚左右图像坐标及三维重建结果。  
- **`Gauge_block_coordinates.xlsx`**：量块反光点中心左右图像坐标及三维重建结果。

---

## 7. 辅助文件
- **`README.md`**：本说明文件。  
- **`requirements.txt`**：Python依赖包列表。

---

## 注意事项
- 所有 `.pth` 权重文件均为 **PyTorch 的 state_dict（仅含权重）**，并非完整的模型实例。加载时请先实例化对应的网络结构，再调用 `model.load_state_dict(torch.load('file.pth'))`。
- 若运行报错，请优先检查 `numpy` 和 `torch` 版本是否与 `requirements.txt` 一致。
