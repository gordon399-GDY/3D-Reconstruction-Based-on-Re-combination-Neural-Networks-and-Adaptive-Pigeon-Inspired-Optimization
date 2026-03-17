
# 仓库文件说明

## 1. 本文方法程序
**Three_Dimensional_Reconstruction_2025_09_29_RL.py**  
本文提出的“神经网络重组——固定模型权重——逆向优化输入”三维重建程序。  
【模型权重】无单独模型文件，权重见正文表1-表4。

## 2. 对比实验程序
**OnlyBPNN_Global_Predict_2026_02_23_.py**  
对应4.2.3.1节 三阶段优化MLP的三维重建（对比实验2——前沿智能方案）。  
【对应模型】`Only_BPNN_model_Global_Optimum.pth`

**Fine_Tuning_Model_with_BN_Predict_2026_02_23.py**  
对应4.2.3.2节 四阶段优化浅层神经网络的三维重建（对比实验2——前沿智能方案）。  
【对应模型】`Deep_Learning_based_3D_Reconstruction_Model.pth`

## 3. 消融实验程序
**OnlyBPNN_2025_11_09_4_36_3_Predict.py**  
对应4.2.4.1节 消融实验1——常规训练MLP的三维重建（无全局优化）。  
【对应模型】`only_bpnn_model.pth`

**Restructuring_NN_GA_2025_11_07.py**  
对应4.2.4.2节 消融实验2主程序（权重矩阵见正文表5-8）。  
【模型权重】程序内已包含局部最优权重，无需外部模型文件。

**Restructuring_NN_GA_2025_11_30.py**  
对应4.2.4.2节 消融实验2补充实验1。  
【模型权重】程序内已包含局部最优权重，无需外部模型文件。

**Restructuring_NN_GA_2025_12_01.py**  
对应4.2.4.2节 消融实验2补充实验2。  
【模型权重】程序内已包含局部最优权重，无需外部模型文件。

## 4. 工程应用实验程序
**Three_Dimensional_Reconstruction_2025_09_28_PCB.py**  
对应4.2.5.1节 工程应用实验1——PCB芯片引脚间距精密测量。  
【模型权重】无单独模型文件，权重见正文表1-表4。

**Three_Dimensional_Reconstruction_2025_11_29_Gauge.py**  
对应4.2.5.2节 工程应用实验2——量块深度精密测量。  
【模型权重】无单独模型文件，权重见正文表1-表4。

## 5. 原始数据文件
**Coordinates_of_calibration_board.doc**  
标定板49个孔心在多个Z轴位置的原始坐标数据。

**PCB_pin_coordinates.xlsx**  
PCB芯片引脚左右图像坐标及重建结果（表9、表10对应完整数据）。

**Gauge_block_coordinates.xlsx**  
量块反光点中心左右图像坐标及重建结果（表11、表12对应完整数据）。

## 6. 辅助文件
**README.md**  
项目说明文件（包含环境配置、运行方法、文件说明等）。

**requirements.txt**  
Python依赖包列表（如numpy, torch, opencv-python等）。

