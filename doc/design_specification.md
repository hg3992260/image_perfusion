# CT和MRI灌注计算及可视化程序设计方案

## 1. 项目概述
本项目旨在开发一个能够处理CT和MRI动态扫描数据，计算脑灌注参数（如CBV, CBF, MTT, TTP等），并进行可视化的软件工具。

## 2. 参考文献
本项目遵循以下文献中的物理原理和计算方法：
*   **CT原理**: *Computed Tomography. Approaches, Applications, and Operations, 2020* (位于 `kb/` 目录)
*   **MRI原理**: *Magnetic resonance imaging: physical principles and sequence design, 2014* (位于 `kb/` 目录)

## 3. 理论基础

### 3.1 核心模型：示踪剂动力学
无论是CT还是MRI灌注，其核心均基于**中心容积定理 (Central Volume Principle)**：
$$ CBF = \frac{CBV}{MTT} $$

组织中的示踪剂浓度 $C_{tissue}(t)$ 可以表示为动脉输入函数 (AIF, $C_{a}(t)$) 与组织残留函数 (Residue Function, $R(t)$) 的卷积，并乘以脑血流量 (CBF)：
$$ C_{tissue}(t) = CBF \cdot [C_{a}(t) \otimes R(t)] $$

其中：
*   $C_{tissue}(t)$: 组织中随时间变化的对比剂浓度。
*   $C_{a}(t)$: 供血动脉（通常选大脑前动脉或中动脉）中的对比剂浓度。
*   $R(t)$: 示踪剂在组织中停留的概率函数（理想情况下 $t=0$ 时为1，随时间衰减）。
*   $\otimes$: 卷积运算。

### 3.2 CT 灌注 (CTP)
*   **信号转换**: CT图像的HU值变化与碘对比剂浓度成线性正比关系。
    $$ C(t) \propto HU(t) - HU_{baseline} $$
*   **关键步骤**:
    1.  基线校正。
    2.  提取 AIF (Arterial Input Function) 和 VOF (Venous Output Function)。
    3.  利用奇异值分解 (SVD) 进行去卷积，解出 $R(t) \cdot CBF$。
    4.  计算参数：
        *   **CBF (Cerebral Blood Flow)**: $R(t)$ 的最大值（理论上 $t=0$ 时）。
        *   **CBV (Cerebral Blood Volume)**: 组织浓度曲线下面积 / AIF曲线下面积（或者利用 VOF 进行刻度校正）。
        *   **MTT (Mean Transit Time)**: $CBV / CBF$。
        *   **TTP (Time to Peak)**: 浓度达到峰值的时间。

### 3.3 MRI 灌注 (DSC-MRI)
*   **信号转换**: 动态磁敏感对比增强 (DSC) MRI 利用钆对比剂的T2*缩短效应。信号强度与浓度呈非线性关系：
    $$ S(t) = S_0 \cdot e^{-\kappa \cdot TE \cdot C(t)} $$
    转换公式为：
    $$ C(t) = -\frac{1}{TE} \cdot \ln\left(\frac{S(t)}{S_0}\right) $$
*   **关键步骤**:
    1.  信号转浓度 (Signal-to-Concentration)。
    2.  AIF 选择（通常选大脑中动脉 MCA）。
    3.  去卷积 (SVD/oSVD/cSVD)。
    4.  漏出效应校正 (Leakage Correction) - 如果血脑屏障破坏。

## 4. 软件架构设计

### 4.1 目录结构
```
src/
├── core/
│   ├── deconvolution.py  # 核心算法：SVD去卷积
│   ├── ct_perfusion.py   # CT特定处理逻辑
│   └── mri_perfusion.py  # MRI特定处理逻辑
├── utils/
│   ├── dicom_loader.py   # DICOM/NIfTI 数据加载
│   └── preprocessing.py  # 图像预处理（平滑、掩膜）
├── visualization/
│   └── plotter.py        # 参数图生成与显示
└── main.py               # 程序入口
```

### 4.2 关键模块说明

#### A. 数据加载 (`dicom_loader`)
*   支持读取 4D DICOM 序列 (x, y, z, time) 或 NIfTI 格式。
*   解析关键元数据：TR, TE (MRI), FrameTime (CT)。

#### B. 预处理 (`preprocessing`)
*   **运动校正**: (可选，初期版本跳过) 使用简单的刚体配准。
*   **高斯平滑**: 降低噪声对去卷积的不利影响。
*   **脑掩膜生成**: 基于阈值去除背景和颅骨。

#### C. 去卷积核心 (`deconvolution`)
*   实现 **oSVD (Oscillation Index SVD)** 或 **Tikhonov Regularization** 算法，以解决反卷积的病态问题。
*   输入: $C_{tissue}(t)$, $C_{a}(t)$, $dt$。
*   输出: $CBF$, $R(t)$。

#### D. 可视化 (`plotter`)
*   生成伪彩图 (Color Maps)。
*   支持并排显示原始图像和参数图。

## 5. 开发计划
1.  搭建基础环境 (`src/requirements.txt`)。
2.  实现 SVD 去卷积算法。
3.  构建 CT 和 MRI 的处理流水线。
4.  集成可视化功能。
5.  测试与验证。
