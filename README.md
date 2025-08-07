# 🚁 Catch Anything V2: Multi-strategy UAV Target Tracking System
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>
> 🎯 *面向飞行视频中的动态目标追踪，适用于多类无人机系统*

Catch Anything V2 是一个融合多策略的目标追踪系统，专为处理无人机拍摄视频中的复杂跟踪任务而设计。系统通过光流法、模板匹配、颜色建模与自适应机制结合，在目标遮挡、快速移动、遮挡或消失等情况下依然实现高鲁棒性跟踪。


---

## 📢 Latest Updates
- **Aug-7-25**: [Catch Anything V2 [Demo]](https://www.bilibili.com/video/BV1J1tNzNEag/) has been uploaded. 🎉🎉
- **Aug-2-25**: [Catch Anything V1 [Project]](https://github.com/Jeffry-wen/Drone-Tracking-with-Optical-Flow-and-Color-Histogram) is released. 🔥🔥


---

## 🚀 与上一版本相比的改进

###  1.异常检测能力进一步增强

* 🔄 实现 **动态模板更新机制**，显著提升对环境变化的适应性
* 📅 引入 **周期性模板矫正流程**，确保模板长期稳定可靠
* 📐 增加 **目标区域面积异常检测**，及时捕捉显著缩放问题
* 🎯 支持 **特征点漂移分析**，发现微小偏移带来的潜在风险



###  2.多策略目标找回机制

* 🧩 融合 **最优模板匹配算法**，提高匹配精度与一致性
* 🧪 应用 **Lab 颜色空间掩码分析**，增强复杂场景下的颜色鲁棒性
* 🔍 引入 **CamShift 区域跟踪方法**，提升目标丢失后的自恢复能力



###  3.人机交互方式优化

* 🕹️ 增加 **实时人工干预功能**，支持快速、精细的人为校正
* ⚡ 优化交互流程，**提升操作灵活性与响应效率**
* 🧑‍💻 提供更友好的接口设计，**强化人机协作体验与系统可控性**

---



## 📌 Highlights

* ✅ **融合多模态跟踪策略**：结合光流追踪 + 模板匹配 + 颜色恢复三种策略，增强鲁棒性
* 🧠 **智能模板自更新机制**：基于面积稳定性自适应替换模板，实时优化追踪窗口
* 🎯 **目标缺失快速回溯策略**：面积或光流点丢失后立即执行初始模板匹配或颜色匹配
* 💡 **动态参数调优**：根据 ROI 尺寸动态调整角点质量、窗口大小与 Canny 阈值
* 💾 **完整视频导出**：实时生成带标注的追踪视频，便于回溯与训练数据生成

---

## 🧠 系统核心处理流程

1. **目标初始化**

   * 用户手动选取初始 ROI
   * 初始化光流点、颜色直方图、初始模板等参数

2. **光流追踪主流程**

   * 基于 Lucas-Kanade 光流跟踪角点
   * 计算中心与范围用于估计当前目标窗口

3. **模板稳定性判断**

   * 如果连续 50 帧内面积变化 $\delta < 0.2$ ，自动更新跟踪模板图像

4. **周期性回溯匹配**

   * 每 50 帧执行一次模板匹配，确保不漂移
   * 匹配得分高于阈值（如 \$0.8\$）时替换光流点和模板

5. **异常处理与多策略恢复**

   * 面积突变或光流丢失时：

     * 首选初始模板匹配
     * 若失败则尝试颜色相似性匹配
     * 预留 CamShift 模块用于多样恢复机制

---
## 🧮 核心算法介绍

### 1. **Lucas-Kanade 光流法**

* 跟踪稀疏特征点并预测其在当前帧中的位置
* 使用金字塔层次提高对尺度与运动的适应性
* 特征点筛选：

  * 中值滤波去除异常移动点
  * $|\mathbf{p}_i - \tilde{\mathbf{p}}| < 50$ 筛选有效点




### 2. **模板匹配 (Normalized Cross-Correlation)**

* 使用 OpenCV 的 `cv2.matchTemplate` 完成灰度图匹配
* 匹配得分计算：

```math
\text{score} = \frac{(T - \bar{T}) \ast (F - \bar{F})}{\|T - \bar{T}\| \cdot \|F - \bar{F}\|}
```

* 分数大于 0.7 视为有效匹配



### 3. **面积稳定性模板更新**

* 当前目标面积为 <code>A\_t</code>，上一帧为 <code>A\_{t-1}</code>，定义变化率：

```math
\delta = \frac{|A_t - A_{t-1}|}{\max(A_{t-1}, 1)}
```

* 若连续 <code>N</code> 帧满足 <code>\delta < 0.2</code>，则视为稳定区域 → 自动更新模板



### 4. **颜色相似性恢复（Lab颜色空间）**

* 提取当前帧 Lab 空间图像，与模板中心颜色统计距离如下：

```math
D(x, y) = \sqrt{\sum_{i=1}^3 \left( L_{x,y}^{(i)} - \mu^{(i)} \right)^2}
```

* 阈值使用第 10 百分位生成掩码
* 通过 `cv2.findContours` 找出最相似区域作为候选目标框

---

### 5. **动态参数调节**

根据 ROI 面积自适应选择参数：

| ROI 尺寸 | 光流参数 (`qualityLevel`, `blockSize`) | LK窗口  | Canny阈值 |
| ------ | ---------------------------------- | ----- | ------- |
| 小于 200 | 高精度，低质量阈值，blockSize=3              | 5x5   | 30\~100 |
| 大于 200 | 常规设置，blockSize=7                   | 21x21 | 50\~150 |

---
## 🖥️ 环境依赖

* Python >= 3.8
* OpenCV >= 4.5
* Numpy >= 1.19

安装依赖：
```bash
pip install opencv-python numpy scipy
```
---

## 🚀 运行方式
快速开始:
```bash
python catch_anything_v2.py
```
1. 修改 `VIDEO_PATH` 为本地无人机视频路径(也可以指定摄像头作为数据源)；
2. 运行脚本后手动框选目标区域；
3. 系统将自动进行目标跟踪并在窗口中展示跟踪过程；
4. 按下`R`可以进行识别过程中的人工校准,按下 `ESC` 键退出。


---
## 📁 文件结构说明

```bash
.
├── catch_anything_v2.py    # 主程序代码
├── README.md            # 项目说明文档（即本文件）
```

---

## 🎬 识别效果演示

![DEMO](https://github.com/Jeffry-wen/Catch-Anything-V2-Multi-strategy-UAV-target-tracking-system/blob/main/img/demo.gif)

🎥你可以在 [这里](https://www.bilibili.com/video/BV1J1tNzNEag/) 观看完整的视频演示



---

## 🎯 应用场景

* 三角翼/固定翼等多类无人机视频目标跟踪
* 安防监控中的小目标移动检测
* 运动分析与航拍视频增强处理
* 遮挡/漂移/缺失等复杂情景下的目标找回

---

## 🧩 未来规划（TODO）

* [ ] 基于先验运动知识提高追踪性能
* [ ] 添加多目标并行跟踪机制
* [ ] 集成深度特征匹配模块（如 SiamFC）
* [ ] 提供图形化 UI 与模型参数调整工具
* [ ] 提供轻量化高帧率版本适用于边缘设备

---
## 📜 许可协议

本项目感谢The shan的大力支持，因项目需要代码暂不公开。📬 [欢迎交流联系](mailto:jwen341@connect.hkust-gz.edu.cn)。

