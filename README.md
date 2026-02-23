# AeroRaster: Extreme Performance SIMD Software Rasterizer

![Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20x86__64-lightgrey.svg)
![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2F%20BMI1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AeroRaster** 是一款专注于极致性能的 Tile-Based 纯 CPU 软光栅渲染器。

本项目基于 **Eigen** 处理线性代数运算，利用 **tinyobjloader** 与 **stb_image** 处理资产管线，将核心工程精力完全聚焦于**光栅化管线 (Rasterization Pipeline)** 的微架构级优化、**数据导向设计 (DOD)** 以及 **AVX2 SIMD 指令集** 的极限吞吐量挖掘。

## 🚀 核心特性与性能高光

在 Intel Core i7-12650H (Alder Lake) 上，渲染 Diablo3 角色（3.2k 顶点，5k 三角形，Blinn-Phong 光照，1024x1024 漫反射/法线/高光贴图），性能表现如下：

| Metric                      | Measurement              |
| :-------------------------- | :----------------------- |
| **Frame Rate**              | **1300+ FPS**            |
| **Total Time (100 Frames)** | **0.075s**               |
| **Geometry & Binning**      | 0.0075s (Single Thread)  |
| **Rasterization**           | 0.068s (16 Threads AVX2) |

---

## 🏗️ 架构解析 (Architecture)

### 1. 极致的数据导向设计 (DOD)
虽然使用 `tinyobjloader` 加载标准 OBJ 模型，但 AeroRaster 在加载后立即将数据转换为严格的 **Structure of Arrays (SoA)** 布局 (`MeshSoA`)，并强制内存向 32 字节（AVX2 寄存器宽度）对齐：
- **L1 Cache 亲和性**：连续的内存布局使得预取器（Hardware Prefetcher）能完美工作。
- **Zero-Overhead Loading**：数据无需重排即可直接加载到 YMM 寄存器 (`_mm256_load_ps`)。
- **False Sharing 消除**：不同线程处理不同 Tile 时，不会因为 Cache Line 脏写导致核心间缓存一致性流量风暴。

### 2. 微架构级光栅化优化
这是本项目性能超越同类竞品的关键。

*   **拒绝 Masked Store**：许多软光栅实现使用 `_mm256_maskstore_ps` 来处理三角形边缘覆盖。经 VTune 分析，该指令在现代 Intel CPU 上会导致微码展开（Microcode Assist）惩罚，且非对齐的 Masked Store 极其昂贵。
*   **BMI1 指令集加速**：我们创新性地利用 BMI1 指令集（`_tzcnt_u32`, `_blsr_u32`）实现了**掩码迭代器 (Bitmask Iterator)**。
    *   通过计算覆盖掩码的尾部零计数（Trailing Zeros），我们能以 **0 周期延迟**直接跳转到下一个有效像素块。
    *   将读-改-写（RMW）操作转化为纯粹的**只写**流，极大降低了内存总线压力。

### 3. 混合架构友好的无锁调度
针对 Intel Alder Lake 的大小核（P-Core / E-Core）架构，传统的静态任务分配会导致严重的“木桶效应”（P-Core 等待 E-Core）。

AeroRaster 实现了一个轻量级**无锁任务系统 (JobSystem)**：
*   **Atomic Fetch-Add**：使用 `std::atomic::fetch_add` 维护全局任务索引，实现纳秒级任务分发。
*   **动态负载平衡**：性能核处理速度快，会自动领取更多 Tile，能效核处理较少，整个系统永远处于满载状态。

---

##  依赖项 (Dependencies)

本项目坚持“不重复造轮子，但造最好的引擎”原则，使用了以下高质量的轻量级单头文件库（Header-only Libraries）：

*   **[Eigen](https://eigen.tuxfamily.org/)**: 高性能线性代数库，提供基础的矩阵与向量运算支持（SIMD 友好）。
*   **[tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)**: 快速、鲁棒的 Wavefront .obj 模型加载器。
*   **[stb_image](https://github.com/nothings/stb)**: 极简的图像加载库，用于解析 TGA/PNG 纹理。

---

## 🛠️ 构建与运行 (Build & Run)

### 环境要求
*   CPU: 支持 AVX2 指令集 (Intel Haswell 及以上, AMD Ryzen 及以上)
*   Compiler: MSVC (Visual Studio 2019+) / GCC 9+ / Clang 10+
*   System: Windows (推荐), Linux

### 构建步骤 (Windows/Visual Studio)
1.  打开 `AeroRaster.sln`。
2.  选择 **Release** 模式（Debug 模式无法启用 SIMD 优化，性能相差 100 倍）。
3.  确保项目属性中已启用 `/arch:AVX2`。
4.  运行项目。

```bash
# 运行后将在 images/ 目录下生成 output_frame_xx.ppm
./x64/Release/AeroRaster.exe
```

---

## 🔮 路线图 (更新中)

*   [ ] **透视矫正插值 (Perspective Correct Interpolation)**: 引入 1/Z 缓冲实现精确纹理映射。
*   [ ] **SIMD 双线性过滤**: 优化 `SampleAVX2` 函数，实现完全向量化的 Bilinear Filtering。
*   [ ] **定点数光栅化**: 探索 16.16 定点数数学库，进一步降低浮点单元（FPU）压力。
*   [ ] **Tile 遮挡剔除**: 实现基于 Hierarchical-Z 的早期 Tile 剔除。

---

## 📄 License

MIT License. Copyright (c) 2026 MDA21