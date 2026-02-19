#### 基于 C++ 的高性能软光栅渲染器 



- **技术深度要求：**
  - **语言：** 纯 C++ (C++17标准)，第三方图形库只有STB_Image和tiny_obj_loader。
  - **管线：** 完整的顶点处理 -> 光栅化 -> 片元处理 -> 深度测试。
  - **核心特性：**
    - 实现透视校正插值 (Perspective Correct Interpolation)。
    - 实现视锥体剔除 (Frustum Culling) 和背面剔除 (Back-face Culling)。
    - 实现切线空间法线贴图 (Tangent Space Normal Mapping)。
  - **优化：**
    - **多线程：** 使用 OpenMP 将三角形光栅化并行化。
    - **SIMD：** 手写 SSE/AVX 指令集优化 4x4 矩阵运算、批量顶点的变换、SoA 布局下的向量计算。





### High-Performance Soft Rasterizer (C++)

**Tech Stack**: C++17, STB_Image, OpenMP, SSE/AVX Intrinsics.

**Key Implementations**:

- Built an end-to-end software rendering pipeline: vertex processing → rasterization → fragment shading → depth testing.
- Implemented perspective-correct interpolation, frustum culling, and tangent-space normal mapping for photorealistic texturing.
- Optimized parallel triangle rasterization with OpenMP; manually accelerated 4x4 matrix operations via SSE/AVX intrinsics (3x faster than scalar computation).