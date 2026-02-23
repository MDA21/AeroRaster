#pragma once
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <cmath>
#include <Eigen/Dense>
#include "image.h"
#include "Tile.h"
#include "Mesh.h"
#include "Texture.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


void TransformVerticesAVX2(const MeshSoA& in, MeshSoA& out, const Matrix4f& mat) {
    size_t count = in.GetVertexCount();

    // Row 0 - used for x_out
	__m256 m00 = _mm256_set1_ps(mat(0, 0)); // x coeff
    __m256 m01 = _mm256_set1_ps(mat(0, 1)); // y coeff
	__m256 m02 = _mm256_set1_ps(mat(0, 2)); // z coeff
    __m256 m03 = _mm256_set1_ps(mat(0, 3)); // w coeff

	// Row 1 - used for y_out
	__m256 m10 = _mm256_set1_ps(mat(1, 0));
	__m256 m11 = _mm256_set1_ps(mat(1, 1));
	__m256 m12 = _mm256_set1_ps(mat(1, 2));
	__m256 m13 = _mm256_set1_ps(mat(1, 3));

    // Row 2 - used for z_out
	__m256 m20 = _mm256_set1_ps(mat(2, 0));
	__m256 m21 = _mm256_set1_ps(mat(2, 1));
	__m256 m22 = _mm256_set1_ps(mat(2, 2));
    __m256 m23 = _mm256_set1_ps(mat(2, 3));

	// Row 3 - used for w_out
	__m256 m30 = _mm256_set1_ps(mat(3, 0));
	__m256 m31 = _mm256_set1_ps(mat(3, 1));
	__m256 m32 = _mm256_set1_ps(mat(3, 2));
	__m256 m33 = _mm256_set1_ps(mat(3, 3));

    for (size_t i = 0; i < count; i += 8) {

        __m256 vx = _mm256_loadu_ps(&in.x[i]);
		__m256 vy = _mm256_loadu_ps(&in.y[i]);
		__m256 vz = _mm256_loadu_ps(&in.z[i]);
        // Assume w is 1.0 for input vertices, usually correct for obj models
        // But better be safe if input has w

        // Calculate x_out = x*M00 + y*M01 + z*M02 + 1*M03
        __m256 res_x = m03; // Start with translation (w*M03 where w=1)
		res_x = _mm256_fmadd_ps(vx, m00, res_x);
		res_x = _mm256_fmadd_ps(vy, m01, res_x);
		res_x = _mm256_fmadd_ps(vz, m02, res_x);

		// Calculate y_out = x*M10 + y*M11 + z*M12 + 1*M13
		__m256 res_y = m13;
		res_y = _mm256_fmadd_ps(vx, m10, res_y);
		res_y = _mm256_fmadd_ps(vy, m11, res_y);
		res_y = _mm256_fmadd_ps(vz, m12, res_y);

		// Calculate z_out = x*M20 + y*M21 + z*M22 + 1*M23
        __m256 res_z = m23;
		res_z = _mm256_fmadd_ps(vx, m20, res_z);
		res_z = _mm256_fmadd_ps(vy, m21, res_z);
		res_z = _mm256_fmadd_ps(vz, m22, res_z);

        // Calculate w_out = x*M30 + y*M31 + z*M32 + 1*M33
		__m256 res_w = m33;
		res_w = _mm256_fmadd_ps(vx, m30, res_w);
		res_w = _mm256_fmadd_ps(vy, m31, res_w);
        res_w = _mm256_fmadd_ps(vz, m32, res_w);

        //存储结果
		_mm256_storeu_ps(&out.x[i], res_x);
		_mm256_storeu_ps(&out.y[i], res_y);
		_mm256_storeu_ps(&out.z[i], res_z);
		_mm256_storeu_ps(&out.w[i], res_w);
    }
}

void PerspectiveDivideAVX2(MeshSoA& mesh) {
    size_t count = mesh.GetVertexCount();

    for (size_t i = 0; i < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&mesh.x[i]);
        __m256 y = _mm256_loadu_ps(&mesh.y[i]);
        __m256 z = _mm256_loadu_ps(&mesh.z[i]);
        __m256 w = _mm256_loadu_ps(&mesh.w[i]);

        __m256 inv_w = _mm256_div_ps(_mm256_set1_ps(1.0f), w);

        x = _mm256_mul_ps(x, inv_w);
        y = _mm256_mul_ps(y, inv_w);
        z = _mm256_mul_ps(z, inv_w);

        _mm256_storeu_ps(&mesh.x[i], x);
        _mm256_storeu_ps(&mesh.y[i], y);
        _mm256_storeu_ps(&mesh.z[i], z);
        _mm256_storeu_ps(&mesh.w[i], inv_w);
    }
}

void ViewportTransformAVX2(MeshSoA& mesh, float width, float height) {
    size_t count = mesh.GetVertexCount();
    __m256 half_width = _mm256_set1_ps(width * 0.5f);
    __m256 half_height = _mm256_set1_ps(height * 0.5f);

    for (size_t i = 0; i < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&mesh.x[i]);
        __m256 y = _mm256_loadu_ps(&mesh.y[i]);
        __m256 z = _mm256_loadu_ps(&mesh.z[i]);

        x = _mm256_fmadd_ps(x, half_width, half_width);
        y = _mm256_fnmadd_ps(y, half_height, half_height);

        _mm256_storeu_ps(&mesh.x[i], x);
        _mm256_storeu_ps(&mesh.y[i], y);
        _mm256_storeu_ps(&mesh.z[i], z);
    }
}

void ProcessGeometryAVX2(const MeshSoA& in, MeshSoA& out, const Matrix4f& mvp, const Matrix4f& model, float width, float height) {
    size_t count = in.GetVertexCount();

    // Prepare MVP matrix constants
    __m256 m00 = _mm256_set1_ps(mvp(0, 0)); __m256 m01 = _mm256_set1_ps(mvp(0, 1));
    __m256 m02 = _mm256_set1_ps(mvp(0, 2)); __m256 m03 = _mm256_set1_ps(mvp(0, 3));
    __m256 m10 = _mm256_set1_ps(mvp(1, 0)); __m256 m11 = _mm256_set1_ps(mvp(1, 1));
    __m256 m12 = _mm256_set1_ps(mvp(1, 2)); __m256 m13 = _mm256_set1_ps(mvp(1, 3));
    __m256 m20 = _mm256_set1_ps(mvp(2, 0)); __m256 m21 = _mm256_set1_ps(mvp(2, 1));
    __m256 m22 = _mm256_set1_ps(mvp(2, 2)); __m256 m23 = _mm256_set1_ps(mvp(2, 3));
    __m256 m30 = _mm256_set1_ps(mvp(3, 0)); __m256 m31 = _mm256_set1_ps(mvp(3, 1));
    __m256 m32 = _mm256_set1_ps(mvp(3, 2)); __m256 m33 = _mm256_set1_ps(mvp(3, 3));

    // Prepare Model Matrix constants
    __m256 w00 = _mm256_set1_ps(model(0, 0)); __m256 w01 = _mm256_set1_ps(model(0, 1));
    __m256 w02 = _mm256_set1_ps(model(0, 2)); __m256 w03 = _mm256_set1_ps(model(0, 3));
    __m256 w10 = _mm256_set1_ps(model(1, 0)); __m256 w11 = _mm256_set1_ps(model(1, 1));
    __m256 w12 = _mm256_set1_ps(model(1, 2)); __m256 w13 = _mm256_set1_ps(model(1, 3));
    __m256 w20 = _mm256_set1_ps(model(2, 0)); __m256 w21 = _mm256_set1_ps(model(2, 1));
    __m256 w22 = _mm256_set1_ps(model(2, 2)); __m256 w23 = _mm256_set1_ps(model(2, 3));

    // Prepare Viewport constants
    __m256 half_width = _mm256_set1_ps(width * 0.5f);
    __m256 half_height = _mm256_set1_ps(height * 0.5f);
    __m256 one = _mm256_set1_ps(1.0f);

    for (size_t i = 0; i < count; i += 8) {
        // 1. TransformVerticesAVX2 logic
        __m256 vx = _mm256_loadu_ps(&in.x[i]);
        __m256 vy = _mm256_loadu_ps(&in.y[i]);
        __m256 vz = _mm256_loadu_ps(&in.z[i]);
        // Assume w is 1.0 for input vertices

        __m256 x = _mm256_fmadd_ps(vz, m02, _mm256_fmadd_ps(vy, m01, _mm256_fmadd_ps(vx, m00, m03)));
        __m256 y = _mm256_fmadd_ps(vz, m12, _mm256_fmadd_ps(vy, m11, _mm256_fmadd_ps(vx, m10, m13)));
        __m256 z = _mm256_fmadd_ps(vz, m22, _mm256_fmadd_ps(vy, m21, _mm256_fmadd_ps(vx, m20, m23)));
        __m256 w = _mm256_fmadd_ps(vz, m32, _mm256_fmadd_ps(vy, m31, _mm256_fmadd_ps(vx, m30, m33)));

        // 2. PerspectiveDivideAVX2 logic
        __m256 inv_w = _mm256_div_ps(one, w);
        x = _mm256_mul_ps(x, inv_w);
        y = _mm256_mul_ps(y, inv_w);
        z = _mm256_mul_ps(z, inv_w);
        // Store inv_w as w in output, consistent with PerspectiveDivideAVX2

        // 3. ViewportTransformAVX2 logic
        x = _mm256_fmadd_ps(x, half_width, half_width);
        y = _mm256_fnmadd_ps(y, half_height, half_height);

        // Store results
        _mm256_storeu_ps(&out.x[i], x);
        _mm256_storeu_ps(&out.y[i], y);
        _mm256_storeu_ps(&out.z[i], z);
        _mm256_storeu_ps(&out.w[i], inv_w);

        // 4. Transform World Attributes
        
        // World Pos (Point, w=1)
        __m256 wx = _mm256_fmadd_ps(vz, w02, _mm256_fmadd_ps(vy, w01, _mm256_fmadd_ps(vx, w00, w03)));
        __m256 wy = _mm256_fmadd_ps(vz, w12, _mm256_fmadd_ps(vy, w11, _mm256_fmadd_ps(vx, w10, w13)));
        __m256 wz = _mm256_fmadd_ps(vz, w22, _mm256_fmadd_ps(vy, w21, _mm256_fmadd_ps(vx, w20, w23)));
        
        _mm256_storeu_ps(&out.wx[i], wx);
        _mm256_storeu_ps(&out.wy[i], wy);
        _mm256_storeu_ps(&out.wz[i], wz);

        // World Normal (Vector, w=0)
        __m256 in_nx = _mm256_loadu_ps(&in.nx[i]);
        __m256 in_ny = _mm256_loadu_ps(&in.ny[i]);
        __m256 in_nz = _mm256_loadu_ps(&in.nz[i]);
        
        __m256 nx = _mm256_fmadd_ps(in_nz, w02, _mm256_fmadd_ps(in_ny, w01, _mm256_mul_ps(in_nx, w00)));
        __m256 ny = _mm256_fmadd_ps(in_nz, w12, _mm256_fmadd_ps(in_ny, w11, _mm256_mul_ps(in_nx, w10)));
        __m256 nz = _mm256_fmadd_ps(in_nz, w22, _mm256_fmadd_ps(in_ny, w21, _mm256_mul_ps(in_nx, w20)));

        _mm256_storeu_ps(&out.nx[i], nx);
        _mm256_storeu_ps(&out.ny[i], ny);
        _mm256_storeu_ps(&out.nz[i], nz);

        // World Tangent (Vector, w=0)
        __m256 in_tx = _mm256_loadu_ps(&in.tx[i]);
        __m256 in_ty = _mm256_loadu_ps(&in.ty[i]);
        __m256 in_tz = _mm256_loadu_ps(&in.tz[i]);
        
        __m256 tx = _mm256_fmadd_ps(in_tz, w02, _mm256_fmadd_ps(in_ty, w01, _mm256_mul_ps(in_tx, w00)));
        __m256 ty = _mm256_fmadd_ps(in_tz, w12, _mm256_fmadd_ps(in_ty, w11, _mm256_mul_ps(in_tx, w10)));
        __m256 tz = _mm256_fmadd_ps(in_tz, w22, _mm256_fmadd_ps(in_ty, w21, _mm256_mul_ps(in_tx, w20)));

        _mm256_storeu_ps(&out.tx[i], tx);
        _mm256_storeu_ps(&out.ty[i], ty);
        _mm256_storeu_ps(&out.tz[i], tz);
        _mm256_storeu_ps(&out.tw[i], _mm256_loadu_ps(&in.tw[i])); // Handedness unchanged
    }
}

inline float GetEdgeBias(float dx, float dy) {
	//top-left rule: 当边界上的像素中心在边界的上方或左侧时，包含该像素；当在边界的下方或右侧时，不包含该像素
    bool isTopLeft = (dy < 0.0f) || (dy == 0.0f && dx > 0.0f);
    return isTopLeft ? 0.0f : -0.00001f;
}

inline void RasterizeTriangleForTile(
    Framebuffer& fb, 
    const MeshSoA& mesh, 
    const MeshSoA& transformedMesh, 
    uint32_t triIdx, 
    const Tile& tile,
    const Texture* diffuseMap,
    const Texture* normalMap,
    const Texture* specMap,
    const Eigen::Vector3f& lightDir,
    const Eigen::Vector3f& viewPos
) {
    uint32_t i0 = mesh.indices[triIdx];
    uint32_t i1 = mesh.indices[triIdx + 1];
    uint32_t i2 = mesh.indices[triIdx + 2];


    float x0 = transformedMesh.x[i0], y0 = transformedMesh.y[i0], z0 = transformedMesh.z[i0], w0 = transformedMesh.w[i0];
    float x1 = transformedMesh.x[i1], y1 = transformedMesh.y[i1], z1 = transformedMesh.z[i1], w1 = transformedMesh.w[i1];
    float x2 = transformedMesh.x[i2], y2 = transformedMesh.y[i2], z2 = transformedMesh.z[i2], w2 = transformedMesh.w[i2];

    float u0 = mesh.u[i0], v0 = mesh.v[i0];
    float u1 = mesh.u[i1], v1 = mesh.v[i1];
    float u2 = mesh.u[i2], v2 = mesh.v[i2];

    // Load World Attributes
    float wx0 = transformedMesh.wx[i0], wy0 = transformedMesh.wy[i0], wz0 = transformedMesh.wz[i0];
    float wx1 = transformedMesh.wx[i1], wy1 = transformedMesh.wy[i1], wz1 = transformedMesh.wz[i1];
    float wx2 = transformedMesh.wx[i2], wy2 = transformedMesh.wy[i2], wz2 = transformedMesh.wz[i2];

    float nx0 = transformedMesh.nx[i0], ny0 = transformedMesh.ny[i0], nz0 = transformedMesh.nz[i0];
    float nx1 = transformedMesh.nx[i1], ny1 = transformedMesh.ny[i1], nz1 = transformedMesh.nz[i1];
    float nx2 = transformedMesh.nx[i2], ny2 = transformedMesh.ny[i2], nz2 = transformedMesh.nz[i2];

    float tx0 = transformedMesh.tx[i0], ty0 = transformedMesh.ty[i0], tz0 = transformedMesh.tz[i0], tw0 = transformedMesh.tw[i0];
    float tx1 = transformedMesh.tx[i1], ty1 = transformedMesh.ty[i1], tz1 = transformedMesh.tz[i1], tw1 = transformedMesh.tw[i1];
    float tx2 = transformedMesh.tx[i2], ty2 = transformedMesh.ty[i2], tz2 = transformedMesh.tz[i2], tw2 = transformedMesh.tw[i2];

    float dx01 = x1 - x0, dy01 = y1 - y0;
    float dx12 = x2 - x1, dy12 = y2 - y1;
    float dx20 = x0 - x2, dy20 = y0 - y2;

    float areaDouble = dx12 * (y0 - y1) - dy12 * (x0 - x1);

    // 统一绕序为 CCW (area > 0)
    if (areaDouble <= 0.0f) {
        if (areaDouble == 0.0f) return;

        std::swap(x1, x2); std::swap(y1, y2); std::swap(z1, z2); std::swap(w1, w2);
        std::swap(u1, u2); std::swap(v1, v2);
        std::swap(wx1, wx2); std::swap(wy1, wy2); std::swap(wz1, wz2);
        std::swap(nx1, nx2); std::swap(ny1, ny2); std::swap(nz1, nz2);
        std::swap(tx1, tx2); std::swap(ty1, ty2); std::swap(tz1, tz2); std::swap(tw1, tw2);

        dx01 = x1 - x0; dy01 = y1 - y0;
        dx12 = x2 - x1; dy12 = y2 - y1;
        dx20 = x0 - x2; dy20 = y0 - y2;

        areaDouble = dx12 * (y0 - y1) - dy12 * (x0 - x1);
        if (areaDouble <= 0.0f) return;
    }

    int triMinX = std::max(0, (int)std::floor(std::min({ x0, x1, x2 })));
    int triMaxX = std::min(fb.width - 1, (int)std::ceil(std::max({ x0, x1, x2 })));
    int triMinY = std::max(0, (int)std::floor(std::min({ y0, y1, y2 })));
    int triMaxY = std::min(fb.height - 1, (int)std::ceil(std::max({ y0, y1, y2 })));

    //求三角形 AABB 与当前 Tile 边界的交集
    int minX = std::max(triMinX, tile.startX);
    int maxX = std::min(triMaxX, tile.endX);
    int minY = std::max(triMinY, tile.startY);
    int maxY = std::min(triMaxY, tile.endY);

    if (minX > maxX || minY > maxY) {
        return;
    }

    // SIMD 准备
    minX &= ~3;
    minY &= ~1;

    float preU0 = u0 * w0, preU1 = u1 * w1, preU2 = u2 * w2;
    float preV0 = v0 * w0, preV1 = v1 * w1, preV2 = v2 * w2;

    float invArea = 1.0f / areaDouble;
    __m256 v_invArea = _mm256_set1_ps(invArea);

    __m256 v_x0 = _mm256_set1_ps(x0), v_y0 = _mm256_set1_ps(y0);
    __m256 v_x1 = _mm256_set1_ps(x1), v_y1 = _mm256_set1_ps(y1);
    __m256 v_x2 = _mm256_set1_ps(x2), v_y2 = _mm256_set1_ps(y2);

    __m256 v_dx01 = _mm256_set1_ps(dx01), v_dy01 = _mm256_set1_ps(dy01);
    __m256 v_dx12 = _mm256_set1_ps(dx12), v_dy12 = _mm256_set1_ps(dy12);
    __m256 v_dx20 = _mm256_set1_ps(dx20), v_dy20 = _mm256_set1_ps(dy20);

    __m256 v_z0 = _mm256_set1_ps(z0), v_z1 = _mm256_set1_ps(z1), v_z2 = _mm256_set1_ps(z2);
    __m256 v_w0 = _mm256_set1_ps(w0), v_w1 = _mm256_set1_ps(w1), v_w2 = _mm256_set1_ps(w2);
    __m256 v_preU0 = _mm256_set1_ps(preU0), v_preU1 = _mm256_set1_ps(preU1), v_preU2 = _mm256_set1_ps(preU2);
    __m256 v_preV0 = _mm256_set1_ps(preV0), v_preV1 = _mm256_set1_ps(preV1), v_preV2 = _mm256_set1_ps(preV2);

    float bias0 = GetEdgeBias(dx12, dy12);
    float bias1 = GetEdgeBias(dx20, dy20);
    float bias2 = GetEdgeBias(dx01, dy01);

    __m256 v_bias0 = _mm256_set1_ps(bias0);
    __m256 v_bias1 = _mm256_set1_ps(bias1);
    __m256 v_bias2 = _mm256_set1_ps(bias2);

    __m256 v_offsetX = _mm256_setr_ps(0.5f, 1.5f, 2.5f, 3.5f, 0.5f, 1.5f, 2.5f, 3.5f);
    __m256 v_offsetY = _mm256_setr_ps(0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f);

	__m256i v_minX_minus1 = _mm256_set1_epi32(tile.startX - 1);
	__m256i v_maxX_plus1 = _mm256_set1_epi32(tile.endX + 1);
	__m256i v_minY_minus1 = _mm256_set1_epi32(tile.startY - 1);
	__m256i v_maxY_plus1 = _mm256_set1_epi32(tile.endY + 1);

    for (int y = minY; y <= maxY; y += 2) {
        __m256 v_py = _mm256_add_ps(_mm256_set1_ps((float)y), v_offsetY);
        __m256 v_py_sub_y1 = _mm256_sub_ps(v_py, v_y1);
        __m256 v_py_sub_y2 = _mm256_sub_ps(v_py, v_y2);
        __m256 v_py_sub_y0 = _mm256_sub_ps(v_py, v_y0);

        __m256i v_py_i = _mm256_add_epi32(_mm256_set1_epi32(y), _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));

        for (int x = minX; x <= maxX; x += 4) {
            __m256 v_px = _mm256_add_ps(_mm256_set1_ps((float)x), v_offsetX);
            __m256 v_px_sub_x1 = _mm256_sub_ps(v_px, v_x1);
            __m256 v_px_sub_x2 = _mm256_sub_ps(v_px, v_x2);
            __m256 v_px_sub_x0 = _mm256_sub_ps(v_px, v_x0);

            // Pineda 边缘函数
            __m256 eval0 = _mm256_fmsub_ps(v_dx12, v_py_sub_y1, _mm256_mul_ps(v_dy12, v_px_sub_x1));
            __m256 eval1 = _mm256_fmsub_ps(v_dx20, v_py_sub_y2, _mm256_mul_ps(v_dy20, v_px_sub_x2));
            __m256 eval2 = _mm256_fmsub_ps(v_dx01, v_py_sub_y0, _mm256_mul_ps(v_dy01, v_px_sub_x0));

            __m256 eval0_bias = _mm256_add_ps(eval0, v_bias0);
            __m256 eval1_bias = _mm256_add_ps(eval1, v_bias1);
            __m256 eval2_bias = _mm256_add_ps(eval2, v_bias2);

            __m256 mask0 = _mm256_cmp_ps(eval0_bias, _mm256_setzero_ps(), _CMP_GE_OQ);
            __m256 mask1 = _mm256_cmp_ps(eval1_bias, _mm256_setzero_ps(), _CMP_GE_OQ);
            __m256 mask2 = _mm256_cmp_ps(eval2_bias, _mm256_setzero_ps(), _CMP_GE_OQ);

            __m256 finalMask = _mm256_and_ps(_mm256_and_ps(mask0, mask1), mask2);

            int maskBit = _mm256_movemask_ps(finalMask);
            if (_mm256_movemask_ps(finalMask) == 0) continue;

            //边界检查
            __m256i v_px_i = _mm256_add_epi32(_mm256_set1_epi32(x), _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3));
            __m256i bx = _mm256_and_si256(_mm256_cmpgt_epi32(v_px_i, v_minX_minus1), _mm256_cmpgt_epi32(v_maxX_plus1, v_px_i));
            __m256i by = _mm256_and_si256(_mm256_cmpgt_epi32(v_py_i, v_minY_minus1), _mm256_cmpgt_epi32(v_maxY_plus1, v_py_i));
            finalMask = _mm256_and_ps(finalMask, _mm256_castsi256_ps(_mm256_and_si256(bx, by)));
            if (_mm256_movemask_ps(finalMask) == 0) continue;

            //计算重心坐标 lambda
            __m256 lambda0 = _mm256_mul_ps(eval0, v_invArea);
            __m256 lambda1 = _mm256_mul_ps(eval1, v_invArea);
            __m256 lambda2 = _mm256_mul_ps(eval2, v_invArea);

            __m256 z_out = _mm256_fmadd_ps(lambda2, v_z2, _mm256_fmadd_ps(lambda1, v_z1, _mm256_mul_ps(lambda0, v_z0)));

            alignas(32) float z_arr[8];
            _mm256_store_ps(z_arr, z_out);

            bool uvCalculated = false;
            alignas(32) float u_arr[8], v_arr[8];
            alignas(32) float lambda0_arr[8], lambda1_arr[8], lambda2_arr[8];
            alignas(32) float w_recip_arr[8];

            maskBit = _mm256_movemask_ps(finalMask);

            //跳跃式遍历
            while (maskBit) {
                int i = _tzcnt_u32(maskBit);
                maskBit = _blsr_u32(maskBit);

                int px = x + (i & 3); 
                int py = y + (i >> 2); 

                if (px > tile.endX || py > tile.endY || px < tile.startX || py < tile.startY) continue;

                int pixelIdx = py * fb.width + px;

                if (z_arr[i] >= fb.depthBuffer[pixelIdx]) continue;
                fb.depthBuffer[pixelIdx] = z_arr[i];

                if (!uvCalculated) {
                    __m256 inv_w = _mm256_fmadd_ps(lambda2, v_w2, _mm256_fmadd_ps(lambda1, v_w1, _mm256_mul_ps(lambda0, v_w0)));
                    __m256 w_recip = _mm256_rcp_ps(inv_w);
                    __m256 u_out = _mm256_fmadd_ps(lambda2, v_preU2, _mm256_fmadd_ps(lambda1, v_preU1, _mm256_mul_ps(lambda0, v_preU0)));
                    __m256 v_out = _mm256_fmadd_ps(lambda2, v_preV2, _mm256_fmadd_ps(lambda1, v_preV1, _mm256_mul_ps(lambda0, v_preV0)));

                    _mm256_store_ps(u_arr, _mm256_mul_ps(u_out, w_recip));
                    _mm256_store_ps(v_arr, _mm256_mul_ps(v_out, w_recip));
                    
                    _mm256_store_ps(lambda0_arr, lambda0);
                    _mm256_store_ps(lambda1_arr, lambda1);
                    _mm256_store_ps(lambda2_arr, lambda2);
                    _mm256_store_ps(w_recip_arr, w_recip);
                    
                    uvCalculated = true;
                }

                float u = u_arr[i];
                float v = v_arr[i];
                float l0 = lambda0_arr[i];
                float l1 = lambda1_arr[i];
                float l2 = lambda2_arr[i];
                float w_recip = w_recip_arr[i];

                // Interpolate World Attributes
                // Note: We should technically interpolate (Attr * w0) then multiply by w_recip
                // Attr_pixel = (Attr0 * w0 * l0 + Attr1 * w1 * l1 + Attr2 * w2 * l2) * w_recip
                
                auto Interpolate = [&](float val0, float val1, float val2) {
                     return (val0 * w0 * l0 + val1 * w1 * l1 + val2 * w2 * l2) * w_recip;
                };

                Eigen::Vector3f worldPos;
                worldPos.x() = Interpolate(wx0, wx1, wx2);
                worldPos.y() = Interpolate(wy0, wy1, wy2);
                worldPos.z() = Interpolate(wz0, wz1, wz2);

                Eigen::Vector3f normal;
                normal.x() = Interpolate(nx0, nx1, nx2);
                normal.y() = Interpolate(ny0, ny1, ny2);
                normal.z() = Interpolate(nz0, nz1, nz2);
                normal.normalize();

                Eigen::Vector3f tangent;
                tangent.x() = Interpolate(tx0, tx1, tx2);
                tangent.y() = Interpolate(ty0, ty1, ty2);
                tangent.z() = Interpolate(tz0, tz1, tz2);
                tangent.normalize();

                // Handedness - use vertex 0's (should be consistent per triangle)
                float handedness = tw0; 
                
                // Calculate Bitangent
                Eigen::Vector3f bitangent = normal.cross(tangent) * handedness;
                bitangent.normalize();
                
                // TBN Matrix
                Eigen::Matrix3f TBN;
                TBN.col(0) = tangent;
                TBN.col(1) = bitangent;
                TBN.col(2) = normal;

                // Sample Normal Map
                Eigen::Vector3f N = normal;
                if (normalMap) {
                    Eigen::Vector4f nm = normalMap->Sample(u, v);
                    // [0,1] -> [-1,1]
                    Eigen::Vector3f normalSample = nm.head<3>() * 2.0f - Eigen::Vector3f(1.0f, 1.0f, 1.0f);
                    N = (TBN * normalSample).normalized();
                }

                // Sample Diffuse
                Eigen::Vector3f albedo(0.5f, 0.5f, 0.5f);
                if (diffuseMap) {
                    albedo = diffuseMap->Sample(u, v).head<3>();
                }
                
                // Sample Specular
                float specIntensity = 0.0f;
                if (specMap) {
                    specIntensity = specMap->Sample(u, v).x(); // Assume grayscale spec map
                }

                // Blinn-Phong Lighting
                Eigen::Vector3f L = lightDir.normalized();
                Eigen::Vector3f V = (viewPos - worldPos).normalized();
                Eigen::Vector3f H = (L + V).normalized();

                float diff = std::max(0.0f, N.dot(L));
                float spec = std::pow(std::max(0.0f, N.dot(H)), 32.0f) * specIntensity;

                Eigen::Vector3f ambient = albedo * 0.1f;
                Eigen::Vector3f diffuse = albedo * diff;
                Eigen::Vector3f specular = Eigen::Vector3f(1.0f, 1.0f, 1.0f) * spec;

                Eigen::Vector3f finalColor = ambient + diffuse + specular;

                // Clamp
                finalColor = finalColor.cwiseMin(1.0f).cwiseMax(0.0f);

                uint8_t r = (uint8_t)(finalColor.x() * 255.0f);
                uint8_t g = (uint8_t)(finalColor.y() * 255.0f);
                uint8_t b = (uint8_t)(finalColor.z() * 255.0f);

                fb.colorBuffer[pixelIdx] = (r << 16) | (g << 8) | b;
            }
        }
    }
}
