#pragma once
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <cmath>
#include <Eigen/Dense>
#include "image.h"
#include "Tile.h"
#include "Mesh.h"

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

inline float GetEdgeBias(float dx, float dy) {
	//top-left rule: 当边界上的像素中心在边界的上方或左侧时，包含该像素；当在边界的下方或右侧时，不包含该像素
    bool isTopLeft = (dy < 0.0f) || (dy == 0.0f && dx > 0.0f);
    return isTopLeft ? 0.0f : -0.00001f;
}

void RasterizeTriangleAVX(
	Framebuffer& fb,
	float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2,
    float w0, float w1, float w2,
    float u0, float u1, float u2,
    float v0, float v1, float v2,
    int minX, int maxX, int minY, int maxY)
{
    float preU0 = u0 * w0;
    float preU1 = u1 * w1;
    float preU2 = u2 * w2;
    float preV0 = v0 * w0;
    float preV1 = v1 * w1;
    float preV2 = v2 * w2;

    float dx01 = x1 - x0, dy01 = y1 - y0;
    float dx12 = x2 - x1, dy12 = y2 - y1;
    float dx20 = x0 - x2, dy20 = y0 - y2;

    float areaDouble = dx01 * dy20 - dx20 * dy01;
    if (areaDouble <= 0.0f) {
        if (areaDouble == 0.0f) return;

        float tx = x1; x1 = x2; x2 = tx;
        float ty = y1; y1 = y2; y2 = ty;
        float tz = z1; z1 = z2; z2 = tz;
        float tw = w1; w1 = w2; w2 = tw;
        float tu = u1; u1 = u2; u2 = tu;
        float tv = v1; v1 = v2; v2 = tv;

        dx01 = x1 - x0; dy01 = y1 - y0;
        dx12 = x2 - x1; dy12 = y2 - y1;
        dx20 = x0 - x2; dy20 = y0 - y2;

        areaDouble = dx12 * (y0 - y1) - dy12 * (x0 - x1);
        if (areaDouble <= 0.0f) return;
    }


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

    minX &= ~3;
    minY &= ~1;

    __m256 v_offsetX = _mm256_setr_ps(0.5f, 1.5f, 2.5f, 3.5f, 0.5f, 1.5f, 2.5f, 3.5f);
    __m256 v_offsetY = _mm256_setr_ps(0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f);

    for (int y = minY; y <= maxY; y += 2) {
        __m256 v_py = _mm256_add_ps(_mm256_set1_ps((float)y), v_offsetY);
        __m256 v_py_sub_y1 = _mm256_sub_ps(v_py, v_y1);
        __m256 v_py_sub_y2 = _mm256_sub_ps(v_py, v_y2);
        __m256 v_py_sub_y0 = _mm256_sub_ps(v_py, v_y0);

        for (int x = minX; x <= maxX; x += 4) {
            __m256 v_px = _mm256_add_ps(_mm256_set1_ps((float)x), v_offsetX);
            __m256 v_px_sub_x1 = _mm256_sub_ps(v_px, v_x1);
            __m256 v_px_sub_x2 = _mm256_sub_ps(v_px, v_x2);
            __m256 v_px_sub_x0 = _mm256_sub_ps(v_px, v_x0);

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
            if (maskBit == 0) continue;

            __m256 lambda0 = _mm256_mul_ps(eval0, v_invArea);
            __m256 lambda1 = _mm256_mul_ps(eval1, v_invArea);
            __m256 lambda2 = _mm256_mul_ps(eval2, v_invArea);

            __m256 z_out = _mm256_fmadd_ps(lambda2, v_z2, _mm256_fmadd_ps(lambda1, v_z1, _mm256_mul_ps(lambda0, v_z0)));
            alignas(32) float z_arr[8];
            _mm256_store_ps(z_arr, z_out);

            __m256 inv_w, real_u, real_v;
            bool uvCalculated = false;
            alignas(32) float u_arr[8], v_arr[8];

            for (int i = 0; i < 8; ++i) {
                if ((maskBit & (1 << i)) == 0) continue;

                int px = x + (i % 4);
                int py = y + (i / 4);
                if (px < 0 || px >= fb.width || py < 0 || py >= fb.height) continue;

                int pixelIdx = py * fb.width + px;
                if (z_arr[i] >= fb.depthBuffer[pixelIdx]) continue;
                fb.depthBuffer[pixelIdx] = z_arr[i];

                if (!uvCalculated) {
                    inv_w = _mm256_fmadd_ps(lambda2, v_w2, _mm256_fmadd_ps(lambda1, v_w1, _mm256_mul_ps(lambda0, v_w0)));
                    __m256 w_recip = _mm256_rcp_ps(inv_w);

                    __m256 u_out = _mm256_fmadd_ps(lambda2, v_preU2, _mm256_fmadd_ps(lambda1, v_preU1, _mm256_mul_ps(lambda0, v_preU0)));
                    __m256 v_out = _mm256_fmadd_ps(lambda2, v_preV2, _mm256_fmadd_ps(lambda1, v_preV1, _mm256_mul_ps(lambda0, v_preV0)));

                    real_u = _mm256_mul_ps(u_out, w_recip);
                    real_v = _mm256_mul_ps(v_out, w_recip);

                    _mm256_store_ps(u_arr, real_u);
                    _mm256_store_ps(v_arr, real_v);
                    uvCalculated = true;
                }

                float u = u_arr[i];
                float v = v_arr[i];
                int check = ((int)(u * 20.0f) + (int)(v * 20.0f)) % 2;
                uint8_t color = check ? 220 : 80;

                float depthVal = std::max(0.0f, std::min(z_arr[i], 1.0f));
                color = (uint8_t)(color * (1.0f - depthVal));

                fb.colorBuffer[pixelIdx] = (color << 16) | (color << 8) | color;
            }
        }
    }
}

inline void RasterizeTriangleForTile(Framebuffer& fb, const MeshSoA& mesh, const MeshSoA& transformedMesh, uint32_t triIdx, const Tile& tile,
    long long& colorChanged, long long& depthChanged, long long& invalidBBoxCount, long long& trianglePassedCount, long long& maskValidCount) {
    uint32_t i0 = mesh.indices[triIdx];
    uint32_t i1 = mesh.indices[triIdx + 1];
    uint32_t i2 = mesh.indices[triIdx + 2];


    float x0 = transformedMesh.x[i0], y0 = transformedMesh.y[i0], z0 = transformedMesh.z[i0], w0 = transformedMesh.w[i0];
    float x1 = transformedMesh.x[i1], y1 = transformedMesh.y[i1], z1 = transformedMesh.z[i1], w1 = transformedMesh.w[i1];
    float x2 = transformedMesh.x[i2], y2 = transformedMesh.y[i2], z2 = transformedMesh.z[i2], w2 = transformedMesh.w[i2];

    float u0 = mesh.u[i0], v0 = mesh.v[i0];
    float u1 = mesh.u[i1], v1 = mesh.v[i1];
    float u2 = mesh.u[i2], v2 = mesh.v[i2];

    float dx01 = x1 - x0, dy01 = y1 - y0;
    float dx12 = x2 - x1, dy12 = y2 - y1;
    float dx20 = x0 - x2, dy20 = y0 - y2;

    float areaDouble = dx01 * dy20 - dx20 * dy01;

    // 自动修正三角形方向：如果是 CCW (area < 0)，则翻转为 CW
    if (areaDouble > 0.0f) {
        std::swap(x1, x2); std::swap(y1, y2); std::swap(z1, z2); std::swap(w1, w2);
        std::swap(u1, u2); std::swap(v1, v2);

        dx01 = x1 - x0; dy01 = y1 - y0;
        dx12 = x2 - x1; dy12 = y2 - y1;
        dx20 = x0 - x2; dy20 = y0 - y2;

        areaDouble = -areaDouble;
    }

    if (areaDouble >= 0.0f) return; // 面积为 0，剔除

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
        invalidBBoxCount++;
        return; // 再次确保交集有效
    }

    trianglePassedCount++;

    // 注意：TILE_SIZE 是 64（4的倍数），所以 tile.startX 也是 4的倍数。
    // 这里的按位与操作只会把 minX 对齐到 Tile 内部或边界的像素块起点，绝不会越界到左侧 Tile！
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

    for (int y = minY; y <= maxY; y += 2) {
        __m256 v_py = _mm256_add_ps(_mm256_set1_ps((float)y), v_offsetY);
        __m256 v_py_sub_y1 = _mm256_sub_ps(v_py, v_y1);
        __m256 v_py_sub_y2 = _mm256_sub_ps(v_py, v_y2);
        __m256 v_py_sub_y0 = _mm256_sub_ps(v_py, v_y0);

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
            if (maskBit == 0) continue; // 如果 8 个像素都不在三角形内，直接跳过计算

            maskValidCount++;

            //计算重心坐标 lambda
            __m256 lambda0 = _mm256_mul_ps(eval0, v_invArea);
            __m256 lambda1 = _mm256_mul_ps(eval1, v_invArea);
            __m256 lambda2 = _mm256_mul_ps(eval2, v_invArea);

            __m256 z_out = _mm256_fmadd_ps(lambda2, v_z2, _mm256_fmadd_ps(lambda1, v_z1, _mm256_mul_ps(lambda0, v_z0)));
            alignas(32) float z_arr[8];
            _mm256_store_ps(z_arr, z_out);

            //用于延迟透视校正的变量声明
            __m256 inv_w, real_u, real_v;
            bool uvCalculated = false;
            alignas(32) float u_arr[8], v_arr[8];

            //提取有效像素
            for (int i = 0; i < 8; ++i) {
                if ((maskBit & (1 << i)) == 0) continue;

                int px = x + (i % 4);
                int py = y + (i / 4);

                //防止边缘块写出 Tile 或 Framebuffer
                //这一步对于 Tiled Rendering 多线程非常重要，宁可冗余不能越界
                if (px > tile.endX || py > tile.endY || px < tile.startX || py < tile.startY) continue;

                int pixelIdx = py * fb.width + px;

                //深度测试
                if (z_arr[i] >= fb.depthBuffer[pixelIdx]) continue;
                fb.depthBuffer[pixelIdx] = z_arr[i];
                depthChanged++;

                //延迟透视校正
                //只有通过了深度测试的像素，才舍得花算力去计算 1/W 和 UV
                if (!uvCalculated) {
                    inv_w = _mm256_fmadd_ps(lambda2, v_w2, _mm256_fmadd_ps(lambda1, v_w1, _mm256_mul_ps(lambda0, v_w0)));
                    __m256 w_recip = _mm256_rcp_ps(inv_w); // 快速求倒数求回真实的 W

                    __m256 u_out = _mm256_fmadd_ps(lambda2, v_preU2, _mm256_fmadd_ps(lambda1, v_preU1, _mm256_mul_ps(lambda0, v_preU0)));
                    __m256 v_out = _mm256_fmadd_ps(lambda2, v_preV2, _mm256_fmadd_ps(lambda1, v_preV1, _mm256_mul_ps(lambda0, v_preV0)));

                    real_u = _mm256_mul_ps(u_out, w_recip);
                    real_v = _mm256_mul_ps(v_out, w_recip);

                    _mm256_store_ps(u_arr, real_u);
                    _mm256_store_ps(v_arr, real_v);
                    uvCalculated = true;
                }

                //纹理棋盘格采样
                float u = u_arr[i];
                float v = v_arr[i];
                int check = ((int)(u * 20.0f) + (int)(v * 20.0f)) % 2;
                uint8_t color = check ? 220 : 80;

                float depthVal = std::max(0.0f, std::min(z_arr[i], 1.0f));
                color = (uint8_t)(color * (1.0f - depthVal));

                fb.colorBuffer[pixelIdx] = (color << 16) | (color << 8) | color;
                colorChanged++;
            }
        }
    }
}


