#pragma once
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <cmath>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct alignas(32) MeshSoA{
    std::vector<float> x, y, z, w;
    std::vector<float> u, v;
    std::vector<float> r, g, b, a;
    std::vector<float> nx, ny, nz;
    std::vector<uint32_t> indices;
	
    size_t original_count = 0;

	size_t GetVertexCount() const { return x.size(); }

	size_t GetTriangleCount() const { return indices.size() / 3; }

    void Reserve(size_t n) {
        x.reserve(n); y.reserve(n); z.reserve(n); w.reserve(n);
        u.reserve(n); v.reserve(n);
        r.reserve(n); g.reserve(n); b.reserve(n); a.reserve(n);
        nx.reserve(n); ny.reserve(n); nz.reserve(n);
	}

    void PushVertex(const Eigen::Vector3f& pos, const Eigen::Vector2f& uv) {
        x.push_back(pos.x());
        y.push_back(pos.y());
        z.push_back(pos.z());
        w.push_back(1.0f);
        u.push_back(uv.x());
        v.push_back(uv.y());
    }

    //填充到8的倍数
    void PadToAlign8() {
        size_t original_size = x.size();
        size_t padded_size = (original_size + 7) & ~7;
        size_t padding_count = padded_size - original_size;

        if (padding_count > 0) {
            for (size_t i = 0; i < padding_count; ++i) {
                x.push_back(0.0f); y.push_back(0.0f); z.push_back(0.0f);
                w.push_back(0.0f); //W=0 意味着它会在透视除法产生 NaN/Inf，或者被视锥裁剪掉
                u.push_back(0.0f); v.push_back(0.0f);
            }
            std::cout << "Mesh padded with " << padding_count << " dummy vertices." << std::endl;
        }
    }
};

using Matrix4f = Eigen::Matrix4f;
using Vector4f = Eigen::Vector4f;
using Vector3f = Eigen::Vector3f;

void TransformVerticesAVX2(const MeshSoA& in, MeshSoA& out, const Matrix4f& mat) {
    size_t count = in.GetVertexCount();

    //x分量
	__m256 m00 = _mm256_set1_ps(mat(0, 0));
    __m256 m10 = _mm256_set1_ps(mat(1, 0));
	__m256 m20 = _mm256_set1_ps(mat(2, 0));
    __m256 m30 = _mm256_set1_ps(mat(3, 0));

	//y分量
	__m256 m01 = _mm256_set1_ps(mat(0, 1));
	__m256 m11 = _mm256_set1_ps(mat(1, 1));
	__m256 m21 = _mm256_set1_ps(mat(2, 1));
	__m256 m31 = _mm256_set1_ps(mat(3, 1));

    //z分量
	__m256 m02 = _mm256_set1_ps(mat(0, 2));
	__m256 m12 = _mm256_set1_ps(mat(1, 2));
	__m256 m22 = _mm256_set1_ps(mat(2, 2));
    __m256 m32 = _mm256_set1_ps(mat(3, 2));

	//w分量
	__m256 m03 = _mm256_set1_ps(mat(0, 3));
	__m256 m13 = _mm256_set1_ps(mat(1, 3));
	__m256 m23 = _mm256_set1_ps(mat(2, 3));
	__m256 m33 = _mm256_set1_ps(mat(3, 3));

    for (size_t i = 0; i < count; i += 8) {

        __m256 vx = _mm256_loadu_ps(&in.x[i]);
		__m256 vy = _mm256_loadu_ps(&in.y[i]);
		__m256 vz = _mm256_loadu_ps(&in.z[i]);

        //计算x
        __m256 res_x = m30;
		res_x = _mm256_fmadd_ps(vx, m00, res_x);
		res_x = _mm256_fmadd_ps(vy, m10, res_x);
		res_x = _mm256_fmadd_ps(vz, m20, res_x);

		//计算y
		__m256 res_y = m31;
		res_y = _mm256_fmadd_ps(vx, m01, res_y);
		res_y = _mm256_fmadd_ps(vy, m11, res_y);
		res_y = _mm256_fmadd_ps(vz, m21, res_y);

		//计算z
        __m256 res_z = m32;
		res_z = _mm256_fmadd_ps(vx, m02, res_z);
		res_z = _mm256_fmadd_ps(vy, m12, res_z);
		res_z = _mm256_fmadd_ps(vz, m22, res_z);

        //计算w
		__m256 res_w = m33;
		res_w = _mm256_fmadd_ps(vx, m03, res_w);
		res_w = _mm256_fmadd_ps(vy, m13, res_w);
        res_w = _mm256_fmadd_ps(vz, m23, res_w);

        //存储结果
		_mm256_storeu_ps(&out.x[i], res_x);
		_mm256_storeu_ps(&out.y[i], res_y);
		_mm256_storeu_ps(&out.z[i], res_z);
		_mm256_storeu_ps(&out.w[i], res_w);
    }
}

void PerspectiveDivideAVX2(MeshSoA& mesh) {
	size_t count = mesh.GetVertexCount();

    for(size_t i = 0; i < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&mesh.x[i]);
        __m256 y = _mm256_loadu_ps(&mesh.y[i]);
        __m256 z = _mm256_loadu_ps(&mesh.z[i]);
        __m256 w = _mm256_loadu_ps(&mesh.w[i]);
        //计算1/w
        __m256 inv_w = _mm256_div_ps(_mm256_set1_ps(1.0f), w);
        //计算透视除法后的坐标
        x = _mm256_mul_ps(x, inv_w);
        y = _mm256_mul_ps(y, inv_w);
        z = _mm256_mul_ps(z, inv_w);
        //存储结果
        _mm256_storeu_ps(&mesh.x[i], x);
        _mm256_storeu_ps(&mesh.y[i], y);
        _mm256_storeu_ps(&mesh.z[i], z);
		_mm256_store_ps(&mesh.w[i], inv_w); //W分量存储1/w，可以在后续的光栅化阶段用于深度测试等用途
	}
}

void ViewportTransformAVX2(MeshSoA& mesh, float width, float height) {
	size_t count = mesh.GetVertexCount();
    __m256 half_width = _mm256_set1_ps(width * 0.5f);
    __m256 half_height = _mm256_set1_ps(height * 0.5f);

    for(size_t i = 0; i < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&mesh.x[i]);
        __m256 y = _mm256_loadu_ps(&mesh.y[i]);
        __m256 z = _mm256_loadu_ps(&mesh.z[i]);

        //视口变换
        x = _mm256_fmadd_ps(x, half_width, half_width);   // x' = (x + 1) * (width / 2)
        y = _mm256_fnmadd_ps(y, half_height, half_height); // y' = (1 - y) * (height / 2)
        //存储结果
        _mm256_storeu_ps(&mesh.x[i], x);
        _mm256_storeu_ps(&mesh.y[i], y);
        _mm256_storeu_ps(&mesh.z[i], z); // Z值不变，仍然在[-1,1]范围内，可以用于深度测试
	}
}