#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <immintrin.h>
#include <Eigen/Dense>
#include <string>
#include "stb_image.h"

using Matrix4f = Eigen::Matrix4f;
using Vector4f = Eigen::Vector4f;
using Vector3f = Eigen::Vector3f;

struct TextureSoA {
    int width = 0, height = 0, channels = 0;
    std::vector<float> r, g, b, a;

    TextureSoA() = default;

    TextureSoA(const std::string& filename) {
        unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 4);
        if (!data) {
            std::cerr << "Failed to load texture: " << filename << std::endl;
            return;
        }

        std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;

        size_t size = width * height;
        r.resize(size);
        g.resize(size);
        b.resize(size);
        a.resize(size);

        // Convert to SoA and normalize to [0, 1]
        for (size_t i = 0; i < size; ++i) {
            r[i] = data[i * 4 + 0] / 255.0f;
            g[i] = data[i * 4 + 1] / 255.0f;
            b[i] = data[i * 4 + 2] / 255.0f;
            a[i] = data[i * 4 + 3] / 255.0f;
        }

        stbi_image_free(data);
    }

    //采样（标量）
    Eigen::Vector4f Sample(float u, float v) const {
        if (r.empty()) return Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f); // Pink error

        u = u - std::floor(u);
        v = v - std::floor(v);

        int x = std::max(0, std::min((int)(u * width), width - 1));
        int y = std::max(0, std::min((int)(v * height), height - 1));
        int idx = y * width + x;

        return Eigen::Vector4f(r[idx], g[idx], b[idx], a[idx]);
    }

	//采样（AVX2版本）
    void SampleAVX2(__m256 u, __m256 v, __m256& out_r, __m256& out_g, __m256& out_b, __m256& out_a) const {
        if (r.empty()) {
            out_r = _mm256_set1_ps(1.0f);
            out_g = _mm256_setzero_ps();
            out_b = _mm256_set1_ps(1.0f);
            out_a = _mm256_set1_ps(1.0f);
            return;
        }

        //Wrap UVs: u - floor(u)
        u = _mm256_sub_ps(u, _mm256_floor_ps(u));
        v = _mm256_sub_ps(v, _mm256_floor_ps(v));

        // Scale by dimensions
        __m256 w = _mm256_set1_ps((float)width);
        __m256 h = _mm256_set1_ps((float)height);
        
        __m256 x_f = _mm256_mul_ps(u, w);
        __m256 y_f = _mm256_mul_ps(v, h);

        // Convert to int
        __m256i x_i = _mm256_cvttps_epi32(x_f);
        __m256i y_i = _mm256_cvttps_epi32(y_f);

        //Clamp
        __m256i max_x = _mm256_set1_epi32(width - 1);
        __m256i max_y = _mm256_set1_epi32(height - 1);
        __m256i zero = _mm256_setzero_si256();

        x_i = _mm256_max_epi32(zero, _mm256_min_epi32(x_i, max_x));
        y_i = _mm256_max_epi32(zero, _mm256_min_epi32(y_i, max_y));

        //Calculate index: y * width + x
        __m256i width_i = _mm256_set1_epi32(width);
        __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(y_i, width_i), x_i);

        //Gather
        out_r = _mm256_i32gather_ps(r.data(), idx, 4);
        out_g = _mm256_i32gather_ps(g.data(), idx, 4);
        out_b = _mm256_i32gather_ps(b.data(), idx, 4);
        out_a = _mm256_i32gather_ps(a.data(), idx, 4);
    }
};


struct alignas(32) MeshSoA{
    std::vector<float> x, y, z, w;
    std::vector<float> u, v;
    std::vector<float> nx, ny, nz;
    std::vector<float> tx, ty, tz, tw; // Tangent + Handedness
    std::vector<float> wx, wy, wz;     // World Position
    std::vector<uint32_t> indices;
	
    size_t original_count = 0;

	size_t GetVertexCount() const { return x.size(); }

	size_t GetTriangleCount() const { return indices.size() / 3; }

    void Reserve(size_t n) {
        x.reserve(n); y.reserve(n); z.reserve(n); w.reserve(n);
        u.reserve(n); v.reserve(n);
        nx.reserve(n); ny.reserve(n); nz.reserve(n);
        tx.reserve(n); ty.reserve(n); tz.reserve(n); tw.reserve(n);
        wx.reserve(n); wy.reserve(n); wz.reserve(n);
	}

    void Resize(size_t n) {
        x.resize(n); y.resize(n); z.resize(n); w.resize(n);
        u.resize(n); v.resize(n);
        nx.resize(n); ny.resize(n); nz.resize(n);
        tx.resize(n); ty.resize(n); tz.resize(n); tw.resize(n);
        wx.resize(n); wy.resize(n); wz.resize(n);
    }


    void PushVertex(const Eigen::Vector3f& pos, const Eigen::Vector2f& uv, const Eigen::Vector3f& norm, const Eigen::Vector4f& tangent) {
        x.push_back(pos.x());
        y.push_back(pos.y());
        z.push_back(pos.z());
        w.push_back(1.0f);
        u.push_back(uv.x());
        v.push_back(uv.y());
        nx.push_back(norm.x());
        ny.push_back(norm.y());
        nz.push_back(norm.z());
        tx.push_back(tangent.x());
        ty.push_back(tangent.y());
        tz.push_back(tangent.z());
        tw.push_back(tangent.w());
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
                nx.push_back(0.0f); ny.push_back(0.0f); nz.push_back(0.0f);
                tx.push_back(0.0f); ty.push_back(0.0f); tz.push_back(0.0f); tw.push_back(0.0f);
                wx.push_back(0.0f); wy.push_back(0.0f); wz.push_back(0.0f);
            }
            std::cout << "Mesh padded with " << padding_count << " dummy vertices." << std::endl;
        }
    }
};
