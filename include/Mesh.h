#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <immintrin.h>
#include <Eigen/Dense>

using Matrix4f = Eigen::Matrix4f;
using Vector4f = Eigen::Vector4f;
using Vector3f = Eigen::Vector3f;

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

    void Resize(size_t n) {
        x.resize(n); y.resize(n); z.resize(n); w.resize(n);
        u.resize(n); v.resize(n);
        r.resize(n); g.resize(n); b.resize(n); a.resize(n);
        nx.resize(n); ny.resize(n); nz.resize(n);
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
