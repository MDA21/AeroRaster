#pragma once
#include <iostream>
#include <vector>
#include <immintrin.h>
#include "MathBase.h"
#include <Eigen/Dense>
#include <cmath>
#include <unordered_map>
#include "tiny_obj_loader.h"

class Model {
public:
    // 这是一个标准的哈希组合器，用于将 (pos_idx, uv_idx) 组合成一个 key
    struct VertexKey {
        int pos_idx;
        int uv_idx;
        int norm_idx;

        bool operator==(const VertexKey& other) const {
            return pos_idx == other.pos_idx && uv_idx == other.uv_idx;
        }
    };

    struct VertexKeyHash {
        size_t operator()(const VertexKey& k) const {
            // 简单的哈希组合
            return std::hash<int>()(k.pos_idx) ^ (std::hash<int>()(k.uv_idx) << 1);
        }
    };

    static bool LoadObj(const std::string& filename, MeshSoA& outMesh) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(), nullptr, true);

        if (!warn.empty()) std::cout << "TinyObj Warn: " << warn << std::endl;
        if (!err.empty()) std::cerr << "TinyObj Err: " << err << std::endl;
        if (!ret) return false;

        //去重容器
        // Map: (OBJ原始索引组合) -> (MeshSoA中的新索引)
        std::unordered_map<VertexKey, uint32_t, VertexKeyHash> uniqueVertices;

        //预估一下大小，避免频繁 realloc
        outMesh.Reserve(attrib.vertices.size() / 3);

        //遍历所有形状和面
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                //构建 Key
                VertexKey key = { index.vertex_index, index.texcoord_index };

                // 检查是否已经是已知的顶点组合
                if (uniqueVertices.count(key) == 0) {
                    //是新顶点，添加到 SoA

                    // 读取位置
                    Eigen::Vector3f pos;
                    pos.x() = attrib.vertices[3 * index.vertex_index + 0];
                    pos.y() = attrib.vertices[3 * index.vertex_index + 1];
                    pos.z() = attrib.vertices[3 * index.vertex_index + 2];

                    // 读取纹理坐标 (OBJ 通常是 Y 轴向上的，而纹理通常是 Y 轴向下的，看情况可能需要 1-v)
                    Eigen::Vector2f uv(0.0f, 0.0f);
                    if (index.texcoord_index >= 0) {
                        uv.x() = attrib.texcoords[2 * index.texcoord_index + 0];
                        uv.y() = 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]; // Flip V usually
                    }

                    // 添加到 MeshSoA
                    outMesh.PushVertex(pos, uv);

                    // 记录新索引
                    uint32_t newIndex = (uint32_t)(outMesh.GetVertexCount() - 1);
                    uniqueVertices[key] = newIndex;
                    outMesh.indices.push_back(newIndex);
                }
                else {
                    //是旧顶点，复用索引
                    outMesh.indices.push_back(uniqueVertices[key]);
                }
            }
        }

        // 4. 关键步骤：AVX2 Padding
        outMesh.PadToAlign8();

        std::cout << "Loaded OBJ: " << filename << "\n"
            << "  Unique Vertices: " << outMesh.GetVertexCount() << "\n"
            << "  Triangles: " << outMesh.GetTriangleCount() << std::endl;

        return true;
    }
};