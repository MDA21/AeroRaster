#pragma once
#include <iostream>
#include <vector>
#include <immintrin.h>
#include "ComputeBase.h"
#include "Mesh.h"
#include <Eigen/Dense>
#include <cmath>
#include <unordered_map>
#include "tiny_obj_loader.h"

class Model {
public:
    struct VertexKey {
        int pos_idx;
        int uv_idx;
        int norm_idx;

        bool operator==(const VertexKey& other) const {
            return pos_idx == other.pos_idx && 
                   uv_idx == other.uv_idx && 
                   norm_idx == other.norm_idx;
        }
    };

    struct VertexKeyHash {
        size_t operator()(const VertexKey& k) const {
            return std::hash<int>()(k.pos_idx) ^ 
                   (std::hash<int>()(k.uv_idx) << 1) ^ 
                   (std::hash<int>()(k.norm_idx) << 2);
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

        std::unordered_map<VertexKey, uint32_t, VertexKeyHash> uniqueVertices;

        outMesh.Reserve(attrib.vertices.size() / 3);

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                VertexKey key = { index.vertex_index, index.texcoord_index, index.normal_index };

                if (uniqueVertices.count(key) == 0) {
                    Eigen::Vector3f pos;
                    pos.x() = attrib.vertices[3 * index.vertex_index + 0];
                    pos.y() = attrib.vertices[3 * index.vertex_index + 1];
                    pos.z() = attrib.vertices[3 * index.vertex_index + 2];

                    Eigen::Vector2f uv(0.0f, 0.0f);
                    if (index.texcoord_index >= 0) {
                        uv.x() = attrib.texcoords[2 * index.texcoord_index + 0];
                        uv.y() = 1.0f - attrib.texcoords[2 * index.texcoord_index + 1];
                    }

                    Eigen::Vector3f norm(0.0f, 0.0f, 0.0f);
                    if (index.normal_index >= 0) {
                        norm.x() = attrib.normals[3 * index.normal_index + 0];
                        norm.y() = attrib.normals[3 * index.normal_index + 1];
                        norm.z() = attrib.normals[3 * index.normal_index + 2];
                    }

                    //≥ı º∑®œﬂ(0,0,0,0)
                    outMesh.PushVertex(pos, uv, norm, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f));

                    uint32_t newIndex = (uint32_t)(outMesh.GetVertexCount() - 1);
                    uniqueVertices[key] = newIndex;
                    outMesh.indices.push_back(newIndex);
                }
                else {
                    outMesh.indices.push_back(uniqueVertices[key]);
                }
            }
        }

        //Compute Tangents
        ComputeTangents(outMesh);

        outMesh.PadToAlign8();

        std::cout << "Loaded OBJ: " << filename << "\n"
            << "  Unique Vertices: " << outMesh.GetVertexCount() << "\n"
            << "  Triangles: " << outMesh.GetTriangleCount() << std::endl;

        return true;
    }

private:
    static void ComputeTangents(MeshSoA& mesh) {
        size_t indexCount = mesh.indices.size();
        size_t vertexCount = mesh.GetVertexCount();

        // Temporary arrays for accumulation
        std::vector<Eigen::Vector3f> tan1(vertexCount, Eigen::Vector3f::Zero());
        std::vector<Eigen::Vector3f> tan2(vertexCount, Eigen::Vector3f::Zero());

        for (size_t i = 0; i < indexCount; i += 3) {
            uint32_t i0 = mesh.indices[i];
            uint32_t i1 = mesh.indices[i+1];
            uint32_t i2 = mesh.indices[i+2];

            Eigen::Vector3f v0(mesh.x[i0], mesh.y[i0], mesh.z[i0]);
            Eigen::Vector3f v1(mesh.x[i1], mesh.y[i1], mesh.z[i1]);
            Eigen::Vector3f v2(mesh.x[i2], mesh.y[i2], mesh.z[i2]);

            Eigen::Vector2f w0(mesh.u[i0], mesh.v[i0]);
            Eigen::Vector2f w1(mesh.u[i1], mesh.v[i1]);
            Eigen::Vector2f w2(mesh.u[i2], mesh.v[i2]);

            float x1 = v1.x() - v0.x();
            float x2 = v2.x() - v0.x();
            float y1 = v1.y() - v0.y();
            float y2 = v2.y() - v0.y();
            float z1 = v1.z() - v0.z();
            float z2 = v2.z() - v0.z();

            float s1 = w1.x() - w0.x();
            float s2 = w2.x() - w0.x();
            float t1 = w1.y() - w0.y();
            float t2 = w2.y() - w0.y();

            float r = 1.0f / (s1 * t2 - s2 * t1);
            if (std::isinf(r) || std::isnan(r)) r = 0.0f; // Handle degenerate UVs

            Eigen::Vector3f sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
            Eigen::Vector3f tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

            tan1[i0] += sdir;
            tan1[i1] += sdir;
            tan1[i2] += sdir;

            tan2[i0] += tdir;
            tan2[i1] += tdir;
            tan2[i2] += tdir;
        }

        for (size_t a = 0; a < vertexCount; a++) {
            Eigen::Vector3f n(mesh.nx[a], mesh.ny[a], mesh.nz[a]);
            Eigen::Vector3f t = tan1[a];

            //Gram-Schmidt orthogonalize
            Eigen::Vector3f tangent = (t - n * n.dot(t)).normalized();
            
            //Calculate handedness
            float w = (n.cross(t).dot(tan2[a]) < 0.0f) ? -1.0f : 1.0f;

            mesh.tx[a] = tangent.x();
            mesh.ty[a] = tangent.y();
            mesh.tz[a] = tangent.z();
            mesh.tw[a] = w;
        }
    }
};
