#include "../include/model.h"
#include "../include/ComputeBase.h"
#include "../include/Camera.h"
#include <vector>
#include <immintrin.h>
#include <iostream>
#include <Eigen/Dense>
#include <cstdio>

int main() {
	MeshSoA mesh;
	if (!Model::LoadObj("F:/VSproject/SIMDRenderer/resources/diablo3_pose/diablo3_pose.obj", mesh)) {
		std::cerr << "Failed to load model." << std::endl;
		return -1;
	}

	MeshSoA transformedMesh;
	transformedMesh.Resize(mesh.GetVertexCount());

	int frameCount = 0;
	float width = 800.0f;
	float height = 600.0f;
	Framebuffer fb(width, height);

	const int MAX_FRAMES = 10;

	while (frameCount < MAX_FRAMES) {

        fb.Clear();
		Matrix4f mvp = Camera::ComputeMVP(width, height, frameCount);

		TransformVerticesAVX2(mesh, transformedMesh, mvp);

		PerspectiveDivideAVX2(transformedMesh);

		ViewportTransformAVX2(transformedMesh, width, height);

        size_t indexCount = mesh.indices.size();
        for (size_t i = 0; i < indexCount; i += 3) {
            uint32_t i0 = mesh.indices[i];
            uint32_t i1 = mesh.indices[i + 1];
            uint32_t i2 = mesh.indices[i + 2];

            float x0 = transformedMesh.x[i0], y0 = transformedMesh.y[i0], z0 = transformedMesh.z[i0], w0 = transformedMesh.w[i0];
            float x1 = transformedMesh.x[i1], y1 = transformedMesh.y[i1], z1 = transformedMesh.z[i1], w1 = transformedMesh.w[i1];
            float x2 = transformedMesh.x[i2], y2 = transformedMesh.y[i2], z2 = transformedMesh.z[i2], w2 = transformedMesh.w[i2];

            float u0 = mesh.u[i0], v0 = mesh.v[i0];
            float u1 = mesh.u[i1], v1 = mesh.v[i1];
            float u2 = mesh.u[i2], v2 = mesh.v[i2];

            int minX = std::max(0, (int)std::floor(std::min({ x0, x1, x2 })));
            int maxX = std::min(fb.width - 1, (int)std::ceil(std::max({ x0, x1, x2 })));
            int minY = std::max(0, (int)std::floor(std::min({ y0, y1, y2 })));
            int maxY = std::min(fb.height - 1, (int)std::ceil(std::max({ y0, y1, y2 })));

            if (minX > maxX || minY > maxY) continue;

            RasterizeTriangleAVX(fb,
                x0, y0, z0, x1, y1, z1, x2, y2, z2,
                w0, w1, w2, u0, u1, u2, v0, v1, v2,
                minX, maxX, minY, maxY);
        }

        char filename[64];
        std::snprintf(filename, sizeof(filename), "output_frame_%02d.ppm", frameCount);

        fb.SaveToPPM(filename);

        std::cout << "Rendered frame " << frameCount << " to " << filename << std::endl;

        frameCount++;
	}

}
