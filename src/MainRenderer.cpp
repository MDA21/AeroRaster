#include "../include/model.h"
#include "../include/ComputeBase.h"
#include "../include/Camera.h"
#include "../include/image.h"
#include "../include/Tile.h"
#include "../include/Mesh.h"
#include <vector>
#include <immintrin.h>
#include <iostream>
#include <Eigen/Dense>
#include <cstdio>
#include <limits>

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
	TileGrid tileGrid(width, height, 64);

	const int MAX_FRAMES = 10;

	while (frameCount < MAX_FRAMES) {

        fb.Clear();
		Matrix4f mvp = Camera::ComputeMVP(width, height, frameCount);

		TransformVerticesAVX2(mesh, transformedMesh, mvp);

		PerspectiveDivideAVX2(transformedMesh);

		ViewportTransformAVX2(transformedMesh, width, height);

        long long colorChanged = 0;
        long long depthChanged = 0;
        long long invalidBBoxCount = 0;
        long long nonFiniteCount = 0;
        long long trianglePassedCount = 0;
        long long maskValidCount = 0;
        float minTx = std::numeric_limits<float>::max();
        float maxTx = std::numeric_limits<float>::lowest();
        float minTy = std::numeric_limits<float>::max();
        float maxTy = std::numeric_limits<float>::lowest();
        float minTz = std::numeric_limits<float>::max();
        float maxTz = std::numeric_limits<float>::lowest();
        float minTw = std::numeric_limits<float>::max();
        float maxTw = std::numeric_limits<float>::lowest();

        size_t vCount = transformedMesh.GetVertexCount();
        for (size_t i = 0; i < vCount; ++i) {
            float x = transformedMesh.x[i];
            float y = transformedMesh.y[i];
            float z = transformedMesh.z[i];
            float w = transformedMesh.w[i];

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w)) {
                nonFiniteCount++;
            }

            if (x < minTx) minTx = x;
            if (x > maxTx) maxTx = x;
            if (y < minTy) minTy = y;
            if (y > maxTy) maxTy = y;
            if (z < minTz) minTz = z;
            if (z > maxTz) maxTz = z;
            if (w < minTw) minTw = w;
            if (w > maxTw) maxTw = w;
        }
		
		tileGrid.ClearBins();
		tileGrid.BinTriangles(mesh, transformedMesh, width, height);

		for (auto& tile : tileGrid.tiles) {
			if (tile.triangleIndices.empty()) continue;
			for (uint32_t triIdx : tile.triangleIndices) {
				RasterizeTriangleForTile(fb, mesh, transformedMesh, triIdx, tile, colorChanged, depthChanged, invalidBBoxCount, trianglePassedCount, maskValidCount);
			}
		}

        std::cout
            << "Stats frame " << frameCount
            << " nonFinite=" << nonFiniteCount
            << " invalidBBox=" << invalidBBoxCount
            << " triPassed=" << trianglePassedCount
            << " maskValid=" << maskValidCount
            << " colorChanged=" << colorChanged
            << " depthChanged=" << depthChanged
            << " x=[" << minTx << "," << maxTx << "]"
            << " y=[" << minTy << "," << maxTy << "]"
            << " z=[" << minTz << "," << maxTz << "]"
            << " w=[" << minTw << "," << maxTw << "]"
            << std::endl;

        char filename[64];
        std::snprintf(filename, sizeof(filename), "output_frame_%02d.ppm", frameCount);

        fb.SaveToPPM(filename);

        std::cout << "Rendered frame " << frameCount << " to " << filename << std::endl;

        frameCount++;
	}

}
