#include "../include/model.h"
#include "../include/ComputeBase.h"
#include "../include/Camera.h"
#include "../include/image.h"
#include "../include/Tile.h"
#include "../include/Mesh.h"
#include "../include/JobSystem.h"
#include <vector>
#include <immintrin.h>
#include <iostream>
#include <Eigen/Dense>
#include <cstdio>
#include <limits>
#include <chrono>

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

	JobSystem jobSystem;
	std::cout << "[Init] JobSystem started with " << jobSystem.GetThreadCount() << " threads." << std::endl;

	const int MAX_FRAMES = 10;

	std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

	while (frameCount < MAX_FRAMES) {

        fb.Clear();
		Matrix4f mvp = Camera::ComputeMVP(width, height, frameCount);

		TransformVerticesAVX2(mesh, transformedMesh, mvp);

		PerspectiveDivideAVX2(transformedMesh);

		ViewportTransformAVX2(transformedMesh, width, height);

        
		
		tileGrid.ClearBins();
		tileGrid.BinTriangles(mesh, transformedMesh, width, height);

		std::vector<uint32_t> acticeTileIndices;
		acticeTileIndices.reserve(tileGrid.tiles.size());
		for(size_t i = 0; i < tileGrid.tiles.size(); ++i) {
			if (!tileGrid.tiles[i].triangleIndices.empty()) {
				acticeTileIndices.push_back((uint32_t)i);
			}
		}

		for (auto& tile : tileGrid.tiles) {
			if (tile.triangleIndices.empty()) continue;
            for (uint32_t triIdx : tile.triangleIndices) {
                RasterizeTriangleForTile(fb, mesh, transformedMesh, triIdx, tile);
            }
		}

		

        char filename[64];
        std::snprintf(filename, sizeof(filename), "output_frame_%02d.ppm", frameCount);

        fb.SaveToPPM(filename);

        std::cout << "Rendered frame " << frameCount << " to " << filename << std::endl;

        frameCount++;
	}
	std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedSeconds = endTime - startTime;
	std::cout << "Total rendering time for " << MAX_FRAMES << " frames: " << elapsedSeconds.count() << " seconds." << std::endl;
	return 0;
}
