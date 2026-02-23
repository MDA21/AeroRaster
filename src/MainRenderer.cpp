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

    TextureSoA diffuseMap("F:/VSproject/SIMDRenderer/resources/diablo3_pose/diablo3_pose_diffuse.tga");
    TextureSoA normalMap("F:/VSproject/SIMDRenderer/resources/diablo3_pose/diablo3_pose_nm.tga");
    TextureSoA specMap("F:/VSproject/SIMDRenderer/resources/diablo3_pose/diablo3_pose_spec.tga");

	MeshSoA transformedMesh;
	transformedMesh.Resize(mesh.GetVertexCount());

	int frameCount = 0;
	float width = 800.0f;
	float height = 600.0f;
	Framebuffer fb(width, height);
	
	JobSystem jobSystem(std::thread::hardware_concurrency());
	std::cout << "[Init] JobSystem started with " << jobSystem.GetThreadCount() << " threads." << std::endl;
	TileGrid tileGrid(width, height, 64);
	//TileGrid tileGrid(width, height, 64, jobSystem.GetThreadCount());

	double geomTime = 0.0;
	double rasterTime = 0.0;
	const int MAX_FRAMES = 10;
	Vector3f eye(0.0f, 0.0f, 3.0f);
	Vector3f target(0.0f, 0.0f, 0.0f);
	Vector3f up(0.0f, 1.0f, 0.0f);
	float fovY = 45.0f;
	float aspect = width / height;
	float zNear = 0.1f;
	float zFar = 100.0f;

	
	Matrix4f view = Camera::GetViewMatrix(eye, target, up);
	Matrix4f projection = Camera::GetProjectionMatrix(fovY, aspect, zNear, zFar);


	std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

	while (frameCount < MAX_FRAMES) {

        fb.Clear();
		
        float angle = frameCount * 0.5f;
		Matrix4f model = Camera::GetModelMatrix(angle, Vector3f(0.0f, 0.0f, 0.0f));

		Matrix4f mvp = projection * view * model;

		//几何处理时间
		auto t_geom_start = std::chrono::high_resolution_clock::now();

		ProcessGeometryAVX2(mesh, transformedMesh, mvp, model, width, height);

        
		
		tileGrid.ClearBins();
		tileGrid.BinTriangles(mesh, transformedMesh, width, height);
		//tileGrid.BinTrianglesMultithreaded(mesh, transformedMesh, width, height, jobSystem);

		auto t_geom_end = std::chrono::high_resolution_clock::now();
		geomTime += std::chrono::duration<double>(t_geom_end - t_geom_start).count();

		std::vector<uint32_t> acticeTileIndices;
		acticeTileIndices.reserve(tileGrid.tiles.size());
		for(size_t i = 0; i < tileGrid.tiles.size(); ++i) {
			if (!tileGrid.tiles[i].triangleIndices.empty()) {
				acticeTileIndices.push_back((uint32_t)i);
			}
		}

		//光栅化时间
		auto t_raster_start = std::chrono::high_resolution_clock::now();

        Eigen::Vector3f lightDir = Eigen::Vector3f(1.0f, 1.0f, 1.0f).normalized();

		jobSystem.Dispatch((uint32_t)acticeTileIndices.size(), [&](uint32_t index) {
			uint32_t tileIdx = acticeTileIndices[index];
			const Tile& tile = tileGrid.tiles[tileIdx];
			for (uint32_t triIdx : tile.triangleIndices) {
				RasterizeTriangleForTile(fb, mesh, transformedMesh, triIdx, tile, &diffuseMap, &normalMap, &specMap, lightDir, eye);
			}
		});

		auto t_raster_end = std::chrono::high_resolution_clock::now();
		rasterTime += std::chrono::duration<double>(t_raster_end - t_raster_start).count();

        char filename[64];
        std::snprintf(filename, sizeof(filename), "output_frame_%02d.ppm", frameCount);
        fb.SaveToPPM(filename);
        std::cout << "Rendered frame " << frameCount << " to " << filename << std::endl;

        frameCount++;
	}
	std::cout << "Geometry & Binning Time: " << geomTime << " s" << std::endl;
	std::cout << "Rasterization Time:    " << rasterTime << " s" << std::endl;
	std::cout << "Rendered " << frameCount << " frames" << std::endl;
	return 0;
}
