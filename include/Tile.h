#pragma once
#include "Mesh.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>
#include <cmath>
#include "JobSystem.h"

struct alignas(64) Tile {
	int startX, startY;
	int endX, endY;

	std::vector<uint32_t> triangleIndices; //存储落在这个Tile内的三角形索引

	Tile() : startX(0), startY(0), endX(0), endY(0) {}

	void Reset() {
		triangleIndices.clear();
		//不释放内存，保持capacity防止后续的malloc竞争
	}
};

class TileGrid {
public:
	int TILE_SIZE = 64;
	int width, height;
	int numTilesX, numTilesY;
	uint32_t numThreads;

	std::vector<Tile> tiles;

	//线程本地桶[threadId][tileIdx][indices]
	std::vector<std::vector<std::vector<uint32_t>>> threadLocalBins;

	TileGrid(int w, int h, int size) {
		width = w;
		height = h;
		TILE_SIZE = size;
		numTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
		numTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;

		tiles.resize(numTilesX * numTilesY);
		threadLocalBins.resize(numThreads);

		for (int y = 0; y < numTilesY; ++y) {
			for (int x = 0; x < numTilesX; ++x) {
				int idx = y * numTilesX + x;
				tiles[idx].startX = x * TILE_SIZE;
				tiles[idx].startY = y * TILE_SIZE;
				tiles[idx].endX = std::min((x + 1) * TILE_SIZE - 1, width - 1);
				tiles[idx].endY = std::min((y + 1) * TILE_SIZE - 1, height - 1);

				tiles[idx].triangleIndices.reserve(128);
			}
		}
	}

	TileGrid(int w, int h, int size, uint32_t threads) {
		width = w;
		height = h;
		TILE_SIZE = size;
		numThreads = threads;
		numTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
		numTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;

		tiles.resize(numTilesX * numTilesY);
		threadLocalBins.resize(numThreads);

		for (int y = 0; y < numTilesY; ++y) {
			for (int x = 0; x < numTilesX; ++x) {
				int idx = y * numTilesX + x;
				tiles[idx].startX = x * TILE_SIZE;
				tiles[idx].startY = y * TILE_SIZE;
				tiles[idx].endX = std::min((x + 1) * TILE_SIZE - 1, width - 1);
				tiles[idx].endY = std::min((y + 1) * TILE_SIZE - 1, height - 1);

				tiles[idx].triangleIndices.reserve(128);
			}
		}

		for (uint32_t t = 0; t < numThreads; ++t) {
			threadLocalBins[t].resize(numTilesX * numTilesY);
			for (int i = 0; i < numTilesX * numTilesY; ++i) {
				threadLocalBins[t][i].reserve(64); //预留一点空间防扩容
			}
		}
	}

	void ClearBins() {
		for (Tile& tile : tiles) {
			tile.Reset();
		}
		for (uint32_t t = 0; t < numThreads; ++t) {
			for (auto& bin : threadLocalBins[t]) {
				bin.clear();
			}
		}
	}

	void BinTriangles(const MeshSoA& mesh, const MeshSoA& transformedMesh, int width, int height) {
		size_t indexCount = mesh.indices.size();

		//提取裸指针 (Raw Pointers)。
		//告诉编译器这些内存绝对连续，消灭一切封装开销。
		const uint32_t* __restrict indices = mesh.indices.data();
		const float* __restrict vx = transformedMesh.x.data();
		const float* __restrict vy = transformedMesh.y.data();

		int w_minus_1 = width - 1;
		int h_minus_1 = height - 1;

		for (size_t i = 0; i < indexCount; i += 3) {
			uint32_t i0 = indices[i];
			uint32_t i1 = indices[i + 1];
			uint32_t i2 = indices[i + 2];

			float x0 = vx[i0], y0 = vy[i0];
			float x1 = vx[i1], y1 = vy[i1];
			float x2 = vx[i2], y2 = vy[i2];

			//纯标量极速 Min/Max，干掉 std::initializer_list
			//编译器会直接将其翻译为极速的 minss 和 maxss 指令！
			float minX_f = std::min(x0, std::min(x1, x2));
			float maxX_f = std::max(x0, std::max(x1, x2));
			float minY_f = std::min(y0, std::min(y1, y2));
			float maxY_f = std::max(y0, std::max(y1, y2));

			//干掉致命的 std::floor 和 std::ceil 库函数调用！
			//C++ 浮点数强转 int 相当于直接截断向下取整 (只需 1 个 Cycle)
			//对于 maxX_f，强转 int 后直接 +1，完美替代极慢的 std::ceil
			int triMinX = std::max(0, (int)minX_f);
			int triMaxX = std::min(w_minus_1, (int)maxX_f + 1);
			int triMinY = std::max(0, (int)minY_f);
			int triMaxY = std::min(h_minus_1, (int)maxY_f + 1);

			int tileStartX = triMinX / TILE_SIZE;
			int tileEndX = triMaxX / TILE_SIZE;
			int tileStartY = triMinY / TILE_SIZE;
			int tileEndY = triMaxY / TILE_SIZE;

			for (int ty = tileStartY; ty <= tileEndY; ++ty) {
				for (int tx = tileStartX; tx <= tileEndX; ++tx) {
					int tileIdx = ty * numTilesX + tx;
					tiles[tileIdx].triangleIndices.push_back(i);
				}
			}
		}
	}

	void BinTrianglesMultithreaded(const MeshSoA& mesh, const MeshSoA& transformedMesh, int width, int height, JobSystem& js) {
		uint32_t numTriangles = (uint32_t)(mesh.indices.size() / 3);
		uint32_t chunkSize = (numTriangles + numThreads - 1) / numThreads;

		//MAP (私有分配，彻底无锁)
		//按三角形维度并行，线程独占自己的 threadLocalBins[threadId]
		js.Dispatch(numThreads, [&](uint32_t threadId) {
			uint32_t startTri = threadId * chunkSize;
			uint32_t endTri = std::min(startTri + chunkSize, numTriangles);

			for (uint32_t tri = startTri; tri < endTri; ++tri) {
				uint32_t i = tri * 3;
				uint32_t i0 = mesh.indices[i];
				uint32_t i1 = mesh.indices[i + 1];
				uint32_t i2 = mesh.indices[i + 2];

				float x0 = transformedMesh.x[i0], y0 = transformedMesh.y[i0];
				float x1 = transformedMesh.x[i1], y1 = transformedMesh.y[i1];
				float x2 = transformedMesh.x[i2], y2 = transformedMesh.y[i2];

				//彻底剔除 std::initializer_list，强制编译器生成极速 maxss / minss 标量指令
				float minX_f = std::min(x0, std::min(x1, x2));
				float maxX_f = std::max(x0, std::max(x1, x2));
				float minY_f = std::min(y0, std::min(y1, y2));
				float maxY_f = std::max(y0, std::max(y1, y2));

				int triMinX = std::max(0, (int)std::floor(minX_f));
				int triMaxX = std::min(width - 1, (int)std::ceil(maxX_f));
				int triMinY = std::max(0, (int)std::floor(minY_f));
				int triMaxY = std::min(height - 1, (int)std::ceil(maxY_f));

				int tileStartX = triMinX / TILE_SIZE;
				int tileEndX = triMaxX / TILE_SIZE;
				int tileStartY = triMinY / TILE_SIZE;
				int tileEndY = triMaxY / TILE_SIZE;

				for (int ty = tileStartY; ty <= tileEndY; ++ty) {
					for (int tx = tileStartX; tx <= tileEndX; ++tx) {
						int tileIdx = ty * numTilesX + tx;
						// 写入线程私有桶，绝对安全
						threadLocalBins[threadId][tileIdx].push_back(i);
					}
				}
			}
			});

		//REDUCE (极速拼装)
		//按 Tile 维度并行，每个线程独占一个 Tile 的拼装任务

		uint32_t totalTiles = numTilesX * numTilesY;
		js.Dispatch(totalTiles, [&](uint32_t tileIdx) {
			//先统计容量，避免中途扩容拷贝
			size_t totalNeeded = 0;
			for (uint32_t t = 0; t < numThreads; ++t) {
				totalNeeded += threadLocalBins[t][tileIdx].size();
			}

			tiles[tileIdx].triangleIndices.reserve(totalNeeded);

			//内存块暴力插入
			for (uint32_t t = 0; t < numThreads; ++t) {
				auto& localBin = threadLocalBins[t][tileIdx];
				if (!localBin.empty()) {
					tiles[tileIdx].triangleIndices.insert(
						tiles[tileIdx].triangleIndices.end(),
						localBin.begin(),
						localBin.end()
					);
				}
			}
			});
	}
	
};