#pragma once
#include "ComputeBase.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>

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
	static constexpr int TILE_SIZE = 64;
	int width, height;
	int numTilesX, numTilesY;

	std::vector<Tile> tiles;

	TileGrid(int w, int h) {
		width = w;
		height = h;
		numTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
		numTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;

		tiles.resize(numTilesX * numTilesY);

		for(int y = 0; y < numTilesY; ++y) {
			for(int x = 0; x < numTilesX; ++x) {
				int idx = y * numTilesX + x;
				tiles[idx].startX = x * TILE_SIZE;
				tiles[idx].startY = y * TILE_SIZE;
				tiles[idx].endX = std::min((x + 1) * TILE_SIZE - 1, width - 1);
				tiles[idx].endY = std::min((y + 1) * TILE_SIZE - 1, height - 1);

				tiles[idx].triangleIndices.reserve(128);
			}
		}
	}

	void ClearBins() {
		for (Tile& tile : tiles) {
			tile.Reset();
		}
	}

	//单线程版分拣
	void BinTriangles(const MeshSoA& mesh, const MeshSoA& transformedMesh, int width, int height) {
		size_t indexCount = mesh.indices.size();

		for (size_t i = 0; i < indexCount; i += 3) {
			uint32_t i0 = mesh.indices[i];
			uint32_t i1 = mesh.indices[i + 1];
			uint32_t i2 = mesh.indices[i + 2];

			float x0 = transformedMesh.x[i0], y0 = transformedMesh.y[i0];
			float x1 = transformedMesh.x[i1], y1 = transformedMesh.y[i1];
			float x2 = transformedMesh.x[i2], y2 = transformedMesh.y[i2];

			// 计算三角形的 AABB
			int minX = std::max(0, (int)std::floor(std::min({ x0, x1, x2 })));
			int maxX = std::min(width - 1, (int)std::ceil(std::max({ x0, x1, x2 })));
			int minY = std::max(0, (int)std::floor(std::min({ y0, y1, y2 })));
			int maxY = std::min(height - 1, (int)std::ceil(std::max({ y0, y1, y2 })));

			if (minX > maxX || minY > maxY) continue;

			// 映射到 Tile 坐标系
			int tileMinX = minX / TILE_SIZE;
			int tileMaxX = maxX / TILE_SIZE;
			int tileMinY = minY / TILE_SIZE;
			int tileMaxY = maxY / TILE_SIZE;

			uint32_t triIndex = static_cast<uint32_t>(i);

			// 将该三角形 ID 压入所有相交的 Tile 的 Bin 中
			for (int ty = tileMinY; ty <= tileMaxY; ++ty) {
				for (int tx = tileMinX; tx <= tileMaxX; ++tx) {
					tiles[ty * numTilesX + tx].triangleIndices.push_back(triIndex);
				}
			}
		}
};

