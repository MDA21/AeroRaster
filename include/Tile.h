#pragma once
#include "Mesh.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>
#include <cmath>

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

	std::vector<Tile> tiles;

	TileGrid(int w, int h, int size) {
		width = w;
		height = h;
		TILE_SIZE = size;
		numTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
		numTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;

		tiles.resize(numTilesX * numTilesY);

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

	void ClearBins() {
		for (Tile& tile : tiles) {
			tile.Reset();
		}
	}

	void BinTriangles(const MeshSoA& mesh, const MeshSoA& transformedMesh, int width, int height) {
		size_t indexCount = mesh.indices.size();

		for(size_t i = 0; i < indexCount; i += 3) {
			uint32_t i0 = mesh.indices[i];
			uint32_t i1 = mesh.indices[i + 1];
			uint32_t i2 = mesh.indices[i + 2];

			float x0 = transformedMesh.x[i0], y0 = transformedMesh.y[i0];
			float x1 = transformedMesh.x[i1], y1 = transformedMesh.y[i1];
			float x2 = transformedMesh.x[i2], y2 = transformedMesh.y[i2];

			int triMinX = std::max(0, (int)std::floor(std::min({ x0, x1, x2 })));
			int triMaxX = std::min(width - 1, (int)std::ceil(std::max({ x0, x1, x2 })));
			int triMinY = std::max(0, (int)std::floor(std::min({ y0, y1, y2 })));
			int triMaxY = std::min(height - 1, (int)std::ceil(std::max({ y0, y1, y2 })));

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
};

