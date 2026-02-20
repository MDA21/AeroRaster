#include "../include/model.h"
#include "../include/MathBase.h"
#include "../include/Camera.h"
#include <vector>
#include <immintrin.h>
#include <iostream>
#include <Eigen/Dense>

int main() {
	MeshSoA mesh;
	if (!Model::LoadObj("F:/VSproject/SIMDRenderer/resources/diablo3_pose/diablo3_pose.obj", mesh)) {
		std::cerr << "Failed to load model." << std::endl;
		return -1;
	}

	MeshSoA transformedMesh;
	transformedMesh.Reserve(mesh.GetVertexCount());

	int frameCount = 0;
	float width = 800.0f;
	float height = 600.0f;

	while (true) {
		Matrix4f mvp = Camera::ComputeMVP(width, height, frameCount++);

		TransformVerticesAVX2(mesh, transformedMesh, mvp);

		PerspectiveDivideAVX2(transformedMesh);

		ViewportTransformAVX2(transformedMesh, width, height);
	}

}