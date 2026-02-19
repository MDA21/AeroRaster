#include "../include/model.h"
#include "../include/MathBase.h"
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
}