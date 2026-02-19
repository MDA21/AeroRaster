#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

void print_avx(__m256 v) {
	float temp[8];
	_mm256_storeu_ps(temp, v); // 将寄存器内容存回内存
	std::cout << "[ ";
	for (int i = 0; i < 8; ++i) std::cout << temp[i] << " ";
	std::cout << "]" << std::endl;
}



//int main() {
//	float a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
//	float b[8] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
//
//	float result[8];
//
//	std::cout << "--- AVX2 Operation ---" << std::endl;
//
//	__m256 va = _mm256_loadu_ps(a);
//	__m256 vb = _mm256_loadu_ps(b);
//	__m256 vc = _mm256_set1_ps(2.0);
//
//	__m256 vres = _mm256_add_ps(va, vb);
//	__m256 vres1 = _mm256_fmadd_ps(va, vc, vb);
//
//	_mm256_storeu_ps(result, vres1);
//
//	print_avx(vres1);
//
//	return 0;
//}