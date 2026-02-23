#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;


struct Framebuffer {
    int width, height;
    vector<uint32_t> colorBuffer; // 格式: 0xRRGGBB
    vector<float> depthBuffer;

    Framebuffer(int w, int h) : width(w), height(h) {
        //增加 16 个像素的 padding 防护，可能会用无条件 SIMD 内存指令
        colorBuffer.resize(w * h + 16, 0x202020);
        depthBuffer.resize(w * h + 16, 1.0f);
    }

    void Clear() {
        fill(colorBuffer.begin(), colorBuffer.end(), 0x202020);
        fill(depthBuffer.begin(), depthBuffer.end(), 1.0f);
    }

    //简单的 PPM 图片输出
    void SaveToPPM(const string& filename) {
        ofstream ofs(filename, ios::binary);
        ofs << "P6\n" << width << " " << height << "\n255\n";
        for (uint32_t c : colorBuffer) {
            uint8_t r = (c >> 16) & 0xFF;
            uint8_t g = (c >> 8) & 0xFF;
            uint8_t b = c & 0xFF;
            ofs << r << g << b;
        }
    }
};