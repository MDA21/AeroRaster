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
        colorBuffer.resize(w * h, 0x202020); // 深灰色背景
        depthBuffer.resize(w * h, 1.0f);     // 深度初始化为1.0 (远裁剪面)
    }

    void Clear() {
        fill(colorBuffer.begin(), colorBuffer.end(), 0x202020);
        fill(depthBuffer.begin(), depthBuffer.end(), 1.0f);
    }

    // 简单的 PPM 图片输出，用于验证渲染结果
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