#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "stb_image.h"
#include <Eigen/Dense>

class Texture {
public:
    int width, height, channels;
    unsigned char* data;

    Texture(const std::string& filename) {
        // stbi_set_flip_vertically_on_load(true); // OBJ UVs usually assume V=0 is bottom
        // Wait, LoadObj already flips V: uv.y() = 1.0f - attrib.texcoords[...]
        // So we might NOT want to flip here, or flip there.
        // Usually textures are stored top-down. OpenGL expects bottom-up.
        // If LoadObj does 1-v, then (0,0) is bottom-left.
        // If texture is loaded top-down, data[0] is top-left.
        // If we sample with (u, 1-v), we access bottom-left? No.
        // Let's stick to standard:
        // LoadObj does 1-v. So UV (0,0) is bottom-left.
        // Texture data[0] is top-left.
        // If we want UV(0,0) to map to bottom-left pixel, we need to flip the texture or flip the sampling Y.
        // Let's rely on stbi_load default (top-down) and LoadObj's 1-v.
        // If UV=(0,0) (bottom-left of model), we want bottom-left of texture.
        // data has top row first. So bottom row is at y=height-1.
        // Sample(0,0) -> y=0 -> Top row. WRONG.
        // So we need to flip Y in Sample or flip on load.
        // stbi_set_flip_vertically_on_load(true) makes data[0] be bottom-left.
        // Let's use that.
        
        data = stbi_load(filename.c_str(), &width, &height, &channels, 4); // Force 4 channels (RGBA)
        if (!data) {
            std::cerr << "Failed to load texture: " << filename << std::endl;
        } else {
            std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;
        }
    }

    ~Texture() {
        if (data) stbi_image_free(data);
    }

    Eigen::Vector4f Sample(float u, float v) const {
        if (!data) return Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f); // Pink error color

        // Wrap (Repeat)
        u = u - floor(u);
        v = v - floor(v);

        // Nearest Neighbor
        // Since we flipped V in LoadObj (1-v), UV(0,0) is bottom-left.
        // But stbi loads top-down by default.
        // If we didn't flip on load:
        // UV(0,0) -> y=0 (Top).
        // If LoadObj flips V, then UV(0,0) corresponds to original V=1.
        // Let's assume LoadObj's 1-v is correct for OpenGL style.
        // If we want Sample(0,0) to access the bottom-left of the image:
        // We should just map v directly if the image is bottom-up.
        // Let's just use stbi_load default (top-down) and see.
        // If it's upside down, we can fix it.
        // Actually, OBJ UVs usually have V=0 at bottom.
        // Images have Y=0 at top.
        // So V=0 should map to Y=height-1.
        // So y = (1-v) * height.
        // BUT LoadObj ALREADY does 1-v. So the V stored in mesh is "OpenGL V" (0 at bottom).
        // So stored V=0 means bottom.
        // If we sample V=0, we want bottom of image.
        // If image is top-down (Y=0 top), bottom is Y=height-1.
        // So y = (1 - v) * height.
        // Wait, LoadObj: uv.y = 1.0 - attrib...
        // So if attrib is 0 (bottom), uv.y becomes 1.
        // If attrib is 1 (top), uv.y becomes 0.
        // This seems backwards for OpenGL? OpenGL V=0 is bottom.
        // If OBJ file has (0,0) as bottom-left.
        // Then attrib is 0.
        // Then uv.y becomes 1.
        // So V=1 is bottom? That's inverted.
        
        // Let's check LoadObj again.
        // uv.y = 1.0f - attrib.texcoords[...]
        // If standard OBJ has (0,0) at bottom-left.
        // Then we are flipping it. So (0,1) becomes (0,0).
        // So now (0,0) is top-left.
        // If (0,0) is top-left, and image is top-down (0 is top).
        // Then y = v * height works perfectly.
        
        // So: LoadObj flips V so that 0 is Top.
        // Image is Top-Down.
        // So direct mapping works.
        
        int x = (int)(u * width);
        int y = (int)(v * height);

        // Clamp to valid range
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));

        int idx = (y * width + x) * 4;

        float r = data[idx + 0] / 255.0f;
        float g = data[idx + 1] / 255.0f;
        float b = data[idx + 2] / 255.0f;
        float a = data[idx + 3] / 255.0f;

        return Eigen::Vector4f(r, g, b, a);
    }
};
