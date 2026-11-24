#pragma once

#include <cstdint>

#include "graphics_internal.hpp"

namespace texture {

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t mip_levels = 1;
};

bool createFromPixels(const uint8_t* rgba, uint32_t width, uint32_t height, VkFormat format, Texture& out);

bool loadFromFile(const char* path, VkFormat format, Texture& out);

void destroy(Texture& texture);

VkSampler createSampler();

}
