#pragma once

#include <cstddef>

#include "graphics_internal.hpp"

namespace gpu {

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out);
void destroyBuffer(Buffer& buffer);

void upload(Buffer& buffer, const void* data, size_t size, size_t offset = 0);

} 
