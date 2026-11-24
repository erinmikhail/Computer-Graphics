#pragma once

#include <cstddef>
#include <functional>

#include "graphics_internal.hpp"

namespace gpu {

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

bool initialize();
void shutdown();

void submitImmediate(const std::function<void(VkCommandBuffer)>& record);

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out);
bool createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out);
void destroyBuffer(Buffer& buffer);

void upload(Buffer& buffer, const void* data, size_t size, size_t offset = 0);

bool uploadToDevice(Buffer& buffer, const void* data, size_t size);

}
