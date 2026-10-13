#include "gpu_buffer.hpp"

#include <cstring>
#include <iostream>

namespace gpu {

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE, 
	};

	const VmaAllocationCreateInfo alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo info = {};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info,
	                    &out.buffer, &out.allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer of size " << size << '\n';
		return false;
	}

	out.mapped = info.pMappedData;
	out.size = size;
	return true;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {

		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

void upload(Buffer& buffer, const void* data, size_t size, size_t offset) {
	std::memcpy(static_cast<char*>(buffer.mapped) + offset, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, offset, size);
}

}
