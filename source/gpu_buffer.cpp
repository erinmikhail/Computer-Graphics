#include "gpu_buffer.hpp"

#include <cstring>
#include <iostream>

namespace gpu {

namespace {

VkCommandPool upload_pool = VK_NULL_HANDLE;
VkFence upload_fence = VK_NULL_HANDLE;

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const VmaAllocationCreateInfo& alloc_info, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
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

}

bool initialize() {
	auto& ctx = graphics::internal::context;

	const VkCommandPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = ctx.graphics_queue_index,
	};
	const VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };

	return vkCreateCommandPool(ctx.device, &pool_info, nullptr, &upload_pool) == VK_SUCCESS &&
	       vkCreateFence(ctx.device, &fence_info, nullptr, &upload_fence) == VK_SUCCESS;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkDestroyFence(ctx.device, upload_fence, nullptr);
	vkDestroyCommandPool(ctx.device, upload_pool, nullptr);
	upload_fence = VK_NULL_HANDLE;
	upload_pool = VK_NULL_HANDLE;
}

void submitImmediate(const std::function<void(VkCommandBuffer)>& record) {
	auto& ctx = graphics::internal::context;

	const VkCommandBufferAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = upload_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(ctx.device, &alloc_info, &cmd);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin);
	record(cmd);
	vkEndCommandBuffer(cmd);

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	vkQueueSubmit(ctx.graphics_queue, 1, &submit, upload_fence);
	vkWaitForFences(ctx.device, 1, &upload_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(ctx.device, 1, &upload_fence);

	vkFreeCommandBuffers(ctx.device, upload_pool, 1, &cmd);
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VmaAllocationCreateInfo alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	return createBuffer(size, usage, alloc_info, out);
}

bool createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VmaAllocationCreateInfo alloc_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	return createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, alloc_info, out);
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

bool uploadToDevice(Buffer& buffer, const void* data, size_t size) {
	if (size == 0) return true;

	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
	upload(staging, data, size);

	submitImmediate([&](VkCommandBuffer cmd) {
		const VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
		vkCmdCopyBuffer(cmd, staging.buffer, buffer.buffer, 1, &region);

		const VkMemoryBarrier barrier = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
		};
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
		                     0, 1, &barrier, 0, nullptr, 0, nullptr);
	});

	destroyBuffer(staging);
	return true;
}

}
