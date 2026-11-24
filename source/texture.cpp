#include "texture.hpp"

#include <algorithm>
#include <bit>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "gpu_buffer.hpp"

namespace texture {

namespace {

void transition(VkCommandBuffer cmd, VkImage image, uint32_t base_mip, uint32_t mip_count,
                VkImageLayout from, VkImageLayout to,
                VkAccessFlags src_access, VkAccessFlags dst_access,
                VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = base_mip,
			.levelCount = mip_count,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool supportsLinearBlit(VkFormat format) {
	VkFormatProperties properties = {};
	vkGetPhysicalDeviceFormatProperties(graphics::internal::context.physical_device, format, &properties);
	const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
	                                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
	return (properties.optimalTilingFeatures & needed) == needed;
}

void generateMipmaps(VkCommandBuffer cmd, const Texture& tex) {
	int32_t w = int32_t(tex.width);
	int32_t h = int32_t(tex.height);

	for (uint32_t level = 1; level < tex.mip_levels; ++level) {
		transition(cmd, tex.image, level - 1, 1,
		           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
		           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		const int32_t next_w = std::max(w / 2, 1);
		const int32_t next_h = std::max(h / 2, 1);
		const VkImageBlit blit = {
			.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 },
			.srcOffsets = { { 0, 0, 0 }, { w, h, 1 } },
			.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 },
			.dstOffsets = { { 0, 0, 0 }, { next_w, next_h, 1 } },
		};
		vkCmdBlitImage(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

		transition(cmd, tex.image, level - 1, 1,
		           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		           VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
		           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
		w = next_w;
		h = next_h;
	}

	transition(cmd, tex.image, tex.mip_levels - 1, 1,
	           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
	           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

}

bool createFromPixels(const uint8_t* rgba, uint32_t width, uint32_t height, VkFormat format, Texture& out) {
	auto& ctx = graphics::internal::context;

	out.width = width;
	out.height = height;
	out.mip_levels = supportsLinearBlit(format) ? uint32_t(std::bit_width(std::max(width, height))) : 1;

	const VkDeviceSize size = VkDeviceSize(width) * height * 4;
	gpu::Buffer staging;
	if (!gpu::createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
	gpu::upload(staging, rgba, size_t(size));

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { width, height, 1 },
		.mipLevels = out.mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo alloc_info = { .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
	if (vmaCreateImage(ctx.allocator, &image_info, &alloc_info, &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create texture image\n";
		gpu::destroyBuffer(staging);
		return false;
	}

	gpu::submitImmediate([&](VkCommandBuffer cmd) {
		transition(cmd, out.image, 0, out.mip_levels,
		           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		           0, VK_ACCESS_TRANSFER_WRITE_BIT,
		           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		const VkBufferImageCopy region = {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
			.imageOffset = { 0, 0, 0 },
			.imageExtent = { width, height, 1 },
		};
		vkCmdCopyBufferToImage(cmd, staging.buffer, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		generateMipmaps(cmd, out);
	});
	gpu::destroyBuffer(staging);

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = out.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, out.mip_levels, 0, 1 },
	};
	if (vkCreateImageView(ctx.device, &view_info, nullptr, &out.view) != VK_SUCCESS) {
		std::cerr << "Failed to create texture view\n";
		return false;
	}
	return true;
}

bool loadFromFile(const char* path, VkFormat format, Texture& out) {
	int width = 0, height = 0, channels = 0;
	stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	if (!pixels) {
		std::cerr << "Cannot load " << path << ": " << stbi_failure_reason() << '\n';
		return false;
	}

	const bool ok = createFromPixels(pixels, uint32_t(width), uint32_t(height), format, out);
	stbi_image_free(pixels);
	return ok;
}

void destroy(Texture& texture) {
	auto& ctx = graphics::internal::context;
	if (texture.view) vkDestroyImageView(ctx.device, texture.view, nullptr);
	if (texture.image) vmaDestroyImage(ctx.allocator, texture.image, texture.allocation);
	texture = {};
}

VkSampler createSampler() {
	const VkSamplerCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.anisotropyEnable = VK_FALSE,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
	};
	VkSampler sampler = VK_NULL_HANDLE;
	if (vkCreateSampler(graphics::internal::context.device, &info, nullptr, &sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create sampler\n";
	}
	return sampler;
}

}
