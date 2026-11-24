#pragma once

#include "graphics_internal.hpp"
#include "math.hpp"

namespace shadow {

constexpr uint32_t resolution = 2048;

struct Map {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	bool initialized = false;
};

struct Renderer {
	Map sun;
	VkFormat format = VK_FORMAT_UNDEFINED;
	VkSampler sampler = VK_NULL_HANDLE;
	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;

	bool initialize(VkShaderModule vertex_shader);
	void destroy();
	void writeDescriptor(VkDescriptorSet set, uint32_t binding, const Map& map) const;
	void begin(VkCommandBuffer cmd, Map& map, float constant_bias, float slope_bias) const;
	void draw(VkCommandBuffer cmd, const mat4& mvp, uint32_t index_count) const;
	void end(VkCommandBuffer cmd, Map& map) const;
};

}
