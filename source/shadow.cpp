#include "shadow.hpp"

#include "mesh.hpp"
#include <cstddef>
#include <initializer_list>

namespace shadow {

namespace {

VkFormat chooseFormat() {
	const auto& ctx = graphics::internal::context;
	for (VkFormat format : { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM }) {
		VkFormatProperties3 properties3 = {};
		properties3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
		VkFormatProperties2 properties2 = {};
		properties2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
		properties2.pNext = &properties3;
		vkGetPhysicalDeviceFormatProperties2(ctx.physical_device, format, &properties2);
		constexpr VkFormatFeatureFlags2 required = VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT |
		                                          VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
		                                          VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
		if ((properties3.optimalTilingFeatures & required) == required) return format;
	}
	return VK_FORMAT_UNDEFINED;
}

bool createMap(Map& map, VkFormat format) {
	auto& ctx = graphics::internal::context;
	const VkImageCreateInfo image = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { resolution, resolution, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo allocation = { .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
	if (vmaCreateImage(ctx.allocator, &image, &allocation, &map.image, &map.allocation, nullptr) != VK_SUCCESS) return false;
	const VkImageViewCreateInfo view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = map.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 },
	};
	return vkCreateImageView(ctx.device, &view, nullptr, &map.view) == VK_SUCCESS;
}

void destroyMap(Map& map) {
	auto& ctx = graphics::internal::context;
	if (map.view) vkDestroyImageView(ctx.device, map.view, nullptr);
	if (map.image) vmaDestroyImage(ctx.allocator, map.image, map.allocation);
	map = {};
}

void transition(VkCommandBuffer cmd, const Map& map, bool to_depth) {
	const VkPipelineStageFlags depth_stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	const VkAccessFlags depth_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = to_depth ? (map.initialized ? VkAccessFlags(VK_ACCESS_SHADER_READ_BIT) : 0u) : depth_access,
		.dstAccessMask = to_depth ? depth_access : VkAccessFlags(VK_ACCESS_SHADER_READ_BIT),
		.oldLayout = to_depth ? (map.initialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED)
		                     : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.newLayout = to_depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = map.image,
		.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 },
	};
	const VkPipelineStageFlags src = to_depth ? VkPipelineStageFlags(map.initialized ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) : depth_stages;
	const VkPipelineStageFlags dst = to_depth ? depth_stages : VkPipelineStageFlags(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	vkCmdPipelineBarrier(cmd, src, dst, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

}

bool Renderer::initialize(VkShaderModule vertex_shader) {
	auto& ctx = graphics::internal::context;
	format = chooseFormat();
	if (format == VK_FORMAT_UNDEFINED || !createMap(sun, format)) return false;
	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
	};
	if (vkCreateSampler(ctx.device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) return false;
	const VkPushConstantRange push = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mat4) };
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push,
	};
	if (vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &layout) != VK_SUCCESS) return false;
	const VkPipelineShaderStageCreateInfo stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = vertex_shader,
		.pName = "main",
	};
	const VkVertexInputBindingDescription binding = { 0, sizeof(mesh::Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
	const VkVertexInputAttributeDescription attribute = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh::Vertex, position) };
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 1,
		.pVertexAttributeDescriptions = &attribute,
	};
	const VkPipelineInputAssemblyStateCreateInfo assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo raster = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_TRUE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const VkPipelineColorBlendStateCreateInfo blend = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
	const VkDynamicState states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 3,
		.pDynamicStates = states,
	};
	const VkPipelineRenderingCreateInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.depthAttachmentFormat = format,
	};
	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &rendering,
		.stageCount = 1,
		.pStages = &stage,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &raster,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth,
		.pColorBlendState = &blend,
		.pDynamicState = &dynamic,
		.layout = layout,
		.renderPass = VK_NULL_HANDLE,
	};
	return vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS;
}

void Renderer::destroy() {
	auto& ctx = graphics::internal::context;
	if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
	if (layout) vkDestroyPipelineLayout(ctx.device, layout, nullptr);
	if (sampler) vkDestroySampler(ctx.device, sampler, nullptr);
	destroyMap(sun);
	*this = {};
}

void Renderer::writeDescriptor(VkDescriptorSet set, uint32_t binding, const Map& map) const {
	const VkDescriptorImageInfo image = { sampler, map.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &image,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void Renderer::begin(VkCommandBuffer cmd, Map& map, float constant_bias, float slope_bias) const {
	transition(cmd, map, true);
	const VkRenderingAttachmentInfo depth = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = map.view,
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = { .depthStencil = { 1.0f, 0 } },
	};
	const VkRenderingInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { { 0, 0 }, { resolution, resolution } },
		.layerCount = 1,
		.pDepthAttachment = &depth,
	};
	vkCmdBeginRendering(cmd, &rendering);
	const VkViewport viewport = { 0, 0, float(resolution), float(resolution), 0, 1 };
	const VkRect2D scissor = { { 0, 0 }, { resolution, resolution } };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	vkCmdSetDepthBias(cmd, constant_bias, 0.0f, slope_bias);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
}

void Renderer::draw(VkCommandBuffer cmd, const mat4& mvp, uint32_t index_count) const {
	vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mvp), &mvp);
	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
}

void Renderer::end(VkCommandBuffer cmd, Map& map) const {
	vkCmdEndRendering(cmd);
	transition(cmd, map, false);
	map.initialized = true;
}

}
