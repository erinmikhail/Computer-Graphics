#include "application.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>

#include "math.hpp"
#include "cone.hpp"
#include "gpu_buffer.hpp"
#include "trajectory.hpp"

#include <imgui.h>

namespace application {


namespace {

struct UniformData {
	alignas(16) mat4 model;  // local → world
	alignas(16) mat4 view;   // world → камера
	alignas(16) mat4 proj;   // камера → clip space
	alignas(16) vec4 color;  // цвет объекта из UI (уже в линейном пространстве)
	alignas(16) vec4 params; // x = 1 — умножать на процедурные цвета вершин, 0 — нет
};
static_assert(sizeof(UniformData) == 3 * 64 + 2 * 16, "UniformData должна совпадать с std140-блоком в шейдере");


constexpr uint32_t max_objects = 16;


VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

// Геометрия конуса общая для всех объектов: объекты отличаются только своим UBO.
gpu::Buffer vertex_buffer;
gpu::Buffer index_buffer;
uint32_t index_count = 0;


struct ObjectSlot {
	gpu::Buffer ubo;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};
ObjectSlot slots[max_objects];
std::vector<uint32_t> free_slots;

struct Animation {
	bool enabled = false;      // двигается ли объект по траектории вообще
	bool playing = true;       // Play / Pause
	float speed = 1.0f;        // множитель скорости ( + отрицательный)
	bool orient_along_path = false; // поворачивать конус вершиной по направлению движения
	trajectory::Params path;
	float time = 0.0f;         // собственное "время анимации" объекта
};

struct SceneObject {
	char name[32] = {};
	uint32_t slot = 0;

	vec3 position = { 0.0f, 0.0f, 0.0f };
	vec3 rotation_deg = { 0.0f, 0.0f, 0.0f };
	vec3 scale = { 1.0f, 1.0f, 1.0f };

	vec3 color = { 1.0f, 1.0f, 1.0f }; 
	bool procedural_colors = true;

	Animation anim;
};

std::vector<SceneObject> objects;
int selected = -1;
int next_object_number = 1;

// Орбитальная камера
struct Camera {
	float yaw_deg = 35.0f;
	float pitch_deg = 25.0f;
	float distance = 5.0f;
	vec3 target = { 0.0f, 0.0f, 0.0f };
};
Camera camera;

struct Projection {
	bool perspective = true;
	float fov_deg = 60.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float ortho_size = 2.0f; 
};
Projection projection;

cone::Params cone_params;

bool cone_dirty = true;

double last_time = -1.0;


std::vector<char> readFile(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Cannot open " << path << '\n';
		return {};
	}

	const size_t size = size_t(file.tellg());
	std::vector<char> data(size);
	file.seekg(0);
	file.read(data.data(), size);
	return data;
}

VkShaderModule createShaderModule(const char* path){
	const std::vector<char> code = readFile(path);
	if (code.empty()) return VK_NULL_HANDLE;

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(), // байт
		.pCode = reinterpret_cast<const uint32_t*>(code.data()), // SPIr-V — массив 32-бит слов
	};

	VkShaderModule module = VK_NULL_HANDLE;
	vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module);
	return module;
}


float srgbToLinear(float c) {
	return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}


mat4 rotationFromEuler(vec3 deg) {
	return rotateZ(radians(deg.z)) * rotateY(radians(deg.y)) * rotateX(radians(deg.x));
}


vec3 cameraEye() {
	const float yaw = radians(camera.yaw_deg);
	const float pitch = radians(camera.pitch_deg);
	const vec3 offset = { std::cos(pitch) * std::sin(yaw),
	                      std::sin(pitch),
	                      std::cos(pitch) * std::cos(yaw) };
	return camera.target + offset * camera.distance;
}

mat4 projectionMatrix(float aspect) {
	if (projection.perspective) {
		return perspective(radians(projection.fov_deg), aspect, projection.near_plane, projection.far_plane);
	}

	const float h = projection.ortho_size;
	const float w = h * aspect;
	return ortho(-w, w, -h, h, projection.near_plane, projection.far_plane);
}

// Model = T * R * S: сначала масштаб, потом поворот, потом перенос 
mat4 modelMatrix(const SceneObject& obj) {
	vec3 position = obj.position;
	mat4 path_rotation = identity();

	if (obj.anim.enabled) {
		position = position + trajectory::evaluate(obj.anim.path, obj.anim.time);
		if (obj.anim.orient_along_path) {
			path_rotation = trajectory::orientationAlongPath(obj.anim.path, obj.anim.time);
		}
	}

	// Поворот по траектории применяется после поворота из UI
	return translate(position) * path_rotation * rotationFromEuler(obj.rotation_deg) * scale(obj.scale);
}

bool addObject() {
	if (free_slots.empty()) return false;

	SceneObject obj;
	obj.slot = free_slots.back();
	free_slots.pop_back();
	std::snprintf(obj.name, sizeof(obj.name), "Конус %d", next_object_number++);

	const int n = int(objects.size());
	obj.position = { float(n % 4) * 1.5f - 2.25f, 0.0f, -float(n / 4) * 1.5f };

	objects.push_back(obj);
	selected = int(objects.size()) - 1;
	return true;
}

void removeObject(int index) {
	if (index < 0 || index >= int(objects.size())) return;
	free_slots.push_back(objects[index].slot);
	objects.erase(objects.begin() + index);
	selected = std::min(selected, int(objects.size()) - 1);
}


void createDefaultScene() {
	addObject();
	objects[0].position = { 0.0f, 0.0f, 0.0f };

	addObject();
	SceneObject& flyer = objects[1];
	flyer.position = { 0.0f, 0.0f, 0.0f };
	flyer.scale = { 0.35f, 0.35f, 0.35f };
	flyer.color = { 1.0f, 0.6f, 0.2f };
	flyer.procedural_colors = false;
	flyer.anim.enabled = true;
	flyer.anim.orient_along_path = true;
	flyer.anim.path.type = trajectory::Type::Trefoil;
	flyer.anim.path.radius = 2.0f;
	flyer.anim.path.freq_a = 0.6f;

	selected = 0;
}


void loadUiFont() {
	const char* const candidates[] = {
		"C:/Windows/Fonts/segoeui.ttf",
		"C:/Windows/Fonts/arial.ttf",
	};
	for (const char* path : candidates) {
		std::error_code ec;
		if (std::filesystem::exists(path, ec)) {
			ImGuiIO& io = ImGui::GetIO();
			if (ImFont* font = io.Fonts->AddFontFromFileTTF(path, 17.0f)) {
				io.FontDefault = font;
			}
			return;
		}
	}
}



bool createDescriptors() {
	auto& ctx = graphics::internal::context;

	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = max_objects,
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetLayout layouts[max_objects];
	std::fill(std::begin(layouts), std::end(layouts), descriptor_set_layout);

	VkDescriptorSet sets[max_objects];
	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = layouts,
	};
	if (vkAllocateDescriptorSets(ctx.device, &alloc_info, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	for (uint32_t i = 0; i < max_objects; ++i) {
		if (!gpu::createBuffer(sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, slots[i].ubo)) {
			return false;
		}
		slots[i].descriptor_set = sets[i];

		const VkDescriptorBufferInfo buffer_info = {
			.buffer = slots[i].ubo.buffer,
			.offset = 0,
			.range = sizeof(UniformData),
		};
		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = sets[i],
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};
		vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
	}

	free_slots.clear();
	for (uint32_t i = max_objects; i > 0; --i) free_slots.push_back(i - 1);
	return true;
}

bool createPipeline() {
	auto& ctx = graphics::internal::context;

	VkShaderModule vert = createShaderModule("shaders/cone.vert.spv");
	VkShaderModule frag = createShaderModule("shaders/cone.frag.spv");
	if (!vert || !frag) {
		if (vert) vkDestroyShaderModule(ctx.device, vert, nullptr);
		if (frag) vkDestroyShaderModule(ctx.device, frag, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[2] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		  .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		  .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main" },
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(cone::Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	
	const VkVertexInputAttributeDescription vertex_attributes[2] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(cone::Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(cone::Vertex, color) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	// количество viewport/scissor.
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,      // LINE — каркасом (пригодится для отладки!)
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	// MSAA выключен: 1 сэмпл на пиксель.
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	// Тест глубины
	const VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	// Смешивание выключено, 4 канала цвета.
	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};
	if (vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		vkDestroyShaderModule(ctx.device, vert, nullptr);
		vkDestroyShaderModule(ctx.device, frag, nullptr);
		return false;
	}

	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth,
		.pColorBlendState = &blend,
		.pDynamicState = &dynamic,
		.layout = pipeline_layout,
		.renderPass = ctx.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);

	// Шейдерные модули на время создания pipeline
	vkDestroyShaderModule(ctx.device, vert, nullptr);
	vkDestroyShaderModule(ctx.device, frag, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}
	return true;
}


void drawProjectionUi() {
	if (!ImGui::CollapsingHeader("Проекция", ImGuiTreeNodeFlags_DefaultOpen)) return;

	int mode = projection.perspective ? 0 : 1;
	ImGui::RadioButton("Перспективная", &mode, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Ортографическая", &mode, 1);
	projection.perspective = (mode == 0);

	if (projection.perspective) {
		ImGui::SliderFloat("FOV (°)", &projection.fov_deg, 10.0f, 150.0f, "%.0f");
	} else {
		ImGui::DragFloat("Размер орто-коробки", &projection.ortho_size, 0.01f, 0.1f, 50.0f, "%.2f");
		ImGui::SetItemTooltip("Половина высоты видимой области. Ширина = размер * aspect.");
	}
	ImGui::DragFloat("Near", &projection.near_plane, 0.01f, 0.01f, 10.0f, "%.2f");
	ImGui::DragFloat("Far", &projection.far_plane, 0.5f, 0.1f, 1000.0f, "%.1f");
	// far обязан дальше near
	projection.far_plane = std::max(projection.far_plane, projection.near_plane + 0.01f);
}

void drawCameraUi() {
	if (!ImGui::CollapsingHeader("Камера")) return;

	ImGui::SliderFloat("Азимут (°)", &camera.yaw_deg, -180.0f, 180.0f, "%.0f");
	ImGui::SliderFloat("Наклон (°)", &camera.pitch_deg, -89.0f, 89.0f, "%.0f");
	ImGui::DragFloat("Дистанция", &camera.distance, 0.05f, 0.2f, 100.0f, "%.2f");
	ImGui::DragFloat3("Цель", &camera.target.x, 0.01f);
	if (!projection.perspective) {
		ImGui::TextDisabled("В ортографии дистанция не меняет размер — меняйте размер коробки.");
	}
	if (ImGui::Button("Сбросить камеру")) camera = {};
}

void drawConeUi() {
	if (!ImGui::CollapsingHeader("Геометрия конуса (общая)")) return;

	bool changed = false;
	changed |= ImGui::SliderInt("Сегменты", &cone_params.segments, int(cone::min_segments), int(cone::max_segments));
	changed |= ImGui::DragFloat("Радиус", &cone_params.radius, 0.01f, 0.05f, 5.0f, "%.2f");
	changed |= ImGui::DragFloat("Высота", &cone_params.height, 0.01f, 0.05f, 5.0f, "%.2f");
	if (changed) cone_dirty = true;
}

void drawObjectListUi() {
	if (!ImGui::CollapsingHeader("Объекты", ImGuiTreeNodeFlags_DefaultOpen)) return;

	ImGui::Text("Объектов: %d / %u", int(objects.size()), max_objects);

	if (ImGui::BeginListBox("##objects", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing()))) {
		for (int i = 0; i < int(objects.size()); ++i) {
			ImGui::PushID(i);
			if (ImGui::Selectable(objects[i].name, selected == i)) selected = i;
			ImGui::PopID();
		}
		ImGui::EndListBox();
	}

	ImGui::BeginDisabled(free_slots.empty());
	if (ImGui::Button("Добавить")) addObject();
	ImGui::EndDisabled();

	ImGui::SameLine();
	ImGui::BeginDisabled(selected < 0);
	if (ImGui::Button("Удалить")) removeObject(selected);
	ImGui::EndDisabled();
}

void drawSelectedObjectUi(SceneObject& obj) {
	ImGui::SeparatorText(obj.name);
	ImGui::InputText("Имя", obj.name, sizeof(obj.name));

	if (ImGui::CollapsingHeader("Трансформация", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Позиция", &obj.position.x, 0.01f);
		ImGui::DragFloat3("Поворот (°)", &obj.rotation_deg.x, 0.5f, -360.0f, 360.0f, "%.1f");
		ImGui::DragFloat3("Масштаб", &obj.scale.x, 0.01f, 0.01f, 10.0f, "%.2f");
		if (ImGui::Button("Сбросить трансформацию")) {
			obj.position = { 0.0f, 0.0f, 0.0f };
			obj.rotation_deg = { 0.0f, 0.0f, 0.0f };
			obj.scale = { 1.0f, 1.0f, 1.0f };
		}
	}

	if (ImGui::CollapsingHeader("Цвет", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Цвет", &obj.color.x);
		ImGui::Checkbox("Процедурные цвета вершин", &obj.procedural_colors);
		ImGui::SetItemTooltip("Итоговый цвет = цвет вершины * выбранный цвет");
	}

	if (ImGui::CollapsingHeader("Анимация", ImGuiTreeNodeFlags_DefaultOpen)) {
		Animation& anim = obj.anim;
		ImGui::Checkbox("Движение по траектории", &anim.enabled);

		ImGui::BeginDisabled(!anim.enabled);
		if (ImGui::Button(anim.playing ? "Пауза" : "Play")) anim.playing = !anim.playing;
		ImGui::SameLine();
		if (ImGui::Button("В начало")) anim.time = 0.0f;

		ImGui::SliderFloat("Скорость", &anim.speed, -5.0f, 5.0f, "%.2f");

		int type = int(anim.path.type);
		if (ImGui::Combo("Траектория", &type, trajectory::type_names, IM_ARRAYSIZE(trajectory::type_names))) {
			anim.path.type = trajectory::Type(type);
		}
		ImGui::DragFloat("Радиус траектории", &anim.path.radius, 0.01f, 0.0f, 20.0f, "%.2f");

		switch (anim.path.type) {
		case trajectory::Type::Lissajous:
			ImGui::DragFloat("Частота X", &anim.path.freq_a, 0.01f, 0.0f, 10.0f, "%.2f");
			ImGui::DragFloat("Частота Y", &anim.path.freq_b, 0.01f, 0.0f, 10.0f, "%.2f");
			ImGui::DragFloat("Частота Z", &anim.path.freq_c, 0.01f, 0.0f, 10.0f, "%.2f");
			ImGui::DragFloat("Фаза X", &anim.path.phase, 0.01f, -PI, PI, "%.2f");
			break;
		case trajectory::Type::Trefoil:
			ImGui::DragFloat("Частота обхода", &anim.path.freq_a, 0.01f, 0.0f, 10.0f, "%.2f");
			break;
		case trajectory::Type::Spiral:
			ImGui::DragFloat("Частота вращения", &anim.path.freq_a, 0.01f, 0.0f, 10.0f, "%.2f");
			ImGui::DragFloat("Частота подъёма", &anim.path.freq_b, 0.01f, 0.0f, 10.0f, "%.2f");
			break;
		}

		ImGui::Checkbox("Поворачивать по направлению движения", &anim.orient_along_path);
		ImGui::EndDisabled();
	}
}

void drawUi() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(420.0f, 680.0f), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Лабораторная №1 — Конус")) {
		ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
		drawProjectionUi();
		drawCameraUi();
		drawConeUi();
		drawObjectListUi();
		if (selected >= 0 && selected < int(objects.size())) {
			drawSelectedObjectUi(objects[selected]);
		}
	}
	ImGui::End();
}

}

bool initialize() {
	loadUiFont();

	if (!createDescriptors()) return false;
	if (!createPipeline()) return false;

	if (!gpu::createBuffer(sizeof(cone::Vertex) * cone::max_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer)) return false;
	if (!gpu::createBuffer(sizeof(uint16_t) * cone::max_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) return false;

	createDefaultScene();
	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkDeviceWaitIdle(ctx.device);

	vkDestroyPipeline(ctx.device, pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);

	vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, descriptor_set_layout, nullptr);

	for (ObjectSlot& slot : slots) {
		gpu::destroyBuffer(slot.ubo);
		slot.descriptor_set = VK_NULL_HANDLE;
	}
	gpu::destroyBuffer(vertex_buffer);
	gpu::destroyBuffer(index_buffer);
}


void update(double time) {

	float dt = last_time < 0.0 ? 0.0f : float(time - last_time);
	last_time = time;
	dt = std::min(dt, 0.1f);

	for (SceneObject& obj : objects) {
		if (obj.anim.enabled && obj.anim.playing) {
			obj.anim.time += dt * obj.anim.speed;
		}
	}

	drawUi();
}

void render(const graphics::internal::FrameData& fd) {
	auto& ctx = graphics::internal::context;
	VkCommandBuffer cmd = fd.command_buffer;
	if (cmd == VK_NULL_HANDLE) return;

	if (cone_dirty) {
		std::vector<cone::Vertex> vertices;
		std::vector<uint16_t> indices;
		cone::build(cone_params, vertices, indices);
		gpu::upload(vertex_buffer, vertices.data(), vertices.size() * sizeof(cone::Vertex));
		gpu::upload(index_buffer, indices.data(), indices.size() * sizeof(uint16_t));
		index_count = uint32_t(indices.size());
		cone_dirty = false;
	}

	const VkExtent2D extent = ctx.swapchain_extent;
	const float aspect = extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;

	const mat4 view = lookAt(cameraEye(), camera.target, { 0.0f, 1.0f, 0.0f });
	const mat4 proj = projectionMatrix(aspect);

	for (const SceneObject& obj : objects) {
		const UniformData data = {
			.model = modelMatrix(obj),
			.view = view,
			.proj = proj,
			.color = { srgbToLinear(obj.color.x), srgbToLinear(obj.color.y), srgbToLinear(obj.color.z), 1.0f },
			.params = { obj.procedural_colors ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f },
		};
		gpu::upload(slots[obj.slot].ubo, &data, sizeof(data));
	}

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin);

	VkClearValue clears[2] = {};
	clears[0].color = {{0.1f, 0.2f, 0.4f, 1.0f}};
	clears[1].depthStencil = {1.0f, 0}; 

	const VkRenderPassBeginInfo rp = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = ctx.render_pass,
		.framebuffer = fd.framebuffer, // набор картинок (цвет + глубина) под этот render pass
		.renderArea = { .offset = {0, 0}, .extent = extent },
		.clearValueCount = 2,
		.pClearValues = clears,
	};

	vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport vp = {
		.x = 0.0f, .y = 0.0f,
		.width = float(extent.width),
		.height = float(extent.height),
		.minDepth = 0.0f, .maxDepth = 1.0f,
	};

	const VkRect2D scissor = {
		.offset = {0, 0},
		.extent = extent,
	};

	vkCmdSetViewport(cmd, 0, 1, &vp);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	for (const SceneObject& obj : objects) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &slots[obj.slot].descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

}
