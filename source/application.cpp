#include "application.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <random>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "math.hpp"
#include "cone.hpp"
#include "mesh.hpp"
#include "gpu_buffer.hpp"
#include "texture.hpp"
#include "trajectory.hpp"

#include <imgui.h>

namespace application {


namespace {

struct SceneData {
	mat4 view;
	mat4 proj;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_light_direction;
	vec4 dir_light_color;
	uint32_t dir_light_enabled;
	uint32_t point_light_count;
	uint32_t spot_light_count;
	uint32_t padding;
};
static_assert(sizeof(SceneData) == 2 * 64 + 4 * 16 + 16, "SceneData должна совпадать с std140-блоком в scene.glsl");
static_assert(offsetof(SceneData, dir_light_enabled) == 192, "счётчики источников идут сразу после четырёх vec4");

struct ObjectData {
	mat4 model;
	vec4 normal_matrix[3];
	vec4 albedo;
	vec4 specular;
	vec4 uv_scale_emission;
};
static_assert(sizeof(ObjectData) == 64 + 3 * 16 + 3 * 16, "ObjectData должна совпадать с std140-блоком в object.glsl");

struct GpuPointLight {
	vec4 position;
	vec4 color;
};
static_assert(sizeof(GpuPointLight) == 32, "GpuPointLight должна совпадать с std430-структурой PointLight");

struct GpuSpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;
	float cos_inner;
	float cos_outer;
	float padding[2];
};
static_assert(sizeof(GpuSpotLight) == 64, "GpuSpotLight должна совпадать с std430-структурой SpotLight");
static_assert(offsetof(GpuSpotLight, cos_inner) == 48, "cosInner идёт сразу после трёх vec4");

struct SamplingPushConstants {
	uint32_t enabled;
	float time;
	float distortion;
	float layer_mix;
	uint32_t seed;
};
static_assert(sizeof(SamplingPushConstants) <= 128, "гарантированный минимум maxPushConstantsSize — 128 байт");

struct MaterialFiles {
	const char* name;
	const char* albedo;
	const char* specular;
	const char* emissive;
};

const MaterialFiles material_files[] = {
	{ "Кирпич",               "textures/brick_albedo.png",   "textures/brick_specular.png",  nullptr },
	{ "Деревянный пол",       "textures/wood_albedo.png",    "textures/wood_specular.png",   nullptr },
	{ "Металлические панели", "textures/panel_albedo.png",   "textures/panel_specular.png",  "textures/panel_emissive.png" },
	{ "Лава",                 "textures/lava_albedo.png",    "textures/lava_specular.png",   "textures/lava_emissive.png" },
	{ "Шахматка",             "textures/checker_albedo.png", nullptr,                        nullptr },
	{ "Мрамор",               "textures/marble_albedo.png",  "textures/marble_specular.png", nullptr },
};
constexpr uint32_t material_count = uint32_t(std::size(material_files));

enum MaterialId : int { Brick, Wood, Panels, Lava, Checker, Marble };

constexpr uint32_t wall_count = 4;

constexpr vec3 world_up = { 0.0f, 1.0f, 0.0f };


GLFWwindow* window = nullptr;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout material_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline lit_pipeline = VK_NULL_HANDLE;
VkPipeline marker_pipeline = VK_NULL_HANDLE;
VkPipeline line_pipeline = VK_NULL_HANDLE;

struct GpuMesh {
	gpu::Buffer vertices;
	gpu::Buffer indices;
	uint32_t index_count = 0;
};

GpuMesh cone_mesh;
GpuMesh plane_mesh;
GpuMesh cube_mesh;
GpuMesh spot_wire_mesh;


gpu::Buffer object_ubo;
uint32_t object_capacity = 0;
VkDeviceSize object_stride = 0;
uint32_t next_object_index = 0;

struct GpuMaterial {
	texture::Texture albedo;
	texture::Texture specular;
	texture::Texture emissive;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};
GpuMaterial materials[material_count];
texture::Texture black_texture;
texture::Texture white_texture;
VkSampler sampler = VK_NULL_HANDLE;

struct LightBuffer {
	gpu::Buffer buffer;
	uint32_t capacity = 0;
};

VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
gpu::Buffer scene_ubo;
LightBuffer point_light_buffer;
LightBuffer spot_light_buffer;


struct Material {
	int texture = Checker;
	vec3 albedo = { 1.0f, 1.0f, 1.0f };
	vec3 specular = { 0.5f, 0.5f, 0.5f };
	float shininess = 32.0f;
	float emission = 1.0f;
};

struct Animation {
	bool enabled = false;
	bool playing = true;
	float speed = 1.0f;
	bool orient_along_path = false;
	trajectory::Params path;
	float time = 0.0f;
};

struct SceneObject {
	char name[32] = {};

	vec3 position = { 0.0f, 0.0f, 0.0f };
	vec3 rotation_deg = { 0.0f, 0.0f, 0.0f };
	vec3 scale = { 1.0f, 1.0f, 1.0f };

	Material material;

	Animation anim;

	bool special_sampling = false;
	uint32_t sampling_seed = 0;
};

std::vector<SceneObject> objects;
int selected_object = -1;
int next_object_number = 1;

struct Surface {
	bool visible = true;
	Material material;
	float tile_size = 2.0f;
};

struct Room {
	float size = 16.0f;
	float floor_height = -0.5f;
	float wall_height = 5.0f;
	Surface floor;
	Surface walls;
};
Room room;

struct SamplingEffect {
	bool playing = true;
	float speed = 1.0f;
	float distortion = 0.02f;
	float layer_mix = 0.6f;
	float time = 0.0f;
};
SamplingEffect sampling_effect;

struct Camera {
	vec3 position = { 0.0f, 2.5f, 7.5f };
	float yaw_deg = 0.0f;
	float pitch_deg = -18.0f;
	float move_speed = 3.0f;
	float mouse_sensitivity = 0.15f;
};
Camera camera;

bool mouse_look = false;
double last_cursor_x = 0.0;
double last_cursor_y = 0.0;

struct Projection {
	bool perspective = true;
	float fov_deg = 60.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float ortho_size = 2.0f;
};
Projection projection;

struct AmbientLight {
	vec3 color = { 0.6f, 0.7f, 1.0f };
	float intensity = 0.06f;
};
AmbientLight ambient;

struct DirectionalLight {
	bool enabled = true;
	vec3 direction = { -0.4f, -1.0f, -0.3f };
	vec3 color = { 1.0f, 0.95f, 0.85f };
	float intensity = 0.35f;
};
DirectionalLight sun;

struct PointLight {
	char name[32] = {};
	vec3 position = { 0.0f, 1.5f, 0.0f };
	vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 4.0f;
	Animation anim;
};

struct SpotLight {
	char name[32] = {};
	vec3 position = { 0.0f, 3.0f, 0.0f };
	vec3 direction = { 0.0f, -1.0f, 0.0f };
	vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 20.0f;
	float inner_deg = 15.0f;
	float outer_deg = 25.0f;
	Animation anim;
};

std::vector<PointLight> point_lights;
int selected_point_light = -1;
int next_point_light_number = 1;

std::vector<SpotLight> spot_lights;
int selected_spot_light = -1;
int next_spot_light_number = 1;

struct LightVisualization {
	bool enabled = true;
	float marker_size = 0.12f;
	float spot_length = 1.5f;
};
LightVisualization light_viz;

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
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module);
	return module;
}


float srgbToLinear(float c) {
	return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

vec3 srgbToLinear(vec3 c) {
	return { srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z) };
}

vec3 safeDirection(vec3 v) {
	const float len = length(v);
	return len > 1e-6f ? v * (1.0f / len) : vec3{ 0.0f, -1.0f, 0.0f };
}


mat4 rotationFromEuler(vec3 deg) {
	return rotateZ(radians(deg.z)) * rotateY(radians(deg.y)) * rotateX(radians(deg.x));
}


vec3 cameraForward() {
	const float yaw = radians(camera.yaw_deg);
	const float pitch = radians(camera.pitch_deg);
	return { std::cos(pitch) * std::sin(yaw),
	         std::sin(pitch),
	        -std::cos(pitch) * std::cos(yaw) };
}

mat4 viewMatrix() {
	return lookAt(camera.position, camera.position + cameraForward(), world_up);
}

mat4 projectionMatrix(float aspect) {
	if (projection.perspective) {
		return perspective(radians(projection.fov_deg), aspect, projection.near_plane, projection.far_plane);
	}

	const float h = projection.ortho_size;
	const float w = h * aspect;
	return ortho(-w, w, -h, h, projection.near_plane, projection.far_plane);
}

vec3 animatedPosition(vec3 base, const Animation& anim) {
	return anim.enabled ? base + trajectory::evaluate(anim.path, anim.time) : base;
}

template <typename Light>
vec3 lightPosition(const Light& light) {
	return animatedPosition(light.position, light.anim);
}

mat4 modelMatrix(const SceneObject& obj) {
	mat4 path_rotation = identity();
	if (obj.anim.enabled && obj.anim.orient_along_path) {
		path_rotation = trajectory::orientationAlongPath(obj.anim.path, obj.anim.time);
	}

	return translate(animatedPosition(obj.position, obj.anim)) * path_rotation *
	       rotationFromEuler(obj.rotation_deg) * scale(obj.scale);
}

mat4 floorModelMatrix() {
	return translate({ 0.0f, room.floor_height, 0.0f }) * scale({ room.size, 1.0f, room.size });
}

mat4 wallModelMatrix(uint32_t side) {
	const float half = room.size * 0.5f;
	return rotateY(radians(90.0f * float(side))) *
	       translate({ 0.0f, room.floor_height + room.wall_height * 0.5f, -half }) *
	       rotateX(radians(90.0f)) * scale({ room.size, 1.0f, room.wall_height });
}

mat4 spotWireModelMatrix(const SpotLight& light, float angle_deg) {
	const float wire_length = light_viz.spot_length;
	const float radius = wire_length * std::tan(radians(angle_deg));
	return translate(lightPosition(light)) * orientY(safeDirection(light.direction)) *
	       scale({ radius, wire_length, radius });
}

ObjectData makeObjectData(const mat4& model, const Material& material, float uv_scale_u = 1.0f, float uv_scale_v = 1.0f) {
	const mat3 normal = normalMatrix(model);
	return {
		.model = model,
		.normal_matrix = { toVec4(normal.c[0], 0.0f), toVec4(normal.c[1], 0.0f), toVec4(normal.c[2], 0.0f) },
		.albedo = toVec4(srgbToLinear(material.albedo), 1.0f),
		.specular = toVec4(srgbToLinear(material.specular), material.shininess),
		.uv_scale_emission = { uv_scale_u, uv_scale_v, material.emission, 0.0f },
	};
}

ObjectData makeMarkerData(const mat4& model, vec3 color) {
	return { .model = model, .albedo = toVec4(color, 1.0f) };
}


void addObject() {
	
	SceneObject obj;
	static std::mt19937 generator{std::random_device{}()};
	obj.sampling_seed = generator();
	std::snprintf(obj.name, sizeof(obj.name), "Конус %d", next_object_number++);

	const int n = int(objects.size());
	obj.position = { float(n % 4) * 1.5f - 2.25f, 0.0f, -float(n / 4) * 1.5f };

	objects.push_back(obj);
	selected_object = int(objects.size()) - 1;
}

void removeObject(int index) {
	if (index < 0 || index >= int(objects.size())) return;
	objects.erase(objects.begin() + index);
	selected_object = std::min(selected_object, int(objects.size()) - 1);
}

PointLight& addPointLight() {
	PointLight light;
	std::snprintf(light.name, sizeof(light.name), "Точечный %d", next_point_light_number++);
	point_lights.push_back(light);
	selected_point_light = int(point_lights.size()) - 1;
	return point_lights.back();
}

SpotLight& addSpotLight() {
	SpotLight light;
	std::snprintf(light.name, sizeof(light.name), "Прожектор %d", next_spot_light_number++);
	spot_lights.push_back(light);
	selected_spot_light = int(spot_lights.size()) - 1;
	return spot_lights.back();
}

template <typename T>
void removeSelected(std::vector<T>& items, int& selected) {
	if (selected < 0 || selected >= int(items.size())) return;
	items.erase(items.begin() + selected);
	selected = std::min(selected, int(items.size()) - 1);
}


void createDefaultScene() {
	room.floor.material = { .texture = Wood, .specular = { 0.6f, 0.6f, 0.6f }, .shininess = 48.0f };
	room.walls.material = { .texture = Brick, .specular = { 0.4f, 0.4f, 0.4f }, .shininess = 16.0f };
	room.walls.tile_size = 3.0f;

	addObject();
	SceneObject& plastic = objects.back();
	plastic.position = { -1.3f, 0.0f, 0.0f };
	plastic.material = { .texture = Checker, .specular = { 0.9f, 0.9f, 0.9f }, .shininess = 64.0f };
	plastic.special_sampling = true;

	addObject();
	SceneObject& gold = objects.back();
	gold.position = { 1.3f, 0.15f, -0.4f };
	gold.scale = { 1.3f, 1.3f, 1.3f };
	gold.material = { .texture = Panels, .albedo = { 0.95f, 0.75f, 0.4f }, .specular = { 1.0f, 0.85f, 0.5f }, .shininess = 200.0f };

	addObject();
	SceneObject& flyer = objects.back();
	flyer.position = { 0.0f, 1.6f, 0.0f };
	flyer.scale = { 0.35f, 0.35f, 0.35f };
	flyer.material = { .texture = Lava, .specular = { 0.3f, 0.3f, 0.3f }, .shininess = 8.0f, .emission = 2.0f };
	flyer.anim.enabled = true;
	flyer.anim.orient_along_path = true;
	flyer.anim.path.type = trajectory::Type::Trefoil;
	flyer.anim.path.radius = 2.0f;
	flyer.anim.path.freq_a = 0.6f;

	selected_object = 0;

	PointLight& orbiting = addPointLight();
	orbiting.position = { 0.0f, 0.9f, 0.0f };
	orbiting.color = { 1.0f, 0.45f, 0.15f };
	orbiting.intensity = 4.0f;
	orbiting.anim.enabled = true;
	orbiting.anim.path.type = trajectory::Type::Circle;
	orbiting.anim.path.radius = 2.6f;
	orbiting.anim.path.freq_a = 0.7f;

	PointLight& cold = addPointLight();
	cold.position = { -2.5f, 1.2f, -2.0f };
	cold.color = { 0.2f, 0.6f, 1.0f };
	cold.intensity = 3.0f;

	SpotLight& spot = addSpotLight();
	spot.position = { 2.5f, 3.5f, 2.5f };
	spot.direction = { -2.5f, -4.0f, -2.5f };
	spot.color = { 1.0f, 0.95f, 0.8f };
	spot.intensity = 25.0f;
	spot.inner_deg = 12.0f;
	spot.outer_deg = 22.0f;
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



void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, VkBuffer buffer,
                           VkDeviceSize range = VK_WHOLE_SIZE) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer,
		.offset = 0,
		.range = range,
	};
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void writeImageDescriptor(VkDescriptorSet set, uint32_t binding, VkImageView view) {
	const VkDescriptorImageInfo image_info = {
		.sampler = sampler,
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &image_info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

bool allocateDescriptorSet(VkDescriptorSetLayout layout, VkDescriptorSet& out) {
	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &layout,
	};
	if (vkAllocateDescriptorSets(graphics::internal::context.device, &alloc_info, &out) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}
	return true;
}

void reserveLightBuffer(LightBuffer& lights, uint32_t count, VkDeviceSize element_size, uint32_t binding) {
	if (lights.buffer.buffer != VK_NULL_HANDLE && count <= lights.capacity) return;

	const uint32_t capacity = std::max({ 1u, count, lights.capacity * 2 });

	gpu::Buffer buffer;
	if (!gpu::createBuffer(capacity * element_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer)) return;

	gpu::destroyBuffer(lights.buffer);
	lights.buffer = buffer;
	lights.capacity = capacity;
	writeBufferDescriptor(scene_descriptor_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer.buffer);
}

bool reserveObjectBuffer(uint32_t count) {
	if (object_ubo.buffer != VK_NULL_HANDLE && count <= object_capacity) return true;

	const uint32_t capacity = std::max({ 64u, count, object_capacity * 2 });

	gpu::Buffer buffer;
	if (!gpu::createBuffer(capacity * object_stride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, buffer)) return false;

	gpu::destroyBuffer(object_ubo);
	object_ubo = buffer;
	object_capacity = capacity;
	for (GpuMaterial& material : materials) {
		writeBufferDescriptor(material.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		                      object_ubo.buffer, sizeof(ObjectData));
	}
	return true;
}

bool createDescriptorSetLayouts() {
	auto& ctx = graphics::internal::context;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(std::size(scene_bindings)),
		.pBindings = scene_bindings,
	};
	if (vkCreateDescriptorSetLayout(ctx.device, &scene_layout_info, nullptr, &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding material_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const VkDescriptorSetLayoutCreateInfo material_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(std::size(material_bindings)),
		.pBindings = material_bindings,
	};
	if (vkCreateDescriptorSetLayout(ctx.device, &material_layout_info, nullptr, &material_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create material descriptor set layout\n";
		return false;
	}
	return true;
}

bool createDescriptors() {
	auto& ctx = graphics::internal::context;

	if (!createDescriptorSetLayouts()) return false;

	constexpr uint32_t set_count = 1 + material_count;
	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = material_count },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 * material_count },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = set_count,
		.poolSizeCount = uint32_t(std::size(pool_sizes)),
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	if (!allocateDescriptorSet(scene_set_layout, scene_descriptor_set)) return false;
	if (!gpu::createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_ubo)) return false;
	writeBufferDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_ubo.buffer);

	reserveLightBuffer(point_light_buffer, 0, sizeof(GpuPointLight), 1);
	reserveLightBuffer(spot_light_buffer, 0, sizeof(GpuSpotLight), 2);
	return point_light_buffer.buffer.buffer != VK_NULL_HANDLE && spot_light_buffer.buffer.buffer != VK_NULL_HANDLE;
}

VkImageView loadMap(const char* path, VkFormat format, texture::Texture& out, const texture::Texture& fallback) {
	if (path && texture::loadFromFile(path, format, out)) return out.view;
	return fallback.view;
}

bool createMaterials() {
	const uint8_t black[4] = { 0, 0, 0, 255 };
	const uint8_t white[4] = { 255, 255, 255, 255 };
	if (!texture::createFromPixels(black, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, black_texture)) return false;
	if (!texture::createFromPixels(white, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, white_texture)) return false;

	sampler = texture::createSampler();
	if (sampler == VK_NULL_HANDLE) return false;

	for (uint32_t i = 0; i < material_count; ++i) {
		const MaterialFiles& files = material_files[i];
		GpuMaterial& material = materials[i];

		if (!allocateDescriptorSet(material_set_layout, material.descriptor_set)) return false;
		writeImageDescriptor(material.descriptor_set, 1,
		                     loadMap(files.albedo, VK_FORMAT_R8G8B8A8_SRGB, material.albedo, white_texture));
		writeImageDescriptor(material.descriptor_set, 2,
		                     loadMap(files.specular, VK_FORMAT_R8G8B8A8_UNORM, material.specular, black_texture));
		writeImageDescriptor(material.descriptor_set, 3,
		                     loadMap(files.emissive, VK_FORMAT_R8G8B8A8_SRGB, material.emissive, black_texture));
	}

	VkPhysicalDeviceProperties properties = {};
	vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	object_stride = (sizeof(ObjectData) + alignment - 1) / alignment * alignment;

	return reserveObjectBuffer(0);
}

void destroyMaterials() {
	for (GpuMaterial& material : materials) {
		texture::destroy(material.albedo);
		texture::destroy(material.specular);
		texture::destroy(material.emissive);
	}
	texture::destroy(black_texture);
	texture::destroy(white_texture);
	vkDestroySampler(graphics::internal::context.device, sampler, nullptr);
}

bool createPipelineLayout() {
	const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, material_set_layout };
	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(SamplingPushConstants),
	};
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = uint32_t(std::size(set_layouts)),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};
	if (vkCreatePipelineLayout(graphics::internal::context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}
	return true;
}

struct PipelineDesc {
	const char* vertex_shader;
	const char* fragment_shader;
	VkPrimitiveTopology topology;
	uint32_t vertex_attribute_count;
	VkCullModeFlags cull_mode;
	bool depth_write;
};

VkPipeline createPipeline(const PipelineDesc& desc) {
	auto& ctx = graphics::internal::context;

	VkShaderModule vert = createShaderModule(desc.vertex_shader);
	VkShaderModule frag = createShaderModule(desc.fragment_shader);
	if (!vert || !frag) {
		if (vert) vkDestroyShaderModule(ctx.device, vert, nullptr);
		if (frag) vkDestroyShaderModule(ctx.device, frag, nullptr);
		return VK_NULL_HANDLE;
	}

	const VkPipelineShaderStageCreateInfo stages[2] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		  .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		  .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main" },
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(mesh::Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[3] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(mesh::Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(mesh::Vertex, normal) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
		  .offset = offsetof(mesh::Vertex, uv) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = desc.vertex_attribute_count,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = desc.topology,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = desc.cull_mode,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

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

	VkPipeline pipeline = VK_NULL_HANDLE;
	const VkResult result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);

	vkDestroyShaderModule(ctx.device, vert, nullptr);
	vkDestroyShaderModule(ctx.device, frag, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline (" << desc.vertex_shader << ")\n";
		return VK_NULL_HANDLE;
	}
	return pipeline;
}

bool createPipelines() {
	if (!createPipelineLayout()) return false;

	lit_pipeline = createPipeline({
		.vertex_shader = "shaders/lighting.vert.spv",
		.fragment_shader = "shaders/lighting.frag.spv",
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.vertex_attribute_count = 3,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
		.depth_write = true,
	});
	marker_pipeline = createPipeline({
		.vertex_shader = "shaders/marker.vert.spv",
		.fragment_shader = "shaders/marker.frag.spv",
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.vertex_attribute_count = 1,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
		.depth_write = true,
	});
	line_pipeline = createPipeline({
		.vertex_shader = "shaders/marker.vert.spv",
		.fragment_shader = "shaders/marker.frag.spv",
		.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		.vertex_attribute_count = 1,
		.cull_mode = VK_CULL_MODE_NONE,
		.depth_write = false,
	});
	return lit_pipeline && marker_pipeline && line_pipeline;
}


bool createMesh(GpuMesh& gpu_mesh, size_t max_vertices, size_t max_indices) {
	return gpu::createDeviceLocalBuffer(sizeof(mesh::Vertex) * max_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, gpu_mesh.vertices) &&
	       gpu::createDeviceLocalBuffer(sizeof(uint16_t) * max_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, gpu_mesh.indices);
}

void uploadMesh(GpuMesh& gpu_mesh, const std::vector<mesh::Vertex>& vertices, const std::vector<uint16_t>& indices) {
	gpu::uploadToDevice(gpu_mesh.vertices, vertices.data(), vertices.size() * sizeof(mesh::Vertex));
	gpu::uploadToDevice(gpu_mesh.indices, indices.data(), indices.size() * sizeof(uint16_t));
	gpu_mesh.index_count = uint32_t(indices.size());
}

using BuildFunction = void (*)(std::vector<mesh::Vertex>&, std::vector<uint16_t>&);

bool createStaticMesh(GpuMesh& gpu_mesh, BuildFunction build) {
	std::vector<mesh::Vertex> vertices;
	std::vector<uint16_t> indices;
	build(vertices, indices);
	if (!createMesh(gpu_mesh, vertices.size(), indices.size())) return false;
	uploadMesh(gpu_mesh, vertices, indices);
	return true;
}

bool createMeshes() {
	return createMesh(cone_mesh, cone::max_vertices, cone::max_indices) &&
	       createStaticMesh(plane_mesh, mesh::buildPlane) &&
	       createStaticMesh(cube_mesh, mesh::buildCube) &&
	       createStaticMesh(spot_wire_mesh, mesh::buildSpotWireframe);
}

void destroyMesh(GpuMesh& gpu_mesh) {
	gpu::destroyBuffer(gpu_mesh.vertices);
	gpu::destroyBuffer(gpu_mesh.indices);
	gpu_mesh.index_count = 0;
}

void bindMesh(VkCommandBuffer cmd, const GpuMesh& gpu_mesh) {
	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &gpu_mesh.vertices.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, gpu_mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT16);
}


void updateCamera(float dt) {
	const ImGuiIO& io = ImGui::GetIO();

	const bool right_button = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
	if (right_button && !mouse_look && !io.WantCaptureMouse) {
		mouse_look = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwGetCursorPos(window, &last_cursor_x, &last_cursor_y);
	} else if (!right_button && mouse_look) {
		mouse_look = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	if (mouse_look) {
		double x = 0.0, y = 0.0;
		glfwGetCursorPos(window, &x, &y);
		camera.yaw_deg = std::remainder(camera.yaw_deg + float(x - last_cursor_x) * camera.mouse_sensitivity, 360.0f);
		camera.pitch_deg = std::clamp(camera.pitch_deg - float(y - last_cursor_y) * camera.mouse_sensitivity, -89.0f, 89.0f);
		last_cursor_x = x;
		last_cursor_y = y;
	}

	if (io.WantCaptureKeyboard) return;

	auto axis = [](int positive_key, int negative_key) {
		return float(glfwGetKey(window, positive_key) == GLFW_PRESS) -
		       float(glfwGetKey(window, negative_key) == GLFW_PRESS);
	};

	const vec3 forward = cameraForward();
	const vec3 right = normalize(cross(forward, world_up));
	const vec3 move = forward * axis(GLFW_KEY_W, GLFW_KEY_S) +
	                  right * axis(GLFW_KEY_D, GLFW_KEY_A) +
	                  world_up * axis(GLFW_KEY_E, GLFW_KEY_Q);
	if (length(move) < 1e-6f) return;

	const bool fast = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
	const float speed = camera.move_speed * (fast ? 3.0f : 1.0f);
	camera.position = camera.position + normalize(move) * (speed * dt);
}

void advanceAnimation(Animation& anim, float dt) {
	if (anim.enabled && anim.playing) {
		anim.time += dt * anim.speed;
	}
}

const char* materialName(void*, int index) {
	return material_files[index].name;
}


template <typename T>
void drawSelectableList(const std::vector<T>& items, int& selected) {
	if (ImGui::BeginListBox("##list", ImVec2(-FLT_MIN, 4 * ImGui::GetTextLineHeightWithSpacing()))) {
		for (int i = 0; i < int(items.size()); ++i) {
			ImGui::PushID(i);
			if (ImGui::Selectable(items[i].name, selected == i)) selected = i;
			ImGui::PopID();
		}
		ImGui::EndListBox();
	}
}

void drawMaterialUi(Material& material) {
	ImGui::Combo("Текстура", &material.texture, materialName, nullptr, int(material_count));
	const MaterialFiles& files = material_files[material.texture];
	ImGui::TextDisabled("specular: %s", files.specular ? files.specular : "нет (чёрный пиксель)");
	ImGui::TextDisabled("emissive: %s", files.emissive ? files.emissive : "нет (чёрный пиксель)");

	ImGui::ColorEdit3("Оттенок альбедо", &material.albedo.x);
	ImGui::SetItemTooltip("Умножается на цвет из текстуры; белый — текстура как есть");
	ImGui::ColorEdit3("Цвет блика", &material.specular.x);
	ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 512.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Яркость свечения", &material.emission, 0.0f, 5.0f, "%.2f");
}

void drawAnimationUi(Animation& anim, bool allow_orientation) {
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
	case trajectory::Type::Circle:
		ImGui::DragFloat("Частота обхода", &anim.path.freq_a, 0.01f, 0.0f, 10.0f, "%.2f");
		break;
	case trajectory::Type::Spiral:
		ImGui::DragFloat("Частота вращения", &anim.path.freq_a, 0.01f, 0.0f, 10.0f, "%.2f");
		ImGui::DragFloat("Частота подъёма", &anim.path.freq_b, 0.01f, 0.0f, 10.0f, "%.2f");
		break;
	}

	if (allow_orientation) {
		ImGui::Checkbox("Поворачивать по направлению движения", &anim.orient_along_path);
	}
	ImGui::EndDisabled();
}

void drawCameraUi() {
	if (!ImGui::CollapsingHeader("Камера", ImGuiTreeNodeFlags_DefaultOpen)) return;
	ImGui::PushID("camera");

	ImGui::TextDisabled("WASD — движение, Q/E — вниз/вверх, Shift — быстрее, ПКМ — обзор");
	ImGui::DragFloat3("Позиция", &camera.position.x, 0.05f);
	ImGui::SliderFloat("Yaw (°)", &camera.yaw_deg, -180.0f, 180.0f, "%.1f");
	ImGui::SliderFloat("Pitch (°)", &camera.pitch_deg, -89.0f, 89.0f, "%.1f");
	ImGui::DragFloat("Скорость (ед/с)", &camera.move_speed, 0.05f, 0.1f, 50.0f, "%.2f");
	ImGui::DragFloat("Чувствительность мыши", &camera.mouse_sensitivity, 0.005f, 0.01f, 1.0f, "%.3f");
	if (ImGui::Button("Сбросить камеру")) camera = {};

	ImGui::PopID();
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
	projection.far_plane = std::max(projection.far_plane, projection.near_plane + 0.01f);
}

void drawAmbientUi() {
	if (!ImGui::CollapsingHeader("Фоновое освещение (ambient)")) return;
	ImGui::PushID("ambient");

	ImGui::ColorEdit3("Цвет", &ambient.color.x);
	ImGui::SliderFloat("Интенсивность", &ambient.intensity, 0.0f, 1.0f, "%.3f");

	ImGui::PopID();
}

void drawDirectionalLightUi() {
	if (!ImGui::CollapsingHeader("Направленный свет")) return;
	ImGui::PushID("directional");

	ImGui::Checkbox("Включён", &sun.enabled);
	ImGui::BeginDisabled(!sun.enabled);
	ImGui::DragFloat3("Направление", &sun.direction.x, 0.01f, -1.0f, 1.0f, "%.2f");
	ImGui::SetItemTooltip("Куда светит; нормализуется перед отправкой в шейдер");
	ImGui::ColorEdit3("Цвет", &sun.color.x);
	ImGui::DragFloat("Интенсивность", &sun.intensity, 0.01f, 0.0f, 20.0f, "%.2f");
	ImGui::EndDisabled();

	ImGui::PopID();
}

void drawPointLightsUi() {
	if (!ImGui::CollapsingHeader("Точечные источники")) return;
	ImGui::PushID("point_lights");

	ImGui::Text("Источников: %d, ёмкость буфера: %u", int(point_lights.size()), point_light_buffer.capacity);
	drawSelectableList(point_lights, selected_point_light);

	if (ImGui::Button("Добавить")) addPointLight();
	ImGui::SameLine();
	ImGui::BeginDisabled(selected_point_light < 0);
	if (ImGui::Button("Удалить")) removeSelected(point_lights, selected_point_light);
	ImGui::EndDisabled();

	if (selected_point_light >= 0) {
		PointLight& light = point_lights[selected_point_light];
		ImGui::SeparatorText(light.name);
		ImGui::DragFloat3("Позиция", &light.position.x, 0.01f);
		ImGui::ColorEdit3("Цвет", &light.color.x);
		ImGui::DragFloat("Интенсивность", &light.intensity, 0.05f, 0.0f, 1000.0f, "%.2f");
		ImGui::SetItemTooltip("Освещённость на расстоянии d: интенсивность / d²");
		drawAnimationUi(light.anim, false);
	}

	ImGui::PopID();
}

void drawSpotLightsUi() {
	if (!ImGui::CollapsingHeader("Прожекторы")) return;
	ImGui::PushID("spot_lights");

	ImGui::Text("Прожекторов: %d, ёмкость буфера: %u", int(spot_lights.size()), spot_light_buffer.capacity);
	drawSelectableList(spot_lights, selected_spot_light);

	if (ImGui::Button("Добавить")) addSpotLight();
	ImGui::SameLine();
	ImGui::BeginDisabled(selected_spot_light < 0);
	if (ImGui::Button("Удалить")) removeSelected(spot_lights, selected_spot_light);
	ImGui::EndDisabled();

	if (selected_spot_light >= 0) {
		SpotLight& light = spot_lights[selected_spot_light];
		ImGui::SeparatorText(light.name);
		ImGui::DragFloat3("Позиция", &light.position.x, 0.01f);
		ImGui::DragFloat3("Направление", &light.direction.x, 0.01f, -10.0f, 10.0f, "%.2f");
		ImGui::ColorEdit3("Цвет", &light.color.x);
		ImGui::DragFloat("Интенсивность", &light.intensity, 0.05f, 0.0f, 1000.0f, "%.2f");
		ImGui::SliderFloat("Внутренний угол (°)", &light.inner_deg, 0.0f, 85.0f, "%.1f");
		ImGui::SliderFloat("Внешний угол (°)", &light.outer_deg, 0.0f, 85.0f, "%.1f");
		if (light.outer_deg <= light.inner_deg) {
			ImGui::TextDisabled("Внешний угол <= внутреннего: край жёсткий");
		}
		drawAnimationUi(light.anim, false);
	}

	ImGui::PopID();
}

void drawLightVisualizationUi() {
	if (!ImGui::CollapsingHeader("Визуализация источников")) return;

	ImGui::Checkbox("Показывать источники", &light_viz.enabled);
	ImGui::BeginDisabled(!light_viz.enabled);
	ImGui::DragFloat("Размер маркера", &light_viz.marker_size, 0.005f, 0.01f, 1.0f, "%.3f");
	ImGui::DragFloat("Длина каркаса прожектора", &light_viz.spot_length, 0.01f, 0.1f, 20.0f, "%.2f");
	ImGui::EndDisabled();
}

void drawConeUi() {
	if (!ImGui::CollapsingHeader("Геометрия конуса (общая)")) return;

	bool changed = false;
	changed |= ImGui::SliderInt("Сегменты", &cone_params.segments, int(cone::min_segments), int(cone::max_segments));
	changed |= ImGui::DragFloat("Радиус", &cone_params.radius, 0.01f, 0.05f, 5.0f, "%.2f");
	changed |= ImGui::DragFloat("Высота", &cone_params.height, 0.01f, 0.05f, 5.0f, "%.2f");
	if (changed) cone_dirty = true;
}

void drawSurfaceUi(const char* id, Surface& surface) {
	ImGui::PushID(id);
	ImGui::Checkbox("Показывать", &surface.visible);
	ImGui::DragFloat("Размер плитки текстуры (м)", &surface.tile_size, 0.01f, 0.1f, 50.0f, "%.2f");
	drawMaterialUi(surface.material);
	ImGui::PopID();
}

void drawRoomUi() {
	if (!ImGui::CollapsingHeader("Комната (пол и стены)")) return;
	ImGui::PushID("room");

	ImGui::DragFloat("Размер", &room.size, 0.1f, 0.5f, 200.0f, "%.1f");
	ImGui::DragFloat("Высота пола", &room.floor_height, 0.01f, -20.0f, 20.0f, "%.2f");
	ImGui::DragFloat("Высота стен", &room.wall_height, 0.01f, 0.1f, 50.0f, "%.2f");
	ImGui::SeparatorText("Пол");
	drawSurfaceUi("floor", room.floor);
	ImGui::SeparatorText("Стены");
	drawSurfaceUi("walls", room.walls);

	ImGui::PopID();
}

void drawSamplingEffectUi() {
	if (!ImGui::CollapsingHeader("Нетривиальное сэмплирование")) return;
	ImGui::PushID("sampling");

	ImGui::TextDisabled("Включается у объекта флажком; параметры идут в push constants");
	if (ImGui::Button(sampling_effect.playing ? "Пауза" : "Play")) sampling_effect.playing = !sampling_effect.playing;
	ImGui::SliderFloat("Скорость", &sampling_effect.speed, 0.0f, 5.0f, "%.2f");
	ImGui::SliderFloat("Искажение UV", &sampling_effect.distortion, 0.0f, 0.1f, "%.3f");
	ImGui::SliderFloat("Доля второго слоя", &sampling_effect.layer_mix, 0.0f, 1.0f, "%.2f");

	ImGui::PopID();
}

void drawObjectListUi() {
	if (!ImGui::CollapsingHeader("Объекты", ImGuiTreeNodeFlags_DefaultOpen)) return;
	ImGui::PushID("objects");

	ImGui::Text("Объектов: %d, ёмкость динамического UBO: %u", int(objects.size()), object_capacity);
	drawSelectableList(objects, selected_object);

	if (ImGui::Button("Добавить")) addObject();

	ImGui::SameLine();
	ImGui::BeginDisabled(selected_object < 0);
	if (ImGui::Button("Удалить")) removeObject(selected_object);
	ImGui::EndDisabled();

	ImGui::PopID();
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

	if (ImGui::CollapsingHeader("Материал", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawMaterialUi(obj.material);
		ImGui::Checkbox("Нетривиальное сэмплирование", &obj.special_sampling);
	}

	if (ImGui::CollapsingHeader("Анимация", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawAnimationUi(obj.anim, true);
	}
}

void drawUi() {
	const ImGuiIO& io = ImGui::GetIO();

	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(390.0f, 600.0f), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Лабораторная №3 — Текстурирование")) {
		ImGui::Text("%.1f FPS", io.Framerate);
		drawCameraUi();
		drawProjectionUi();
		drawAmbientUi();
		drawDirectionalLightUi();
		drawPointLightsUi();
		drawSpotLightsUi();
		drawLightVisualizationUi();
		drawSamplingEffectUi();
	}
	ImGui::End();

	ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 400.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(390.0f, 600.0f), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Объекты")) {
		drawConeUi();
		drawRoomUi();
		drawObjectListUi();
		if (selected_object >= 0 && selected_object < int(objects.size())) {
			drawSelectedObjectUi(objects[selected_object]);
		}
	}
	ImGui::End();
}


uint32_t uploadPointLights() {
	reserveLightBuffer(point_light_buffer, uint32_t(point_lights.size()), sizeof(GpuPointLight), 1);
	const uint32_t count = std::min(uint32_t(point_lights.size()), point_light_buffer.capacity);

	std::vector<GpuPointLight> data(count);
	for (uint32_t i = 0; i < count; ++i) {
		const PointLight& light = point_lights[i];
		data[i] = {
			.position = toVec4(lightPosition(light), 1.0f),
			.color = toVec4(srgbToLinear(light.color), light.intensity),
		};
	}
	if (count > 0) gpu::upload(point_light_buffer.buffer, data.data(), count * sizeof(GpuPointLight));
	return count;
}

uint32_t uploadSpotLights() {
	reserveLightBuffer(spot_light_buffer, uint32_t(spot_lights.size()), sizeof(GpuSpotLight), 2);
	const uint32_t count = std::min(uint32_t(spot_lights.size()), spot_light_buffer.capacity);

	std::vector<GpuSpotLight> data(count);
	for (uint32_t i = 0; i < count; ++i) {
		const SpotLight& light = spot_lights[i];
		data[i] = {
			.position = toVec4(lightPosition(light), 1.0f),
			.direction = toVec4(safeDirection(light.direction), 0.0f),
			.color = toVec4(srgbToLinear(light.color), light.intensity),
			.cos_inner = std::cos(radians(light.inner_deg)),
			.cos_outer = std::cos(radians(light.outer_deg)),
		};
	}
	if (count > 0) gpu::upload(spot_light_buffer.buffer, data.data(), count * sizeof(GpuSpotLight));
	return count;
}

void bindObject(VkCommandBuffer cmd, const ObjectData& data, int material) {
	const uint32_t offset = uint32_t(next_object_index++ * object_stride);
	gpu::upload(object_ubo, &data, sizeof(data), offset);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        1, 1, &materials[material].descriptor_set, 1, &offset);
}

void pushSampling(VkCommandBuffer cmd, bool enabled, uint32_t seed) {
    const SamplingPushConstants constants = {
        .enabled = enabled ? 1u : 0u,
        .time = sampling_effect.time,
        .distortion = sampling_effect.distortion,
        .layer_mix = sampling_effect.layer_mix,
        .seed = seed,
    };
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(constants), &constants);
}

void drawLightMarkers(VkCommandBuffer cmd) {
	const vec3 marker_scale = { light_viz.marker_size, light_viz.marker_size, light_viz.marker_size };

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, marker_pipeline);
	bindMesh(cmd, cube_mesh);
	for (const PointLight& light : point_lights) {
		bindObject(cmd, makeMarkerData(translate(lightPosition(light)) * scale(marker_scale), srgbToLinear(light.color)), 0);
		vkCmdDrawIndexed(cmd, cube_mesh.index_count, 1, 0, 0, 0);
	}
	for (const SpotLight& light : spot_lights) {
		bindObject(cmd, makeMarkerData(translate(lightPosition(light)) * scale(marker_scale), srgbToLinear(light.color)), 0);
		vkCmdDrawIndexed(cmd, cube_mesh.index_count, 1, 0, 0, 0);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
	bindMesh(cmd, spot_wire_mesh);
	for (const SpotLight& light : spot_lights) {
		bindObject(cmd, makeMarkerData(spotWireModelMatrix(light, light.outer_deg), { 1.0f, 1.0f, 1.0f }), 0);
		vkCmdDrawIndexed(cmd, spot_wire_mesh.index_count, 1, 0, 0, 0);
		bindObject(cmd, makeMarkerData(spotWireModelMatrix(light, light.inner_deg), { 1.0f, 1.0f, 0.0f }), 0);
		vkCmdDrawIndexed(cmd, spot_wire_mesh.index_count, 1, 0, 0, 0);
	}
}

uint32_t countDraws() {
	uint32_t count = 1 + wall_count + uint32_t(objects.size());
	if (light_viz.enabled) {
		count += uint32_t(point_lights.size() + 3 * spot_lights.size());
	}
	return count;
}

}

bool initialize(GLFWwindow* app_window) {
	window = app_window;
	loadUiFont();

	if (!gpu::initialize()) return false;
	if (!createDescriptors()) return false;
	if (!createMaterials()) return false;
	if (!createPipelines()) return false;
	if (!createMeshes()) return false;

	createDefaultScene();
	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkDeviceWaitIdle(ctx.device);

	vkDestroyPipeline(ctx.device, lit_pipeline, nullptr);
	vkDestroyPipeline(ctx.device, marker_pipeline, nullptr);
	vkDestroyPipeline(ctx.device, line_pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);

	vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, material_set_layout, nullptr);

	destroyMaterials();
	gpu::destroyBuffer(object_ubo);
	gpu::destroyBuffer(scene_ubo);
	gpu::destroyBuffer(point_light_buffer.buffer);
	gpu::destroyBuffer(spot_light_buffer.buffer);

	destroyMesh(cone_mesh);
	destroyMesh(plane_mesh);
	destroyMesh(cube_mesh);
	destroyMesh(spot_wire_mesh);

	gpu::shutdown();
}


void update(double time) {

	float dt = last_time < 0.0 ? 0.0f : float(time - last_time);
	last_time = time;
	dt = std::min(dt, 0.1f);

	updateCamera(dt);

	for (SceneObject& obj : objects) advanceAnimation(obj.anim, dt);
	for (PointLight& light : point_lights) advanceAnimation(light.anim, dt);
	for (SpotLight& light : spot_lights) advanceAnimation(light.anim, dt);
	if (sampling_effect.playing) sampling_effect.time += dt * sampling_effect.speed;

	drawUi();
}

void render(const graphics::internal::FrameData& fd) {
	auto& ctx = graphics::internal::context;
	VkCommandBuffer cmd = fd.command_buffer;
	if (cmd == VK_NULL_HANDLE) return;

	if (cone_dirty) {
		std::vector<mesh::Vertex> vertices;
		std::vector<uint16_t> indices;
		cone::build(cone_params, vertices, indices);
		uploadMesh(cone_mesh, vertices, indices);
		cone_dirty = false;
	}

	const VkExtent2D extent = ctx.swapchain_extent;
	const float aspect = extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;

	const SceneData scene = {
		.view = viewMatrix(),
		.proj = projectionMatrix(aspect),
		.camera_position = toVec4(camera.position, 1.0f),
		.ambient = toVec4(srgbToLinear(ambient.color), ambient.intensity),
		.dir_light_direction = toVec4(safeDirection(sun.direction), 0.0f),
		.dir_light_color = toVec4(srgbToLinear(sun.color), sun.intensity),
		.dir_light_enabled = sun.enabled ? 1u : 0u,
		.point_light_count = uploadPointLights(),
		.spot_light_count = uploadSpotLights(),
	};
	gpu::upload(scene_ubo, &scene, sizeof(scene));

	if (!reserveObjectBuffer(countDraws())) return;
	next_object_index = 0;

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin);

	VkClearValue clears[2] = {};
	clears[0].color = {{0.01f, 0.012f, 0.02f, 1.0f}};
	clears[1].depthStencil = {1.0f, 0};

	const VkRenderPassBeginInfo rp = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = ctx.render_pass,
		.framebuffer = fd.framebuffer,
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

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_descriptor_set, 0, nullptr);

	pushSampling(cmd, false, 0u);
	bindMesh(cmd, plane_mesh);

	if (room.floor.visible) {
		const float repeat = room.size / room.floor.tile_size;
		bindObject(cmd, makeObjectData(floorModelMatrix(), room.floor.material, repeat, repeat), room.floor.material.texture);
		vkCmdDrawIndexed(cmd, plane_mesh.index_count, 1, 0, 0, 0);
	}

	if (room.walls.visible) {
		const float repeat_u = room.size / room.walls.tile_size;
		const float repeat_v = room.wall_height / room.walls.tile_size;
		for (uint32_t side = 0; side < wall_count; ++side) {
			bindObject(cmd, makeObjectData(wallModelMatrix(side), room.walls.material, repeat_u, repeat_v),
			           room.walls.material.texture);
			vkCmdDrawIndexed(cmd, plane_mesh.index_count, 1, 0, 0, 0);
		}
	}

	bindMesh(cmd, cone_mesh);
	for (const SceneObject& obj : objects) {
		pushSampling(cmd, obj.special_sampling, obj.sampling_seed);
		bindObject(cmd, makeObjectData(modelMatrix(obj), obj.material), obj.material.texture);
		vkCmdDrawIndexed(cmd, cone_mesh.index_count, 1, 0, 0, 0);
	}

	if (light_viz.enabled) {
		drawLightMarkers(cmd);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

}
