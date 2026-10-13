#pragma once

#include <cstdint>
#include <vector>

#include "math.hpp"

namespace cone {

struct Vertex {
	vec3 position;
	vec3 color;
};

constexpr uint32_t min_segments = 3;
constexpr uint32_t max_segments = 128;

constexpr uint32_t max_vertices = max_segments + 2;
constexpr uint32_t max_indices = max_segments * 6;

struct Params {
	int segments = 32;
	float radius = 0.5f;
	float height = 1.0f;
};

void build(const Params& params, std::vector<Vertex>& vertices, std::vector<uint16_t>& indices);

} 
