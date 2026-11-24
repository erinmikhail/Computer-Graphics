#pragma once

#include <cstdint>
#include <vector>

#include "mesh.hpp"

namespace cone {

constexpr uint32_t min_segments = 3;
constexpr uint32_t max_segments = 128;

constexpr uint32_t max_vertices = 3 * max_segments + 2;
constexpr uint32_t max_indices = max_segments * 6;

struct Params {
	int segments = 32;
	float radius = 0.5f;
	float height = 1.0f;
};

void build(const Params& params, std::vector<mesh::Vertex>& vertices, std::vector<uint16_t>& indices);

}
