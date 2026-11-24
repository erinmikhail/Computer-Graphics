#pragma once

#include <cstdint>
#include <vector>

#include "math.hpp"

namespace mesh {

struct Vertex {
	vec3 position;
	vec3 normal;
	vec2 uv;
};

void buildPlane(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices);

void buildCube(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices);

void buildSpotWireframe(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices);

}
