#include "cone.hpp"

#include <algorithm>
#include <cmath>

namespace cone {

namespace {

vec3 proceduralColor(vec3 p, float radius, float height) {
	return { p.x / (2.0f * radius) + 0.5f,
	         p.y / height + 0.5f,
	         p.z / (2.0f * radius) + 0.5f };
}

} 

void build(const Params& params, std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const uint32_t n = uint32_t(std::clamp(params.segments, int(min_segments), int(max_segments)));
	const float r = params.radius;
	const float h = params.height;

	vertices.clear();
	indices.clear();
	vertices.reserve(n + 2);
	indices.reserve(n * 6);

	auto push = [&](vec3 p) { vertices.push_back({ p, proceduralColor(p, r, h) }); };

	constexpr uint16_t apex = 0;
	constexpr uint16_t base_center = 1;
	constexpr uint16_t first_ring = 2;

	push({ 0.0f,  h * 0.5f, 0.0f });
	push({ 0.0f, -h * 0.5f, 0.0f });

	for (uint32_t i = 0; i < n; ++i) {
		const float angle = 2.0f * PI * float(i) / float(n);

		push({ r * std::cos(angle), -h * 0.5f, r * std::sin(angle) });
	}

	for (uint32_t i = 0; i < n; ++i) {
		const uint16_t cur = uint16_t(first_ring + i);
		const uint16_t next = uint16_t(first_ring + (i + 1) % n);

		indices.push_back(apex);
		indices.push_back(next);
		indices.push_back(cur);

		indices.push_back(base_center);
		indices.push_back(cur);
		indices.push_back(next);
	}
}

}
