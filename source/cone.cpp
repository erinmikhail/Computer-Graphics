#include "cone.hpp"

#include <algorithm>
#include <cmath>

namespace cone {

void build(const Params& params, std::vector<mesh::Vertex>& vertices, std::vector<uint16_t>& indices) {
	const uint32_t n = uint32_t(std::clamp(params.segments, int(min_segments), int(max_segments)));
	const float r = params.radius;
	const float h = params.height;
	const float y_top = h * 0.5f;
	const float y_bottom = -h * 0.5f;

	vertices.clear();
	indices.clear();
	vertices.reserve(3 * n + 2);
	indices.reserve(n * 6);

	auto sideNormal = [&](float angle) {
		return normalize(vec3{ h * std::cos(angle), r, h * std::sin(angle) });
	};
	auto ringAngle = [&](float i) { return 2.0f * PI * i / float(n); };

	const uint16_t first_side = 0;
	for (uint32_t i = 0; i <= n; ++i) {
		const float angle = ringAngle(float(i));
		vertices.push_back({ { r * std::cos(angle), y_bottom, r * std::sin(angle) }, sideNormal(angle),
		                     { 1.0f - float(i) / float(n), 1.0f } });
	}

	const uint16_t first_apex = uint16_t(vertices.size());
	for (uint32_t i = 0; i < n; ++i) {
		vertices.push_back({ { 0.0f, y_top, 0.0f }, sideNormal(ringAngle(float(i) + 0.5f)),
		                     { 1.0f - (float(i) + 0.5f) / float(n), 0.0f } });
	}

	const vec3 down = { 0.0f, -1.0f, 0.0f };
	const uint16_t base_center = uint16_t(vertices.size());
	vertices.push_back({ { 0.0f, y_bottom, 0.0f }, down, { 0.5f, 0.5f } });

	const uint16_t first_base = uint16_t(vertices.size());
	for (uint32_t i = 0; i < n; ++i) {
		const float angle = ringAngle(float(i));
		const float c = std::cos(angle), s = std::sin(angle);
		vertices.push_back({ { r * c, y_bottom, r * s }, down, { 0.5f + 0.5f * c, 0.5f + 0.5f * s } });
	}

	for (uint32_t i = 0; i < n; ++i) {
		const uint32_t next = (i + 1) % n;

		indices.push_back(uint16_t(first_apex + i));
		indices.push_back(uint16_t(first_side + i + 1));
		indices.push_back(uint16_t(first_side + i));

		indices.push_back(base_center);
		indices.push_back(uint16_t(first_base + i));
		indices.push_back(uint16_t(first_base + next));
	}
}

}
