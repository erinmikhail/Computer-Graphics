#include "mesh.hpp"

namespace mesh {

void buildPlane(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const vec3 up = { 0.0f, 1.0f, 0.0f };

	vertices = {
		{ { -0.5f, 0.0f, -0.5f }, up, { 0.0f, 0.0f } },
		{ {  0.5f, 0.0f, -0.5f }, up, { 1.0f, 0.0f } },
		{ {  0.5f, 0.0f,  0.5f }, up, { 1.0f, 1.0f } },
		{ { -0.5f, 0.0f,  0.5f }, up, { 0.0f, 1.0f } },
	};
	indices = { 0, 2, 1, 0, 3, 2 };
}

void buildCube(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const vec3 faces[6][2] = {
		{ {  1,  0,  0 }, { 0, 1, 0 } },
		{ { -1,  0,  0 }, { 0, 0, 1 } },
		{ {  0,  1,  0 }, { 0, 0, 1 } },
		{ {  0, -1,  0 }, { 1, 0, 0 } },
		{ {  0,  0,  1 }, { 1, 0, 0 } },
		{ {  0,  0, -1 }, { 0, 1, 0 } },
	};

	vertices.clear();
	indices.clear();

	for (const auto& face : faces) {
		const vec3 n = face[0];
		const vec3 u = face[1] * 0.5f;
		const vec3 v = cross(n, face[1]) * 0.5f;
		const vec3 center = n * 0.5f;
		const uint16_t base = uint16_t(vertices.size());

		vertices.push_back({ center - u - v, n, { 0.0f, 1.0f } });
		vertices.push_back({ center + u - v, n, { 1.0f, 1.0f } });
		vertices.push_back({ center + u + v, n, { 1.0f, 0.0f } });
		vertices.push_back({ center - u + v, n, { 0.0f, 0.0f } });

		indices.insert(indices.end(), { base, uint16_t(base + 1), uint16_t(base + 2),
		                                base, uint16_t(base + 2), uint16_t(base + 3) });
	}
}

void buildSpotWireframe(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	constexpr uint32_t segments = 32;
	constexpr uint32_t ribs = 8;
	const vec3 no_normal = { 0.0f, 0.0f, 0.0f };

	vertices.clear();
	indices.clear();

	constexpr uint16_t apex = 0;
	constexpr uint16_t first_ring = 1;
	vertices.push_back({ { 0.0f, 0.0f, 0.0f }, no_normal, {} });

	for (uint32_t i = 0; i < segments; ++i) {
		const float angle = 2.0f * PI * float(i) / float(segments);
		vertices.push_back({ { std::cos(angle), 1.0f, std::sin(angle) }, no_normal, {} });
	}

	const uint16_t axis_end = uint16_t(vertices.size());
	vertices.push_back({ { 0.0f, 1.0f, 0.0f }, no_normal, {} });

	for (uint32_t i = 0; i < segments; ++i) {
		const uint16_t cur = uint16_t(first_ring + i);
		const uint16_t next = uint16_t(first_ring + (i + 1) % segments);
		indices.insert(indices.end(), { cur, next });

		if (i % (segments / ribs) == 0) {
			indices.insert(indices.end(), { apex, cur });
		}
	}
	indices.insert(indices.end(), { apex, axis_end });
}

}
