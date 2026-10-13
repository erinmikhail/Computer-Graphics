#pragma once

#include "math.hpp"

namespace trajectory {

enum class Type : int {
	Lissajous = 0, // фигура Лиссажу в 3D
	Trefoil,       // узел-трилистник
	Spiral,        // спираль, которая вверх-вниз
};

inline const char* const type_names[] = { "Фигура Лиссажу", "Узел-трилистник", "Спираль" };

struct Params {
	Type type = Type::Lissajous;
	float radius = 1.0f; // общий размах траектории
	float freq_a = 3.0f; // частота по X (для трилистника/спирали — множитель скорости обхода)
	float freq_b = 2.0f; // частота по Y
	float freq_c = 1.0f; // частота по Z
	float phase = 0.5f;  // сдвиг фазы по X (для Лиссажу меняет форму кривой)
};

// Точка траектории в момент t 
inline vec3 evaluate(const Params& p, float t) {
	switch (p.type) {
	case Type::Lissajous:

		return { p.radius * std::sin(p.freq_a * t + p.phase),
		         p.radius * std::sin(p.freq_b * t),
		         p.radius * std::sin(p.freq_c * t) };

	case Type::Trefoil: {

		const float s = p.freq_a * t;
		const float k = p.radius / 3.0f;
		return { k * (std::sin(s) + 2.0f * std::sin(2.0f * s)),
		         k * (-std::sin(3.0f * s)),
		         k * (std::cos(s) - 2.0f * std::cos(2.0f * s)) };
	}

	case Type::Spiral: {

		const float s = p.freq_a * t;
		return { p.radius * std::cos(s),
		         p.radius * std::sin(p.freq_b * t * 0.25f),
		         p.radius * std::sin(s) };
	}
	}
	return { 0.0f, 0.0f, 0.0f };
}

inline mat4 orientationAlongPath(const Params& p, float t) {
	constexpr float eps = 1e-3f;
	const vec3 d = evaluate(p, t + eps) - evaluate(p, t);
	if (length(d) < 1e-7f) return identity();

	const vec3 y = normalize(d);
	const vec3 helper = std::abs(y.z) < 0.99f ? vec3{ 0.0f, 0.0f, 1.0f } : vec3{ 1.0f, 0.0f, 0.0f };
	const vec3 x = normalize(cross(y, helper));
	const vec3 z = cross(x, y); // x × y = z — правая тройка, без отражения

	return {{ { x.x, x.y, x.z, 0 },
	          { y.x, y.y, y.z, 0 },
	          { z.x, z.y, z.z, 0 },
	          { 0,   0,   0,   1 } }};
}

}
