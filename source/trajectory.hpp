#pragma once

#include "math.hpp"

namespace trajectory {

enum class Type : int {
	Lissajous = 0,
	Trefoil,
	Spiral,
	Circle,
};

inline const char* const type_names[] = { "Фигура Лиссажу", "Узел-трилистник", "Спираль", "Окружность" };

struct Params {
	Type type = Type::Lissajous;
	float radius = 1.0f;
	float freq_a = 3.0f;
	float freq_b = 2.0f;
	float freq_c = 1.0f;
	float phase = 0.5f;
};

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

	case Type::Circle: {
		const float s = p.freq_a * t;
		return { p.radius * std::cos(s), 0.0f, p.radius * std::sin(s) };
	}
	}
	return { 0.0f, 0.0f, 0.0f };
}

inline mat4 orientationAlongPath(const Params& p, float t) {
	constexpr float eps = 1e-3f;
	const vec3 d = evaluate(p, t + eps) - evaluate(p, t);
	if (length(d) < 1e-7f) return identity();

	return orientY(d);
}

}
