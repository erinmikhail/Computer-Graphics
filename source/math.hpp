#pragma once

#include <cmath>

constexpr float PI = 3.14159265358979f;

inline float radians(float degrees) { return degrees * PI / 180.0f; }

//Векторы

struct vec3 { float x, y, z; };
struct vec4 { float x, y, z, w; };

inline vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline vec3 operator*(vec3 v, float s) { return {v.x * s, v.y * s, v.z * s}; }

inline float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline vec3 cross(vec3 a, vec3 b) {
	return { a.y * b.z - a.z * b.y,
	         a.z * b.x - a.x * b.z,
	         a.x * b.y - a.y * b.x };
}

inline float length(vec3 v) { return std::sqrt(dot(v, v)); }
inline vec3 normalize(vec3 v) { return v * (1.0f / length(v)); }

inline vec4 operator+(vec4 a, vec4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline vec4 operator*(vec4 v, float s) { return {v.x * s, v.y * s, v.z * s, v.w * s}; }


struct mat4 { vec4 c[4]; };
static_assert(sizeof(mat4) == 64, "mat4 должна занимать ровно 16 float подряд");

inline vec4 operator*(const mat4& m, vec4 v) {
	return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z + m.c[3] * v.w;
}

// (A·B)
inline mat4 operator*(const mat4& a, const mat4& b) {
	return {{ a * b.c[0], a * b.c[1], a * b.c[2], a * b.c[3] }};
}

inline mat4 identity() {
	return {{ {1, 0, 0, 0},
	          {0, 1, 0, 0},     
	          {0, 0, 1, 0},     
	          {0, 0, 0, 1} }};  
}

//Аффинные преобразования

inline mat4 translate(vec3 t) {
	return {{ {1, 0, 0, 0},
	          {0, 1, 0, 0},
	          {0, 0, 1, 0},
	          {t.x, t.y, t.z, 1} }}; 
}


inline mat4 scale(vec3 s) {
	return {{ {s.x, 0,   0,   0},
              {0,   s.y, 0,   0},
              {0,   0,   s.z, 0},
              {0,   0,   0,   1} }}; 
}

// Поворот вокруг Y
inline mat4 rotateY(float a) {
	const float c = std::cos(a), s = std::sin(a);
	return {{ { c, 0, -s, 0},     
	          { 0, 1,  0, 0}, 
	          { s, 0,  c, 0}, 
	          { 0, 0,  0, 1} }};
}


inline mat4 rotateX(float a) { 
    const float c = std::cos(a), s = std::sin(a);
    return {{ {1, 0, 0, 0},
              {0, c, s, 0},
              {0, -s, c, 0},
              {0, 0, 0, 1} }};
 }


inline mat4 rotateZ(float a) {
    const float c = std::cos(a), s = std::sin(a);
    return {{ {c, s, 0, 0},
              {-s, c, 0, 0},
              {0, 0, 1, 0},
              {0, 0, 0, 1} }};
}

//Проекция

inline mat4 perspective(float fovY, float aspect, float n, float f) {
	const float k = 1.0f / std::tan(fovY / 2.0f);
	return {{ {k / aspect, 0,  0,                  0},
	          {0,         -k,  0,                  0},
	          {0,          0,  f / (n - f),       -1},
	          {0,          0,  n * f / (n - f),    0} }};
}

inline mat4 ortho(float l, float r, float b, float t, float n, float f) {
	return {{ {2.0f / (r - l),     0,                   0,                0},
	          {0,                 -2.0f / (t - b),      0,                0},
	          {0,                  0,                  -1.0f / (f - n),   0},
	          {-(r + l) / (r - l), (t + b) / (t - b),  -n / (f - n),      1} }};
}

inline mat4 lookAt(vec3 eye, vec3 center, vec3 up) {
	const vec3 f = normalize(center - eye);
	const vec3 s = normalize(cross(f, up));
	const vec3 u = cross(s, f);
	return {{ {s.x, u.x, -f.x, 0},
	          {s.y, u.y, -f.y, 0},
	          {s.z, u.z, -f.z, 0},
	          {-dot(s, eye), -dot(u, eye), dot(f, eye), 1} }};
}