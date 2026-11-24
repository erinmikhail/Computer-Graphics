#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene.glsl"
#include "object.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 vWorldPosition;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec2 vUV;

void main() {
    vec4 world = object.model * vec4(inPosition, 1.0);
    vWorldPosition = world.xyz;
    vWorldNormal = object.normalMatrix * inNormal;
    vUV = inUV * object.uvScaleEmission.xy;
    gl_Position = scene.proj * scene.view * world;
}
