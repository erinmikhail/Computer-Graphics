#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene.glsl"
#include "object.glsl"

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = scene.proj * scene.view * object.model * vec4(inPosition, 1.0);
}
