#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform ShadowTransform {
    mat4 mvp;
} shadow;

void main() {
    gl_Position = shadow.mvp * vec4(inPosition, 1.0);
}
