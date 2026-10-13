#version 450

layout(set = 0, binding = 0, std140) uniform ObjectData {
    mat4 model;  // local -> world
    mat4 view;   // world -> камера
    mat4 proj;   // камера -> clip space
    vec4 color;  // цвет объекта из UI
    vec4 params; // x: 1 - использовать цвета вершин, 0 - нет
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    vColor = inColor;
}
