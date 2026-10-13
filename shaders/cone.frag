#version 450

layout(set = 0, binding = 0, std140) uniform ObjectData {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 color;
    vec4 params;
} ubo;

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 vertex_color = mix(vec3(1.0), vColor, ubo.params.x);
    outColor = vec4(vertex_color * ubo.color.rgb, ubo.color.a);
}
