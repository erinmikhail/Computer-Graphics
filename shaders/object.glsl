layout(set = 1, binding = 0, std140) uniform ObjectData {
    mat4 model;
    mat3 normalMatrix;
    vec4 albedo;
    vec4 specular;
    vec4 uvScaleEmission;
} object;
