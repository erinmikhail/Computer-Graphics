layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view;
    mat4 proj;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 dirLightDirection;
    vec4 dirLightColor;
    uint dirLightEnabled;
    uint pointLightCount;
    uint spotLightCount;
    uint sunShadowEnabled;
    mat4 sunViewProj;
} scene;
