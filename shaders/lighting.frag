#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene.glsl"
#include "object.glsl"

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    float cosInner;
    float cosOuter;
    vec2 padding;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(set = 0, binding = 3) uniform sampler2DShadow sunShadowMap;

layout(set = 1, binding = 1) uniform sampler2D albedoMap;
layout(set = 1, binding = 2) uniform sampler2D specularMap;
layout(set = 1, binding = 3) uniform sampler2D emissiveMap;

layout(push_constant) uniform SamplingParams {
    uint enabled;
    float time;
    float distortion;
    float layerMix;
    uint seed;
} sampling;

layout(location = 0) in vec3 vWorldPosition;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec2 vUV;

layout(location = 0) out vec4 outColor;

const float minDistanceSq = 1e-4;

float random01(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return float(value >> 8u) / 16777216.0;
}

vec3 sampleAlbedo(vec2 uv) {
    if (sampling.enabled == 0u) {
        return texture(albedoMap, uv).rgb;
    }

    float t = sampling.time;

    vec2 wave = sampling.distortion * vec2(sin(uv.y * 20.0 + t * 3.0), cos(uv.x * 20.0 + t * 2.0));
    vec3 a = texture(albedoMap, uv + wave).rgb;

    const float tileCount = 4.0;
    uvec2 tile = uvec2(clamp(floor(uv * tileCount), 0.0, tileCount - 1.0));
    uint tileIndex = tile.x + tile.y * uint(tileCount);
    uint tileSeed = sampling.seed ^ (tileIndex * 0x9e3779b9u);
    float phase = random01(tileSeed) * 6.2831853;
    vec2 offset = vec2(
        random01(tileSeed ^ 0x9e3779b9u),
        random01(tileSeed ^ 0x85ebca6bu)
    );

    float angle = t * 0.3 + phase;
    mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    vec2 tileCenter = (vec2(tile) + 0.5) / tileCount;
    vec2 secondUV = rotation * (uv - tileCenter) * 2.0 + vec2(t * 0.1, 0.0) + offset;
    vec2 secondDx = rotation * dFdx(uv) * 2.0;
    vec2 secondDy = rotation * dFdy(uv) * 2.0;
    vec3 b = textureGrad(albedoMap, secondUV, secondDx, secondDy).rgb;
    vec3 tileTint = 0.65 + 0.35 * cos(phase + vec3(0.0, 2.0943951, 4.1887902));

    vec3 tint = 0.6 + 0.4 * cos(6.28318 * (t * 0.2 + vec3(0.0, 0.33, 0.67)));

    return mix(a, b * tileTint, sampling.layerMix) * tint;
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, vec3 specularColor) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL == 0.0) {
        return vec3(0.0);
    }

    vec3 H = normalize(L + V);
    vec3 diffuse = albedo * NdotL;
    vec3 specular = specularColor * pow(max(dot(N, H), 0.0), object.specular.a);
    return (diffuse + specular) * radiance;
}

float attenuation(float distanceSq) {
    return 1.0 / max(distanceSq, minDistanceSq);
}

float shadowVisibility(sampler2DShadow depthMap, mat4 lightViewProj, vec3 worldPosition) {
    vec4 clip = lightViewProj * vec4(worldPosition, 1.0);
    if (clip.w <= 0.0) return 1.0;
    vec3 ndc = clip.xyz / clip.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (ndc.z <= 0.0 || ndc.z >= 1.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;

    vec2 texel = 1.0 / vec2(textureSize(depthMap, 0));
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            visibility += textureLod(depthMap, vec3(uv + vec2(x, y) * texel, ndc.z), 0.0);
        }
    }
    return visibility / 9.0;
}

void main() {
    vec3 N = normalize(vWorldNormal);
    vec3 V = normalize(scene.cameraPosition.xyz - vWorldPosition);

    vec3 albedo = object.albedo.rgb * sampleAlbedo(vUV);
    vec3 specularColor = object.specular.rgb * texture(specularMap, vUV).rgb;
    vec3 emissive = texture(emissiveMap, vUV).rgb * object.uvScaleEmission.z;

    vec3 color = scene.ambient.rgb * scene.ambient.a * albedo;

    if (scene.dirLightEnabled != 0u) {
        vec3 L = normalize(-scene.dirLightDirection.xyz);
        float visibility = scene.sunShadowEnabled != 0u ? shadowVisibility(sunShadowMap, scene.sunViewProj, vWorldPosition) : 1.0;
        color += visibility * blinnPhong(N, V, L, scene.dirLightColor.rgb * scene.dirLightColor.a, albedo, specularColor);
    }

    for (uint i = 0u; i < scene.pointLightCount; ++i) {
        PointLight light = pointLights[i];
        vec3 toLight = light.position.xyz - vWorldPosition;
        float distanceSq = dot(toLight, toLight);
        vec3 L = toLight * inversesqrt(max(distanceSq, minDistanceSq));

        vec3 radiance = light.color.rgb * light.color.a * attenuation(distanceSq);
        color += blinnPhong(N, V, L, radiance, albedo, specularColor);
    }

    for (uint i = 0u; i < scene.spotLightCount; ++i) {
        SpotLight light = spotLights[i];
        vec3 toLight = light.position.xyz - vWorldPosition;
        float distanceSq = dot(toLight, toLight);
        vec3 L = toLight * inversesqrt(max(distanceSq, minDistanceSq));

        float cosAngle = dot(-L, light.direction.xyz);
        float spot = light.cosInner > light.cosOuter
            ? smoothstep(light.cosOuter, light.cosInner, cosAngle)
            : step(light.cosOuter, cosAngle);

        vec3 radiance = light.color.rgb * light.color.a * attenuation(distanceSq) * spot;
        color += blinnPhong(N, V, L, radiance, albedo, specularColor);
    }

    color += emissive;

    outColor = vec4(color, 1.0);
}
