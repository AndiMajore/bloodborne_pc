// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the ray-traced AO's surface cache, read at a ray hit (vk_rtao_surface.cpp). A hit on the
// static instance (custom index 1) maps to its pool triangle by the static BLAS's geometry map;
// returns the radiance and whether the triangle was ever seen. Needs GL_EXT_buffer_reference,
// GL_EXT_shader_explicit_arithmetic_types_int64 and GL_EXT_ray_query.
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer SurfaceParts { uvec2 p[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer SurfaceCache { uvec2 c[]; };

// The same from a hit's ids (the AO trace's records of shared rays, ticket 69).
bool SurfaceRadianceIds(uint custom_index, uint geometry, uint primitive, uint64_t cache_address,
                        uint64_t parts_address, uint part_count, out vec3 radiance) {
    radiance = vec3(0.0);
    if (cache_address == 0ul || custom_index != 1u || geometry >= part_count) {
        return false;
    }
    const uint tri = SurfaceParts(parts_address).p[geometry].x + primitive;
    const uvec2 v = SurfaceCache(cache_address).c[tri];
    const vec2 ba = unpackHalf2x16(v.y);
    if (ba.y <= 0.0) {
        return false;
    }
    radiance = vec3(unpackHalf2x16(v.x), ba.x);
    return true;
}

bool SurfaceRadiance(rayQueryEXT q, uint64_t cache_address, uint64_t parts_address, uint part_count,
                     out vec3 radiance) {
    radiance = vec3(0.0);
    if (cache_address == 0ul || rayQueryGetIntersectionInstanceCustomIndexEXT(q, true) != 1) {
        return false;
    }
    const uint geometry = uint(rayQueryGetIntersectionGeometryIndexEXT(q, true));
    if (geometry >= part_count) {
        return false;
    }
    const uint tri = SurfaceParts(parts_address).p[geometry].x +
                     uint(rayQueryGetIntersectionPrimitiveIndexEXT(q, true));
    const uvec2 v = SurfaceCache(cache_address).c[tri];
    const vec2 ba = unpackHalf2x16(v.y);
    if (ba.y <= 0.0) {
        return false;
    }
    radiance = vec3(unpackHalf2x16(v.x), ba.x);
    return true;
}
