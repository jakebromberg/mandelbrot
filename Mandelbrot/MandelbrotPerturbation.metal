//
//  MandelbrotPerturbation.metal
//  Mandelbrot
//
//  Perturbation theory shader for deep zooms beyond single-precision limits.
//  Uses a precomputed reference orbit to calculate pixel values as small
//  perturbations from the reference, allowing Float precision to suffice.
//
//  Double-precision values are passed as pairs of floats (hi, lo) where:
//  - hi contains the high-order bits
//  - lo contains the low-order bits (the error from float conversion)
//  This "double-float" representation allows ~48 bits of precision.
//

#include <metal_stdlib>
using namespace metal;

struct PerturbationParams {
    // Reference center as double-float pairs: (hi_x, lo_x, hi_y, lo_y)
    float4 referenceCenterHiLo;
    // View center as double-float pairs: (hi_x, lo_x, hi_y, lo_y)
    float4 viewCenterHiLo;
    // Scale as double-float pair: (hi, lo)
    float2 scaleHiLo;
    // Scale * aspect as double-float pair: (hi, lo)
    float2 scaleAspectHiLo;
    uint width;               // Image width (in pixels)
    uint height;              // Image height (in pixels)
    uint maxIterations;       // Maximum iterations
    uint orbitLength;         // Length of reference orbit
    uint colorMode;           // 0 = HSV, 1 = Palette
    uint skipIterations;      // Number of iterations to skip via series approximation
};

// Glitch detection threshold: |δ|² > ε·|z_ref|² indicates numerical instability
constant float kGlitchThreshold = 1e-6f;

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
inline float2 complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

// Converts HSV values (each in the range [0, 1]) to an RGB color.
half3 hsv_to_rgb_perturb(half h, half s, half v) {
    half c = v * s;
    float h_prime_f = float(h) * 6.0f;
    float mod_val = fmod(h_prime_f, 2.0f);
    half x = c * (1.0 - fabs(half(mod_val) - 1.0));

    half3 rgb;
    if (h_prime_f < 1.0f) {
        rgb = half3(c, x, 0);
    } else if (h_prime_f < 2.0f) {
        rgb = half3(x, c, 0);
    } else if (h_prime_f < 3.0f) {
        rgb = half3(0, c, x);
    } else if (h_prime_f < 4.0f) {
        rgb = half3(0, x, c);
    } else if (h_prime_f < 5.0f) {
        rgb = half3(x, 0, c);
    } else {
        rgb = half3(c, 0, x);
    }

    half m = v - c;
    return rgb + half3(m, m, m);
}

constant float kInvLog2Perturb = 1.442695f; // 1 / log(2)

// Double-float addition: (a_hi, a_lo) + (b_hi, b_lo) -> result as float
// This computes the sum with extended precision, returning the high part
inline float df_add_to_float(float a_hi, float a_lo, float b_hi, float b_lo) {
    // For computing delta_c, we need (pixel - reference)
    // pixel = view_center + offset
    // delta_c = (view_center + offset) - reference_center
    // The offset is small relative to pixel, so standard float suffices for final delta
    float sum_hi = a_hi + b_hi;
    float sum_lo = a_lo + b_lo;
    return sum_hi + sum_lo;
}

kernel void mandelbrotPerturbationKernel(
    texture2d<half, access::write> output [[texture(0)]],
    texture2d<half> paletteTex [[texture(1)]],
    constant PerturbationParams &params [[buffer(0)]],
    constant float2 *referenceOrbit [[buffer(1)]],
    sampler paletteSampler [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Ensure we're within bounds.
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    // Compute normalized pixel coordinates
    float2 coord = float2(gid) / float2(params.width, params.height);

    // Compute the offset from view center in the complex plane
    // offset = (coord - 0.5) * scale
    float offset_x = (coord.x - 0.5f) * params.scaleAspectHiLo.x;
    float offset_y = (coord.y - 0.5f) * params.scaleHiLo.x;

    // Compute delta_c = (viewCenter + offset) - referenceCenter
    // Since offset is small at deep zoom, and we only need delta_c as float,
    // we compute: delta_c = (viewCenter - referenceCenter) + offset
    // The (viewCenter - referenceCenter) is precomputable but small, so we do it here
    float delta_c_x = (params.viewCenterHiLo.x - params.referenceCenterHiLo.x)
                    + (params.viewCenterHiLo.y - params.referenceCenterHiLo.y)
                    + offset_x;
    float delta_c_y = (params.viewCenterHiLo.z - params.referenceCenterHiLo.z)
                    + (params.viewCenterHiLo.w - params.referenceCenterHiLo.w)
                    + offset_y;

    float2 delta_c = float2(delta_c_x, delta_c_y);

    // Initialize perturbation delta
    float2 delta = float2(0.0);

    uint iteration = 0;
    const uint maxIter = min(params.maxIterations, params.orbitLength);

    // Perturbation iteration:
    // The full z value at each iteration is z_full = z_ref + delta
    // The recurrence: δₙ₊₁ = 2·zₙ·δₙ + δₙ² + δc
    for (uint i = 0; i < maxIter; i++) {
        float2 z_ref = referenceOrbit[i];
        float2 z_full = z_ref + delta;

        // Check for escape using the full z value
        float mag_sq = dot(z_full, z_full);
        if (mag_sq > 65536.0f) {
            break;
        }

        // Perturbation formula: δₙ₊₁ = 2·zₙ·δₙ + δₙ² + δc
        // = 2 * z_ref * delta + delta * delta + delta_c
        float2 two_z_ref = 2.0f * z_ref;
        float2 term1 = complex_mul(two_z_ref, delta);
        float2 term2 = complex_mul(delta, delta);
        delta = term1 + term2 + delta_c;

        iteration++;
    }

    // If the point didn't escape within the orbit length, color it black
    if (iteration >= maxIter) {
        output.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    // Compute final |z|² for smooth coloring
    float2 z_ref_final = (iteration < params.orbitLength) ? referenceOrbit[iteration] : float2(0.0);
    float2 z_full_final = z_ref_final + delta;
    float final_mag_sq = dot(z_full_final, z_full_final);

    // Smooth the iteration count
    float iteration_f = float(iteration);
    float log_zn = fast::log(final_mag_sq) * 0.5f;
    float nu = fast::log(log_zn * kInvLog2Perturb) * kInvLog2Perturb;
    iteration_f = iteration_f + 1.0f - nu;

    // Normalize for color mapping
    half normIter = half(iteration_f / float(params.maxIterations));

    half4 color;
    if (params.colorMode == 0u) {
        // HSV mode
        half hue = half(pow(float(normIter), 0.3333f));
        half saturation = 1.0;
        half value = 1.0;
        half3 rgb = hsv_to_rgb_perturb(hue, saturation, value);
        color = half4(rgb, 1.0);
    } else {
        // Palette LUT mode
        half u = clamp(normIter, half(0.0), half(0.9999));
        half v = half(0.5);
        color = paletteTex.sample(paletteSampler, float2(u, v));
        color.a = half(1.0);
    }

    output.write(color, gid);
}

// Kernel with series approximation for skipping early iterations.
// Series: δₙ ≈ Aₙ·δc + Bₙ·δc² allows starting from skipIterations instead of 0.
kernel void mandelbrotPerturbationWithSeriesKernel(
    texture2d<half, access::write> output [[texture(0)]],
    texture2d<half> paletteTex [[texture(1)]],
    constant PerturbationParams &params [[buffer(0)]],
    constant float2 *referenceOrbit [[buffer(1)]],
    constant float2 *seriesA [[buffer(2)]],
    constant float2 *seriesB [[buffer(3)]],
    sampler paletteSampler [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Ensure we're within bounds.
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    // Compute normalized pixel coordinates
    float2 coord = float2(gid) / float2(params.width, params.height);

    // Compute the offset from view center in the complex plane
    float offset_x = (coord.x - 0.5f) * params.scaleAspectHiLo.x;
    float offset_y = (coord.y - 0.5f) * params.scaleHiLo.x;

    // Compute delta_c = (viewCenter + offset) - referenceCenter
    float delta_c_x = (params.viewCenterHiLo.x - params.referenceCenterHiLo.x)
                    + (params.viewCenterHiLo.y - params.referenceCenterHiLo.y)
                    + offset_x;
    float delta_c_y = (params.viewCenterHiLo.z - params.referenceCenterHiLo.z)
                    + (params.viewCenterHiLo.w - params.referenceCenterHiLo.w)
                    + offset_y;

    float2 delta_c = float2(delta_c_x, delta_c_y);

    // Use series approximation to initialize delta at skipIterations
    // δₙ ≈ Aₙ·δc + Bₙ·δc²
    uint skipIter = min(params.skipIterations, params.orbitLength);
    float2 delta;

    if (skipIter > 0 && skipIter <= params.orbitLength) {
        uint seriesIdx = skipIter - 1;
        float2 A = seriesA[seriesIdx];
        float2 B = seriesB[seriesIdx];
        float2 dc_sq = complex_mul(delta_c, delta_c);
        delta = complex_mul(A, delta_c) + complex_mul(B, dc_sq);
    } else {
        delta = float2(0.0);
        skipIter = 0;
    }

    uint iteration = skipIter;
    const uint maxIter = min(params.maxIterations, params.orbitLength);

    // Continue perturbation iteration from skipIter
    for (uint i = skipIter; i < maxIter; i++) {
        float2 z_ref = referenceOrbit[i];
        float2 z_full = z_ref + delta;

        // Check for escape
        float mag_sq = dot(z_full, z_full);
        if (mag_sq > 65536.0f) {
            break;
        }

        // Perturbation formula: δₙ₊₁ = 2·zₙ·δₙ + δₙ² + δc
        float2 two_z_ref = 2.0f * z_ref;
        float2 term1 = complex_mul(two_z_ref, delta);
        float2 term2 = complex_mul(delta, delta);
        delta = term1 + term2 + delta_c;

        iteration++;
    }

    // If the point didn't escape, color it black
    if (iteration >= maxIter) {
        output.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    // Compute final |z|² for smooth coloring
    float2 z_ref_final = (iteration < params.orbitLength) ? referenceOrbit[iteration] : float2(0.0);
    float2 z_full_final = z_ref_final + delta;
    float final_mag_sq = dot(z_full_final, z_full_final);

    // Smooth the iteration count
    float iteration_f = float(iteration);
    float log_zn = fast::log(final_mag_sq) * 0.5f;
    float nu = fast::log(log_zn * kInvLog2Perturb) * kInvLog2Perturb;
    iteration_f = iteration_f + 1.0f - nu;

    // Normalize for color mapping
    half normIter = half(iteration_f / float(params.maxIterations));

    half4 color;
    if (params.colorMode == 0u) {
        // HSV mode
        half hue = half(pow(float(normIter), 0.3333f));
        half saturation = 1.0;
        half value = 1.0;
        half3 rgb = hsv_to_rgb_perturb(hue, saturation, value);
        color = half4(rgb, 1.0);
    } else {
        // Palette LUT mode
        half u = clamp(normIter, half(0.0), half(0.9999));
        half v = half(0.5);
        color = paletteTex.sample(paletteSampler, float2(u, v));
        color.a = half(1.0);
    }

    output.write(color, gid);
}

// Kernel with glitch detection that writes glitch info to a buffer.
// Glitch condition: |δ|² > ε·|z_ref|² indicates numerical instability.
// Output buffer stores iteration where glitch was detected (0 if no glitch).
kernel void mandelbrotPerturbationWithGlitchDetectionKernel(
    texture2d<half, access::write> output [[texture(0)]],
    texture2d<half> paletteTex [[texture(1)]],
    constant PerturbationParams &params [[buffer(0)]],
    constant float2 *referenceOrbit [[buffer(1)]],
    device uint *glitchBuffer [[buffer(2)]],
    sampler paletteSampler [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    uint pixelIdx = gid.y * params.width + gid.x;

    float2 coord = float2(gid) / float2(params.width, params.height);

    float offset_x = (coord.x - 0.5f) * params.scaleAspectHiLo.x;
    float offset_y = (coord.y - 0.5f) * params.scaleHiLo.x;

    float delta_c_x = (params.viewCenterHiLo.x - params.referenceCenterHiLo.x)
                    + (params.viewCenterHiLo.y - params.referenceCenterHiLo.y)
                    + offset_x;
    float delta_c_y = (params.viewCenterHiLo.z - params.referenceCenterHiLo.z)
                    + (params.viewCenterHiLo.w - params.referenceCenterHiLo.w)
                    + offset_y;

    float2 delta_c = float2(delta_c_x, delta_c_y);
    float2 delta = float2(0.0);

    uint iteration = 0;
    const uint maxIter = min(params.maxIterations, params.orbitLength);

    for (uint i = 0; i < maxIter; i++) {
        float2 z_ref = referenceOrbit[i];
        float2 z_full = z_ref + delta;

        float mag_sq = dot(z_full, z_full);
        if (mag_sq > 65536.0f) {
            break;
        }

        // Glitch detection: |δ|² > ε·|z_ref|²
        float delta_mag_sq = dot(delta, delta);
        float z_ref_mag_sq = dot(z_ref, z_ref);
        if (delta_mag_sq > kGlitchThreshold * z_ref_mag_sq && z_ref_mag_sq > 1e-20f) {
            glitchBuffer[pixelIdx] = i;
            output.write(half4(1.0, 0.0, 1.0, 1.0), gid);  // Magenta = glitch
            return;
        }

        float2 two_z_ref = 2.0f * z_ref;
        float2 term1 = complex_mul(two_z_ref, delta);
        float2 term2 = complex_mul(delta, delta);
        delta = term1 + term2 + delta_c;

        iteration++;
    }

    glitchBuffer[pixelIdx] = 0;

    if (iteration >= maxIter) {
        output.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    float2 z_ref_final = (iteration < params.orbitLength) ? referenceOrbit[iteration] : float2(0.0);
    float2 z_full_final = z_ref_final + delta;
    float final_mag_sq = dot(z_full_final, z_full_final);

    float iteration_f = float(iteration);
    float log_zn = fast::log(final_mag_sq) * 0.5f;
    float nu = fast::log(log_zn * kInvLog2Perturb) * kInvLog2Perturb;
    iteration_f = iteration_f + 1.0f - nu;

    half normIter = half(iteration_f / float(params.maxIterations));

    half4 color;
    if (params.colorMode == 0u) {
        half hue = half(pow(float(normIter), 0.3333f));
        half3 rgb = hsv_to_rgb_perturb(hue, half(1.0), half(1.0));
        color = half4(rgb, 1.0);
    } else {
        half u_tex = clamp(normIter, half(0.0), half(0.9999));
        color = paletteTex.sample(paletteSampler, float2(u_tex, 0.5));
        color.a = half(1.0);
    }

    output.write(color, gid);
}

// Kernel with both series approximation and glitch detection.
kernel void mandelbrotPerturbationWithSeriesAndGlitchKernel(
    texture2d<half, access::write> output [[texture(0)]],
    texture2d<half> paletteTex [[texture(1)]],
    constant PerturbationParams &params [[buffer(0)]],
    constant float2 *referenceOrbit [[buffer(1)]],
    constant float2 *seriesA [[buffer(2)]],
    constant float2 *seriesB [[buffer(3)]],
    device uint *glitchBuffer [[buffer(4)]],
    sampler paletteSampler [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    uint pixelIdx = gid.y * params.width + gid.x;

    float2 coord = float2(gid) / float2(params.width, params.height);

    float offset_x = (coord.x - 0.5f) * params.scaleAspectHiLo.x;
    float offset_y = (coord.y - 0.5f) * params.scaleHiLo.x;

    float delta_c_x = (params.viewCenterHiLo.x - params.referenceCenterHiLo.x)
                    + (params.viewCenterHiLo.y - params.referenceCenterHiLo.y)
                    + offset_x;
    float delta_c_y = (params.viewCenterHiLo.z - params.referenceCenterHiLo.z)
                    + (params.viewCenterHiLo.w - params.referenceCenterHiLo.w)
                    + offset_y;

    float2 delta_c = float2(delta_c_x, delta_c_y);

    uint skipIter = min(params.skipIterations, params.orbitLength);
    float2 delta;

    if (skipIter > 0 && skipIter <= params.orbitLength) {
        uint seriesIdx = skipIter - 1;
        float2 A = seriesA[seriesIdx];
        float2 B = seriesB[seriesIdx];
        float2 dc_sq = complex_mul(delta_c, delta_c);
        delta = complex_mul(A, delta_c) + complex_mul(B, dc_sq);
    } else {
        delta = float2(0.0);
        skipIter = 0;
    }

    uint iteration = skipIter;
    const uint maxIter = min(params.maxIterations, params.orbitLength);

    for (uint i = skipIter; i < maxIter; i++) {
        float2 z_ref = referenceOrbit[i];
        float2 z_full = z_ref + delta;

        float mag_sq = dot(z_full, z_full);
        if (mag_sq > 65536.0f) {
            break;
        }

        float delta_mag_sq = dot(delta, delta);
        float z_ref_mag_sq = dot(z_ref, z_ref);
        if (delta_mag_sq > kGlitchThreshold * z_ref_mag_sq && z_ref_mag_sq > 1e-20f) {
            glitchBuffer[pixelIdx] = i;
            output.write(half4(1.0, 0.0, 1.0, 1.0), gid);
            return;
        }

        float2 two_z_ref = 2.0f * z_ref;
        float2 term1 = complex_mul(two_z_ref, delta);
        float2 term2 = complex_mul(delta, delta);
        delta = term1 + term2 + delta_c;

        iteration++;
    }

    glitchBuffer[pixelIdx] = 0;

    if (iteration >= maxIter) {
        output.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    float2 z_ref_final = (iteration < params.orbitLength) ? referenceOrbit[iteration] : float2(0.0);
    float2 z_full_final = z_ref_final + delta;
    float final_mag_sq = dot(z_full_final, z_full_final);

    float iteration_f = float(iteration);
    float log_zn = fast::log(final_mag_sq) * 0.5f;
    float nu = fast::log(log_zn * kInvLog2Perturb) * kInvLog2Perturb;
    iteration_f = iteration_f + 1.0f - nu;

    half normIter = half(iteration_f / float(params.maxIterations));

    half4 color;
    if (params.colorMode == 0u) {
        half hue = half(pow(float(normIter), 0.3333f));
        half3 rgb = hsv_to_rgb_perturb(hue, half(1.0), half(1.0));
        color = half4(rgb, 1.0);
    } else {
        half u_tex = clamp(normIter, half(0.0), half(0.9999));
        color = paletteTex.sample(paletteSampler, float2(u_tex, 0.5));
        color.a = half(1.0);
    }

    output.write(color, gid);
}

// MARK: - Multi-Reference Rendering

struct MultiReferenceParams {
    float4 viewCenterHiLo;      // (hi_x, lo_x, hi_y, lo_y)
    float2 scaleHiLo;           // (hi, lo)
    float2 scaleAspectHiLo;     // (hi, lo)
    uint width;
    uint height;
    uint maxIterations;
    uint regionCount;
    uint colorMode;
    uint padding1;
    uint padding2;
    uint padding3;
};

// Region metadata: [offset, length, skipIterations, padding]
struct RegionInfo {
    uint orbitOffset;
    uint orbitLength;
    uint skipIterations;
    uint padding;
};

// Kernel using multiple reference points based on screen region.
// Each pixel selects its reference based on which region it falls into.
kernel void mandelbrotMultiReferenceKernel(
    texture2d<half, access::write> output [[texture(0)]],
    texture2d<half> paletteTex [[texture(1)]],
    constant MultiReferenceParams &params [[buffer(0)]],
    constant float2 *allOrbits [[buffer(1)]],
    constant float4 *regionBounds [[buffer(2)]],      // [minX, minY, maxX, maxY] per region
    constant float4 *regionCenters [[buffer(3)]],     // [real_hi, real_lo, imag_hi, imag_lo] per region
    constant RegionInfo *regionInfo [[buffer(4)]],
    sampler paletteSampler [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    // Normalized pixel coordinates
    float2 normCoord = float2(gid) / float2(params.width, params.height);

    // Find which region this pixel belongs to
    uint regionIdx = 0;
    for (uint i = 0; i < params.regionCount; i++) {
        float4 b = regionBounds[i];
        if (normCoord.x >= b.x && normCoord.x < b.z &&
            normCoord.y >= b.y && normCoord.y < b.w) {
            regionIdx = i;
            break;
        }
    }

    // Get region-specific data
    RegionInfo info = regionInfo[regionIdx];
    float4 refCenter = regionCenters[regionIdx];
    constant float2 *orbit = allOrbits + info.orbitOffset;
    uint orbitLength = info.orbitLength;

    // Compute pixel position in complex plane
    float offset_x = (normCoord.x - 0.5f) * params.scaleAspectHiLo.x;
    float offset_y = (normCoord.y - 0.5f) * params.scaleHiLo.x;

    // Compute delta_c = (viewCenter + offset) - referenceCenter for this region
    float delta_c_x = (params.viewCenterHiLo.x - refCenter.x)
                    + (params.viewCenterHiLo.y - refCenter.y)
                    + offset_x;
    float delta_c_y = (params.viewCenterHiLo.z - refCenter.z)
                    + (params.viewCenterHiLo.w - refCenter.w)
                    + offset_y;

    float2 delta_c = float2(delta_c_x, delta_c_y);
    float2 delta = float2(0.0);

    uint iteration = 0;
    const uint maxIter = min(params.maxIterations, orbitLength);

    for (uint i = 0; i < maxIter; i++) {
        float2 z_ref = orbit[i];
        float2 z_full = z_ref + delta;

        float mag_sq = dot(z_full, z_full);
        if (mag_sq > 65536.0f) {
            break;
        }

        float2 two_z_ref = 2.0f * z_ref;
        float2 term1 = complex_mul(two_z_ref, delta);
        float2 term2 = complex_mul(delta, delta);
        delta = term1 + term2 + delta_c;

        iteration++;
    }

    if (iteration >= maxIter) {
        output.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    float2 z_ref_final = (iteration < orbitLength) ? orbit[iteration] : float2(0.0);
    float2 z_full_final = z_ref_final + delta;
    float final_mag_sq = dot(z_full_final, z_full_final);

    float iteration_f = float(iteration);
    float log_zn = fast::log(final_mag_sq) * 0.5f;
    float nu = fast::log(log_zn * kInvLog2Perturb) * kInvLog2Perturb;
    iteration_f = iteration_f + 1.0f - nu;

    half normIter = half(iteration_f / float(params.maxIterations));

    half4 color;
    if (params.colorMode == 0u) {
        half hue = half(pow(float(normIter), 0.3333f));
        half3 rgb = hsv_to_rgb_perturb(hue, half(1.0), half(1.0));
        color = half4(rgb, 1.0);
    } else {
        half u_tex = clamp(normIter, half(0.0), half(0.9999));
        color = paletteTex.sample(paletteSampler, float2(u_tex, 0.5));
        color.a = half(1.0);
    }

    output.write(color, gid);
}
