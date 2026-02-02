#include <metal_stdlib>
using namespace metal;

struct MandelbrotParams {
    float scale;         // The height (in complex plane units) of the view.
    float scaleAspect;   // Precomputed scale * aspect ratio.
    float2 center;       // The center of the view in the complex plane.
    uint width;          // The width of the output image (in pixels).
    uint height;         // The height of the output image (in pixels).
    uint maxIterations;  // Maximum iterations for the escape time algorithm.
    uint colorMode;      // 0 = HSV, 1 = Palette
    uint options;        // bit 0: enable periodicity check
};

// Converts HSV values (each in the range [0, 1]) to an RGB color.

half3 hsv_to_rgb(half h, half s, half v)
{
    half c = v * s;  // Chroma.
    
    // Convert h to float and scale to [0,6] range.
    float h_prime_f = float(h) * 6.0f;
    
    // Use fmod on floats.
    float mod_val = fmod(h_prime_f, 2.0f);
    
    // Cast the result back to half and compute x.
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

constant float kInvLog2 = 1.442695f; // 1 / log(2)

kernel void mandelbrotKernel(
    texture2d<half, access::write> output [[texture(0)]],
    texture2d<half> paletteTex [[texture(1)]],
    constant MandelbrotParams &params [[buffer(0)]],
    sampler paletteSampler [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    // Ensure we're within bounds.
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    // Normalize pixel coordinate to [0, 1] and map it to the complex plane.
    float2 coord = float2(gid) / float2(params.width, params.height);
    float x0 = (coord.x - 0.5) * params.scaleAspect + params.center.x;
    float y0 = (coord.y - 0.5) * params.scale + params.center.y;
    
    // Cardioid check:
    float xshift = x0 - 0.25f;
    float r  = xshift * xshift + y0 * y0;
    if (r * (r + xshift) <= 0.25f * y0 * y0) {
        // The point is in the main cardioid. Color it black and return.
        output.write(half4(0, 0, 0, 1), gid);
        return;
    }

    // Period-2 bulb check:
    float xplus1 = x0 + 1.0f;
    if (xplus1 * xplus1 + y0 * y0 <= 1.0f / 16.0f) {
        // In the period-2 bulb
        output.write(half4(0, 0, 0, 1), gid);
        return;
    }
    
    // Initialize iteration variables.
    float x = 0.0;
    float y = 0.0;
    float x_2 = 0.0;
    float y_2 = 0.0;
    uint iteration = 0;
    const uint maxIterations = params.maxIterations;
    const bool enablePeriodicity = (params.options & 0x1u) != 0u;
    // For a cheap periodicity detection, compare to a saved z every 32 iterations.
    const float periodicityEpsilon = 1e-12f;
    float x_prev = 0.0f;
    float y_prev = 0.0f;
    
    
    // Escape time algorithm. Using 256^2 = 65536 as escape radius for smoother coloring.
    while ((x_2 + y_2 <= 65536.0) && (iteration < maxIterations)) {
        float xtemp = x_2 - y_2 + x0;
        y = 2.0 * x * y + y0;
        x = xtemp;
        x_2 = x*x;
        y_2 = y*y;
        iteration++;

        if (enablePeriodicity && ((iteration & 31u) == 0u)) {
            float dx = fabs(x - x_prev);
            float dy = fabs(y - y_prev);
            if (dx + dy < periodicityEpsilon) {
                // Consider the point interior (likely periodic orbit)
                iteration = maxIterations;
                break;
            }
            x_prev = x;
            y_prev = y;
        }
    }
    
    // If the point is in the Mandelbrot set, color it black.
    if (iteration == maxIterations) {
        output.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    
    // Smooth the iteration count for better coloring.
    float iteration_f = float(iteration);
    float log_zn = fast::log(x_2 + y_2) * 0.5f;
    float nu     = fast::log(log_zn * kInvLog2) * kInvLog2;

    iteration_f = iteration_f + 1.0 - nu;
    
    // Normalize the iteration count for color mapping.
    half normIter = half(iteration_f / float(maxIterations));

    half4 color;
    if (params.colorMode == 0u) {
        // HSV mode
        half hue = half(pow(float(normIter), 0.3333f));
        half saturation = 1.0;
        half value = 1.0;
        half3 rgb = hsv_to_rgb(hue, saturation, value);
        color = half4(rgb, 1.0);
    } else {
        // Palette LUT mode: sample a 1D palette implemented as a 2D texture with height=1.
        // Avoid exact 0/1 to stay inside texture coordinates.
        half u = clamp(normIter, half(0.0), half(0.9999));
        half v = half(0.5);
        color = paletteTex.sample(paletteSampler, float2(u, v));
        color.a = half(1.0);
    }

    // Write the color to the output texture.
    output.write(color, gid);
}
