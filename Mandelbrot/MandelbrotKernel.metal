#include <metal_stdlib>
using namespace metal;

struct MandelbrotParams {
    float scale;         // The height (in complex plane units) of the view.
    float2 center;       // The center of the view in the complex plane.
    uint width;          // The width of the output image (in pixels).
    uint height;         // The height of the output image (in pixels).
    uint maxIterations;  // Maximum iterations for the escape time algorithm.
};

// Converts HSV values (each in the range [0, 1]) to an RGB color.
#include <metal_stdlib>
using namespace metal;

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

inline float inv_log2() { return 1.442695f; } // 1 / log(2)

kernel void mandelbrotKernel(
    texture2d<half, access::write> output [[texture(0)]],
    constant MandelbrotParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    // Ensure we're within bounds.
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    
    // Compute aspect ratio.
    float aspect = float(params.width) / float(params.height);
    
    // Normalize pixel coordinate to [0, 1] and map it to the complex plane.
    float2 coord = float2(gid) / float2(params.width, params.height);
    float x0 = (coord.x - 0.5) * params.scale * aspect + params.center.x;
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
    const uint scaledMaxIterations = params.maxIterations;
    
    
    // Escape time algorithm.
    while ((x_2 + y_2 <= 4.0) && (iteration < scaledMaxIterations)) {
        float xtemp = x_2 - y_2 + x0;
        y = 2.0 * x * y + y0;
        x = xtemp;
        x_2 = x*x;
        y_2 = y*y;
        iteration++;
    }
    
    // If the point is in the Mandelbrot set, color it black.
    if (iteration == scaledMaxIterations) {
        output.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    
    // Smooth the iteration count for better coloring.
    float iteration_f = float(iteration);
    float log_zn = fast::log(x_2 + y_2) * 0.5f;
    float nu     = fast::log(log_zn * inv_log2()) * inv_log2();

    iteration_f = iteration_f + 1.0 - nu;
    
    // Normalize the iteration count and use it as the hue.
    // (You can adjust the multiplier on iteration_f to change the color cycle.)
    half normIter = half(iteration_f / float(scaledMaxIterations));
    half hue = half(pow(float(normIter), 0.3333f));
    half saturation = 1.0;
    half value = 1.0;
    
    // Convert the HSV color to RGB.
    half3 rgb = hsv_to_rgb(hue, saturation, value);
    half4 color = half4(rgb, 1.0);
    
    // Write the color to the output texture.
    output.write(color, gid);
}
