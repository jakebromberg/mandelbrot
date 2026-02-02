# Mandelbrot

An interactive Mandelbrot set explorer for iOS, built with Metal compute shaders.

## Features

- **Real-time rendering** using Metal compute shaders
- **Pinch-to-zoom** with focal point preservation (zoom centers on your pinch location)
- **Pan gestures** to explore the fractal
- **Deep zoom support** via perturbation theory, enabling zooms beyond 10^14
- **Adaptive quality** - lower resolution during gestures for smooth interaction, full quality on release
- **Dynamic iteration scaling** - automatically increases iterations at deeper zoom levels
- **Two color modes**: HSV gradient and palette-based coloring

## Rendering Modes

### Standard Mode
Used at shallow zoom levels (scale > 10^-6). Computes the Mandelbrot iteration directly using single-precision floats on the GPU. Includes optimizations:
- Cardioid and period-2 bulb early-out checks
- Periodicity detection to identify interior points quickly
- Smooth iteration coloring with continuous potential

### Perturbation Mode
Automatically activates at deep zoom levels (scale < 10^-6) where single-precision floats lose accuracy. Uses perturbation theory:

1. Computes a **reference orbit** at the screen center using `Double` precision on the CPU
2. For each pixel, computes only the **perturbation delta** from the reference using the recurrence:
   ```
   δₙ₊₁ = 2·zₙ·δₙ + δₙ² + δc
   ```
3. Since deltas remain small, `Float` precision suffices for GPU computation

This extends the zoom range from ~10^7 (float limit) to ~10^14 (double limit).

## UI Indicators

The top bar displays:
- **Zoom depth**: Shows magnification as "10^X" at deep zooms or "Nx" at shallow zooms
- **Rendering mode**: Blue dot for Standard, green dot for Perturbation
- **Color mode picker**: Toggle between HSV and Palette coloring

## Interesting Locations to Explore

| Location | Coordinates | Suggested Zoom |
|----------|-------------|----------------|
| Seahorse Valley | (-0.743643887037151, 0.131825904205330) | 10^-10 |
| Elephant Valley | (0.250006, 0.0) | 10^-8 |
| Main cardioid cusp | (-0.75, 0.0) | 10^-3 |
| Period-3 bulb | (-0.125, 0.75) | 10^-2 |

## Technical Details

### Dependencies
- [swift-numerics](https://github.com/apple/swift-numerics) - Provides `Complex<Double>` for reference orbit computation

### Architecture
- `MandelbrotView.swift` - SwiftUI/UIKit bridge with gesture handling
- `Renderer.swift` - Metal rendering pipeline with dual-mode support
- `ReferenceOrbit.swift` - CPU-side reference orbit computation
- `MandelbrotKernel.metal` - Standard GPU shader
- `MandelbrotPerturbation.metal` - Perturbation theory GPU shader

### Requirements
- iOS 18.0+
- Device with Metal support

## Building

Open `Mandelbrot.xcodeproj` in Xcode and build for an iOS device or simulator.

## Background

This project was created to learn shader programming in Metal, and also made for my kids to explore the beauty of fractals.
