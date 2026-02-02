//
//  ContentView.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/7/25.
//

import SwiftUI
import Numerics

struct ContentView: View {
    @State private var colorMode: UInt32 = 0 // 0=HSV, 1=Palette
    @State private var currentScale: Double = 2.0
    @State private var currentCenter: Complex<Double> = Complex(-0.75, 0.0)
    @State private var renderingMode: RenderingMode = .standard
    @State private var precisionLevel: PrecisionLevel = .double
    @State private var fpsTracker = FPSTracker()

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .top) {
                MandelbrotView(
                    size: geometry.size,
                    scale: 2.0,
                    center: Complex(-0.75, 0.0),
                    colorMode: colorMode,
                    onStateChange: { scale, center, mode, precision in
                        currentScale = scale
                        currentCenter = center
                        renderingMode = mode
                        precisionLevel = precision
                    },
                    fpsTracker: fpsTracker
                )
                .edgesIgnoringSafeArea(.all)

                HStack(spacing: 12) {
                    // Zoom depth indicator
                    ZoomIndicator(scale: currentScale)

                    // Mode indicator
                    ModeIndicator(mode: renderingMode)

                    // Precision indicator (only show at deep zooms)
                    if precisionLevel == .doubleDouble {
                        PrecisionIndicator(precision: precisionLevel)
                    }

                    // FPS indicator
                    FPSIndicator(tracker: fpsTracker)

                    Spacer()

                    // Color mode picker
                    Picker("Color", selection: $colorMode) {
                        Text("HSV").tag(UInt32(0))
                        Text("Palette").tag(UInt32(1))
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 140)
                }
                .padding(12)
                .background(.ultraThinMaterial, in: Capsule())
                .padding([.top, .horizontal], 16)
            }
        }
    }
}

struct ZoomIndicator: View {
    let scale: Double

    private var zoomExponent: Double {
        -log10(scale)
    }

    private var formattedZoom: String {
        if zoomExponent < 1 {
            return String(format: "%.1fx", 1.0 / scale)
        } else {
            return String(format: "10^%.1f", zoomExponent)
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .medium))
            Text(formattedZoom)
                .font(.system(size: 13, weight: .medium, design: .monospaced))
        }
        .foregroundColor(.primary.opacity(0.8))
    }
}

struct ModeIndicator: View {
    let mode: RenderingMode

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(mode == .perturbation ? Color.green : Color.blue)
                .frame(width: 8, height: 8)
            Text(mode == .perturbation ? "Perturb" : "Standard")
                .font(.system(size: 12, weight: .medium))
        }
        .foregroundColor(.primary.opacity(0.8))
    }
}

struct PrecisionIndicator: View {
    let precision: PrecisionLevel

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "number.circle.fill")
                .font(.system(size: 10))
            Text(precision == .doubleDouble ? "DD" : "D")
                .font(.system(size: 12, weight: .medium, design: .monospaced))
        }
        .foregroundColor(.orange.opacity(0.9))
    }
}

struct FPSIndicator: View {
    let tracker: FPSTracker

    var body: some View {
        HStack(spacing: 8) {
            // App FPS
            HStack(spacing: 2) {
                Text("App:")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.primary.opacity(0.6))
                Text(String(format: "%.0f", tracker.appFPS))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
            }
            // GPU FPS
            HStack(spacing: 2) {
                Text("GPU:")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.primary.opacity(0.6))
                Text(String(format: "%.0f", tracker.gpuFPS))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
            }
        }
        .foregroundColor(.primary.opacity(0.8))
    }
}

#Preview {
    ContentView()
}

