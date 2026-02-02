//
//  ContentView.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/7/25.
//

import SwiftUI
import Numerics

struct ContentView: View {
    @State private var renderer: Renderer? = Renderer()

    var body: some View {
        GeometryReader { geometry in
            if let renderer {
                ZStack(alignment: .top) {
                    MandelbrotView(renderer: renderer, size: geometry.size)
                        .edgesIgnoringSafeArea(.all)

                    HStack(spacing: 12) {
                        ZoomIndicator(scale: renderer.scale)
                        ModeIndicator(mode: renderer.renderingMode)

                        if renderer.precisionLevel == .doubleDouble {
                            PrecisionIndicator(precision: renderer.precisionLevel)
                        }

                        FPSIndicator(appFPS: renderer.appFPS, gpuFPS: renderer.gpuFPS)

                        Spacer()

                        Picker("Color", selection: Binding(
                            get: { renderer.colorMode },
                            set: { renderer.colorMode = $0 }
                        )) {
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
            } else {
                Text("Metal is not supported on this device")
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
    let appFPS: Double
    let gpuFPS: Double

    var body: some View {
        HStack(spacing: 8) {
            HStack(spacing: 2) {
                Text("App:")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.primary.opacity(0.6))
                Text(String(format: "%.0f", appFPS))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
            }
            HStack(spacing: 2) {
                Text("GPU:")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.primary.opacity(0.6))
                Text(String(format: "%.0f", gpuFPS))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
            }
        }
        .foregroundColor(.primary.opacity(0.8))
    }
}

#Preview {
    ContentView()
}
