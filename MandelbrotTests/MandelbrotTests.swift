//
//  MandelbrotTests.swift
//  MandelbrotTests
//
//  Created by Jake Bromberg on 2/7/25.
//

import Testing
import Foundation
import Numerics
@testable import Mandelbrot

@Suite("Mandelbrot Renderer Tests")
struct MandelbrotTests {

    // MARK: - Performance Tests

    @Test("setView low quality performance - gesture hot path")
    func setViewLowQualityPerformance() throws {
        let renderer = try #require(Renderer())

        let start = CFAbsoluteTimeGetCurrent()
        for i in 0..<100 {
            let scale = 2.0 / Double(i + 1)
            let center = Complex(-0.75 + Double(i) * 0.001, 0.0)
            renderer.setView(center: center, scale: scale, lowQuality: true)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // 100 calls should complete in < 100ms (1ms each)
        #expect(elapsed < 0.1, "100 setView calls took \(elapsed * 1000)ms, expected < 100ms")
    }

    @Test("setView full quality performance - gesture end")
    func setViewFullQualityPerformance() throws {
        let renderer = try #require(Renderer())

        let start = CFAbsoluteTimeGetCurrent()
        renderer.setView(center: Complex(-0.75, 0.1), scale: 1e-8, lowQuality: false)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Full quality update should complete in < 500ms
        #expect(elapsed < 0.5, "Full quality setView took \(elapsed * 1000)ms, expected < 500ms")
    }

    @Test("Reference orbit computation performance")
    func referenceOrbitComputationPerformance() throws {
        let start = CFAbsoluteTimeGetCurrent()
        let orbit = ReferenceOrbit(center: Complex(-0.75, 0.1))
        orbit.computeOrbit(maxIterations: 1000)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        #expect(elapsed < 0.1, "Orbit computation took \(elapsed * 1000)ms, expected < 100ms")
    }

    @Test("Reference orbit with series approximation performance")
    func referenceOrbitWithSeriesPerformance() throws {
        let start = CFAbsoluteTimeGetCurrent()
        let orbit = ReferenceOrbit(center: Complex(-0.75, 0.1))
        orbit.computeOrbitWithSeries(maxIterations: 1000, screenDiagonalSquared: 1e-12)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        #expect(elapsed < 0.1, "Orbit with series took \(elapsed * 1000)ms, expected < 100ms")
    }

    @Test("Double-double precision orbit performance")
    func doubleDoubleOrbitPerformance() throws {
        let start = CFAbsoluteTimeGetCurrent()
        let orbit = ReferenceOrbitDD(center: Complex(-0.75, 0.1))
        orbit.computeOrbit(maxIterations: 1000)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        #expect(elapsed < 0.2, "DD orbit computation took \(elapsed * 1000)ms, expected < 200ms")
    }

    @Test("GPU buffer packing performance")
    func orbitPackingPerformance() throws {
        let orbit = ReferenceOrbit(center: Complex(-0.75, 0.1))
        orbit.computeOrbit(maxIterations: 5000)

        let start = CFAbsoluteTimeGetCurrent()
        _ = orbit.packForGPU()
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        #expect(elapsed < 0.01, "Orbit packing took \(elapsed * 1000)ms, expected < 10ms")
    }

    // MARK: - Diagnostic Tests

    @Test("Gesture frame timing diagnostic")
    func gestureFrameTimingDiagnostic() throws {
        let renderer = try #require(Renderer())

        var frameTimes: [Double] = []
        let iterations = 50

        for i in 0..<iterations {
            let t = Double(i) / Double(iterations - 1)
            let scale = 2.0 * Foundation.pow(1e-6 / 2.0, t)
            let center = Complex(-0.75 + t * 0.01, t * 0.01)

            let start = CFAbsoluteTimeGetCurrent()
            renderer.setView(center: center, scale: scale, lowQuality: true)
            let elapsed = Double(CFAbsoluteTimeGetCurrent() - start) * 1000.0

            frameTimes.append(elapsed)
        }

        let avg = frameTimes.reduce(0, +) / Double(frameTimes.count)
        let maxTime = frameTimes.max() ?? 0

        // For 60fps, each frame should be < 16ms
        #expect(avg < 16, "Average frame time \(avg)ms exceeds 16ms target")
        #expect(maxTime < 33, "Max frame time \(maxTime)ms exceeds 33ms (30fps floor)")
    }

    // MARK: - Correctness Tests

    @Test("Renderer initialization")
    func rendererInitialization() throws {
        let renderer = try #require(Renderer())

        #expect(renderer.scale == 2.0)
        #expect(renderer.center.real == -0.75)
        #expect(renderer.center.imaginary == 0.0)
        #expect(renderer.renderingMode == .standard)
    }

    @Test("Rendering mode transitions based on scale")
    func renderingModeTransition() throws {
        let renderer = try #require(Renderer())

        // At scale 2.0, should be standard mode
        renderer.setView(center: Complex(-0.75, 0.0), scale: 2.0, lowQuality: false)
        #expect(renderer.renderingMode == .standard)

        // At scale 1e-7 (below perturbation threshold), should switch to perturbation
        renderer.setView(center: Complex(-0.75, 0.0), scale: 1e-7, lowQuality: false)
        #expect(renderer.renderingMode == .perturbation)
    }

    @Test("Precision level transitions based on zoom depth")
    func precisionLevelTransition() throws {
        let renderer = try #require(Renderer())

        // At moderate zoom, should use double precision
        renderer.setView(center: Complex(-0.75, 0.0), scale: 1e-10, lowQuality: false)
        #expect(renderer.precisionLevel == .double)

        // At ultra-deep zoom, should switch to double-double
        renderer.setView(center: Complex(-0.75, 0.0), scale: 1e-15, lowQuality: false)
        #expect(renderer.precisionLevel == .doubleDouble)
    }

    @Test("Reference orbit escapes for point outside set")
    func referenceOrbitEscape() throws {
        let orbit = ReferenceOrbit(center: Complex(2.0, 0.0))
        let count = orbit.computeOrbit(maxIterations: 1000)

        #expect(count < 10, "Point at c=2 should escape quickly, took \(count) iterations")
        #expect(orbit.didEscape)
    }

    @Test("Reference orbit stays bounded for point in set")
    func referenceOrbitInSet() throws {
        let orbit = ReferenceOrbit(center: Complex(-0.5, 0.0))
        let count = orbit.computeOrbit(maxIterations: 1000)

        #expect(count == 1000, "Point at c=-0.5 should not escape")
        #expect(!orbit.didEscape)
    }

    @Test("DoubleDouble arithmetic preserves precision")
    func doubleDoubleArithmetic() throws {
        // Test that DoubleDouble preserves precision in the low part
        let a = DoubleDouble(1.0)
        let b = DoubleDouble(hi: 0, lo: 1e-20)
        let sum = a + b

        #expect(sum.hi == 1.0)
        #expect(sum.lo > 0, "DoubleDouble should preserve small additions in lo part")

        // Test multiplication precision
        let c = DoubleDouble(1.0 + 1e-15)
        let d = DoubleDouble(1.0 + 1e-15)
        let product = c * d
        #expect(product.hi > 1.0, "(1 + ε)² should be > 1")
    }

    @Test("Display properties only update on gesture end")
    func displayPropertiesThrottling() throws {
        let renderer = try #require(Renderer())

        let initialDisplayScale = renderer.displayScale

        // Low quality update should NOT change display properties
        renderer.setView(center: Complex(0, 0), scale: 0.5, lowQuality: true)
        #expect(renderer.displayScale == initialDisplayScale, "displayScale should not change during gesture")

        // Full quality update SHOULD change display properties
        renderer.setView(center: Complex(0, 0), scale: 0.5, lowQuality: false)
        #expect(renderer.displayScale == 0.5, "displayScale should update on gesture end")
    }
}
