//
//  MandelbrotTests.swift
//  MandelbrotTests
//
//  Created by Jake Bromberg on 2/7/25.
//

import XCTest
import Numerics
@testable import Mandelbrot

final class MandelbrotTests: XCTestCase {

    var renderer: Renderer!

    override func setUpWithError() throws {
        renderer = Renderer()
        XCTAssertNotNil(renderer, "Renderer should initialize")
    }

    override func tearDownWithError() throws {
        renderer = nil
    }

    // MARK: - Performance Tests

    /// Tests the performance of setView() during gestures (lowQuality: true)
    /// This is the hot path during pinch/pan - should be < 1ms
    func testSetViewLowQualityPerformance() throws {
        let renderer = try XCTUnwrap(self.renderer)

        // Simulate a zoom gesture with many updates
        measure {
            for i in 0..<100 {
                let scale = 2.0 / Double(i + 1)
                let center = Complex(-0.75 + Double(i) * 0.001, 0.0)
                renderer.setView(center: center, scale: scale, lowQuality: true)
            }
        }
    }

    /// Tests the performance of setView() with full quality updates
    /// This runs on gesture end - can be slower but should complete reasonably
    func testSetViewFullQualityPerformance() throws {
        let renderer = try XCTUnwrap(self.renderer)

        measure {
            // Single full-quality update at a deep zoom level
            renderer.setView(center: Complex(-0.75, 0.1), scale: 1e-8, lowQuality: false)
        }
    }

    /// Tests reference orbit computation at various zoom depths
    func testReferenceOrbitComputationPerformance() throws {
        measure {
            let orbit = ReferenceOrbit(center: Complex(-0.75, 0.1))
            orbit.computeOrbit(maxIterations: 1000)
        }
    }

    /// Tests reference orbit computation with series approximation
    func testReferenceOrbitWithSeriesPerformance() throws {
        measure {
            let orbit = ReferenceOrbit(center: Complex(-0.75, 0.1))
            orbit.computeOrbitWithSeries(maxIterations: 1000, screenDiagonalSquared: 1e-12)
        }
    }

    /// Tests double-double precision orbit computation
    func testDoubleDoubleOrbitPerformance() throws {
        measure {
            let orbit = ReferenceOrbitDD(center: Complex(-0.75, 0.1))
            orbit.computeOrbit(maxIterations: 1000)
        }
    }

    /// Tests GPU buffer packing performance
    func testOrbitPackingPerformance() throws {
        let orbit = ReferenceOrbit(center: Complex(-0.75, 0.1))
        orbit.computeOrbit(maxIterations: 5000)

        measure {
            _ = orbit.packForGPU()
        }
    }

    // MARK: - Diagnostic Tests

    /// Simulates a zoom gesture and measures individual frame times
    /// Helps identify which frames cause stuttering
    func testGestureFrameTimingDiagnostic() throws {
        let renderer = try XCTUnwrap(self.renderer)

        var frameTimes: [Double] = []
        let iterations = 50

        // Simulate zooming from scale 2.0 to 1e-6
        for i in 0..<iterations {
            let t = Double(i) / Double(iterations - 1)
            let scale = 2.0 * pow(1e-6 / 2.0, t)  // Exponential zoom
            let center = Complex(-0.75 + t * 0.01, t * 0.01)

            let start = CFAbsoluteTimeGetCurrent()
            renderer.setView(center: center, scale: scale, lowQuality: true)
            let elapsed = Double(CFAbsoluteTimeGetCurrent() - start) * 1000.0  // ms

            frameTimes.append(elapsed)
        }

        let avg = frameTimes.reduce(0, +) / Double(frameTimes.count)
        let max = frameTimes.max() ?? 0
        let min = frameTimes.min() ?? 0

        print("Frame timing (lowQuality: true):")
        print("  Average: \(String(format: "%.2f", avg)) ms")
        print("  Min: \(String(format: "%.2f", min)) ms")
        print("  Max: \(String(format: "%.2f", max)) ms")
        print("  Frames > 16ms (60fps): \(frameTimes.filter { $0 > 16 }.count)")
        print("  Frames > 33ms (30fps): \(frameTimes.filter { $0 > 33 }.count)")

        // For 60fps, each frame should be < 16ms
        XCTAssertLessThan(avg, 16, "Average frame time should support 60fps")
    }

    /// Tests if @Observable property access has significant overhead
    func testObservablePropertyAccessPerformance() throws {
        let renderer = try XCTUnwrap(self.renderer)

        measure {
            for _ in 0..<10000 {
                _ = renderer.scale
                _ = renderer.center
                _ = renderer.renderingMode
                _ = renderer.precisionLevel
                _ = renderer.appFPS
                _ = renderer.gpuFPS
            }
        }
    }

    // MARK: - Correctness Tests

    func testRendererInitialization() throws {
        let renderer = try XCTUnwrap(self.renderer)
        XCTAssertEqual(renderer.scale, 2.0)
        XCTAssertEqual(renderer.center.real, -0.75)
        XCTAssertEqual(renderer.center.imaginary, 0.0)
        XCTAssertEqual(renderer.renderingMode, .standard)
    }

    func testRenderingModeTransition() throws {
        let renderer = try XCTUnwrap(self.renderer)

        // At scale 2.0, should be standard mode
        renderer.setView(center: Complex(-0.75, 0.0), scale: 2.0, lowQuality: false)
        XCTAssertEqual(renderer.renderingMode, .standard)

        // At scale 1e-7 (below perturbation threshold), should switch to perturbation
        renderer.setView(center: Complex(-0.75, 0.0), scale: 1e-7, lowQuality: false)
        XCTAssertEqual(renderer.renderingMode, .perturbation)
    }

    func testPrecisionLevelTransition() throws {
        let renderer = try XCTUnwrap(self.renderer)

        // At moderate zoom, should use double precision
        renderer.setView(center: Complex(-0.75, 0.0), scale: 1e-10, lowQuality: false)
        XCTAssertEqual(renderer.precisionLevel, .double)

        // At ultra-deep zoom, should switch to double-double
        renderer.setView(center: Complex(-0.75, 0.0), scale: 1e-15, lowQuality: false)
        XCTAssertEqual(renderer.precisionLevel, .doubleDouble)
    }

    func testReferenceOrbitEscape() throws {
        // Point outside the set - should escape quickly
        let orbit = ReferenceOrbit(center: Complex(2.0, 0.0))
        let count = orbit.computeOrbit(maxIterations: 1000)
        XCTAssertLessThan(count, 10, "Point at c=2 should escape quickly")
        XCTAssertTrue(orbit.didEscape)
    }

    func testReferenceOrbitInSet() throws {
        // Point deep in the set - should not escape
        let orbit = ReferenceOrbit(center: Complex(-0.5, 0.0))
        let count = orbit.computeOrbit(maxIterations: 1000)
        XCTAssertEqual(count, 1000, "Point at c=-0.5 should not escape")
        XCTAssertFalse(orbit.didEscape)
    }

    func testDoubleDoubleArithmetic() throws {
        // Test that DoubleDouble preserves precision in the low part
        let a = DoubleDouble(1.0)
        let b = DoubleDouble(hi: 0, lo: 1e-20)
        let sum = a + b

        // The hi part should be 1.0, and lo should contain the small value
        XCTAssertEqual(sum.hi, 1.0)
        XCTAssertGreaterThan(sum.lo, 0, "DoubleDouble should preserve small additions in lo part")

        // Test multiplication precision
        let c = DoubleDouble(1.0 + 1e-15)
        let d = DoubleDouble(1.0 + 1e-15)
        let product = c * d
        // (1 + ε)² = 1 + 2ε + ε² ≈ 1 + 2e-15
        XCTAssertGreaterThan(product.hi, 1.0)
    }
}
