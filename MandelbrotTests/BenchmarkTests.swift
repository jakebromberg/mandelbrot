//
//  BenchmarkTests.swift
//  MandelbrotTests
//
//  Automated performance benchmarks that fail on regressions.
//

import Testing
import Foundation
import Numerics
@testable import Mandelbrot

@Suite("Performance Benchmarks")
struct BenchmarkTests {

    // MARK: - Benchmark Infrastructure

    func createBenchmark() throws -> RenderBenchmark {
        let renderer = try #require(Renderer())
        let benchmark = RenderBenchmark(renderer: renderer)
        benchmark.setUp(size: CGSize(width: 512, height: 512))  // Smaller for faster CI
        return benchmark
    }

    // MARK: - Individual Scenario Tests

    @Test("Zoom in benchmark meets 30fps target")
    func zoomInBenchmark() throws {
        let benchmark = try createBenchmark()
        let result = benchmark.run(.zoomIn(frames: 50))

        print(result)
        #expect(result.averageFPS >= 30, "Zoom benchmark should achieve 30fps, got \(result.averageFPS)")
        #expect(result.droppedFrames30 == 0, "Should have no frames > 33ms, had \(result.droppedFrames30)")
    }

    @Test("Pan benchmark meets 30fps target")
    func panBenchmark() throws {
        let benchmark = try createBenchmark()
        let result = benchmark.run(.pan(frames: 50))

        print(result)
        #expect(result.averageFPS >= 30, "Pan benchmark should achieve 30fps, got \(result.averageFPS)")
    }

    @Test("Zoom and pan benchmark meets 30fps target")
    func zoomAndPanBenchmark() throws {
        let benchmark = try createBenchmark()
        let result = benchmark.run(.zoomAndPan(frames: 50))

        print(result)
        #expect(result.averageFPS >= 30, "Zoom+pan benchmark should achieve 30fps, got \(result.averageFPS)")
    }

    @Test("Deep zoom stress test completes without crashes")
    func deepZoomStressTest() throws {
        let benchmark = try createBenchmark()
        let result = benchmark.run(.deepZoomStress(frames: 30))

        print(result)
        // Deep zoom is expected to be slower, just verify it doesn't crash
        #expect(result.frameCount == 30, "All frames should complete")
        #expect(result.averageFPS >= 10, "Deep zoom should achieve at least 10fps")
    }

    // MARK: - Regression Tests

    @Test("Low quality setView is fast enough for 60fps")
    func setViewLowQualitySpeed() throws {
        let renderer = try #require(Renderer())

        // Measure time for 60 setView calls (simulating 1 second of gestures)
        let start = CFAbsoluteTimeGetCurrent()
        for i in 0..<60 {
            let t = Double(i) / 59.0
            renderer.setView(
                center: Complex(-0.75 + t * 0.1, t * 0.1),
                scale: 2.0 * Foundation.pow(1e-6, t),
                lowQuality: true
            )
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // 60 calls should take < 100ms (leaves plenty of headroom for GPU work)
        #expect(elapsed < 0.1, "60 setView calls took \(elapsed * 1000)ms, should be < 100ms")
    }

    @Test("Frame time consistency - no outliers")
    func frameTimeConsistency() throws {
        let benchmark = try createBenchmark()
        let result = benchmark.run(.zoomIn(frames: 100))

        // P95 should not be more than 3x the average
        let ratio = result.p95FrameTime / result.averageFrameTime
        #expect(ratio < 3.0, "P95/avg ratio is \(ratio), should be < 3 for consistent frame times")
    }

    // MARK: - Specific Optimization Tests

    @Test("Display properties don't update during gestures")
    func displayPropertyThrottling() throws {
        let renderer = try #require(Renderer())
        let initialScale = renderer.displayScale

        // Simulate 100 gesture frames
        for i in 0..<100 {
            renderer.setView(
                center: Complex(0, 0),
                scale: Double(i + 1) * 0.01,
                lowQuality: true
            )
            // displayScale should NOT change during gesture
            #expect(renderer.displayScale == initialScale)
        }

        // Final update should change displayScale
        renderer.setView(center: Complex(0, 0), scale: 0.5, lowQuality: false)
        #expect(renderer.displayScale == 0.5)
    }

    @Test("Expensive operations are skipped in low quality mode")
    func lowQualitySkipsExpensiveOps() throws {
        let renderer = try #require(Renderer())

        // Set up at a scale that would trigger perturbation mode
        renderer.setView(center: Complex(-0.75, 0), scale: 1e-8, lowQuality: false)
        #expect(renderer.renderingMode == .perturbation)

        // Time a low-quality update (should be fast)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10 {
            renderer.setView(center: Complex(-0.75, 0.001), scale: 1e-8, lowQuality: true)
        }
        let lowQualityTime = CFAbsoluteTimeGetCurrent() - start

        // Time a full-quality update (may recompute orbits)
        let start2 = CFAbsoluteTimeGetCurrent()
        renderer.setView(center: Complex(-0.75, 0.1), scale: 1e-8, lowQuality: false)
        let fullQualityTime = CFAbsoluteTimeGetCurrent() - start2

        // Low quality batch should be faster than single full quality
        // (This verifies expensive ops are being skipped)
        print("Low quality (10x): \(lowQualityTime * 1000)ms")
        print("Full quality (1x): \(fullQualityTime * 1000)ms")

        // 10 low-quality updates should take less than 2x one full-quality update
        #expect(lowQualityTime < fullQualityTime * 2,
               "Low quality should be significantly faster than full quality")
    }
}

// MARK: - Benchmark Runner (for manual execution)

@Suite("Benchmark Runner", .disabled("Run manually with --filter"))
struct BenchmarkRunner {

    @Test("Full benchmark suite with report")
    func fullBenchmarkSuite() throws {
        let renderer = try #require(Renderer())
        let benchmark = RenderBenchmark(renderer: renderer)
        benchmark.setUp(size: CGSize(width: 1024, height: 1024))

        let passed = benchmark.runAndReport()
        #expect(passed, "Some benchmarks failed to meet performance targets")
    }
}
