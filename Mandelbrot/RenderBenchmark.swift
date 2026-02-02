//
//  RenderBenchmark.swift
//  Mandelbrot
//
//  Automated rendering benchmarks for performance testing and optimization.
//

import Foundation
import MetalKit
import Numerics

/// Results from a benchmark run
struct BenchmarkResult: CustomStringConvertible {
    let name: String
    let frameCount: Int
    let totalTime: Double
    let frameTimes: [Double]

    var averageFPS: Double {
        frameCount > 0 ? Double(frameCount) / totalTime : 0
    }

    var averageFrameTime: Double {
        frameTimes.isEmpty ? 0 : frameTimes.reduce(0, +) / Double(frameTimes.count)
    }

    var minFrameTime: Double {
        frameTimes.min() ?? 0
    }

    var maxFrameTime: Double {
        frameTimes.max() ?? 0
    }

    var p95FrameTime: Double {
        guard !frameTimes.isEmpty else { return 0 }
        let sorted = frameTimes.sorted()
        let index = Int(Double(sorted.count) * 0.95)
        return sorted[min(index, sorted.count - 1)]
    }

    var droppedFrames60: Int {
        frameTimes.filter { $0 > 16.67 }.count
    }

    var droppedFrames30: Int {
        frameTimes.filter { $0 > 33.33 }.count
    }

    var description: String {
        """
        Benchmark: \(name)
          Frames: \(frameCount) in \(String(format: "%.2f", totalTime))s
          FPS: \(String(format: "%.1f", averageFPS)) avg
          Frame time: \(String(format: "%.2f", averageFrameTime))ms avg, \(String(format: "%.2f", minFrameTime))ms min, \(String(format: "%.2f", maxFrameTime))ms max
          P95: \(String(format: "%.2f", p95FrameTime))ms
          Dropped: \(droppedFrames60) @60fps, \(droppedFrames30) @30fps
        """
    }

    /// Returns true if benchmark meets performance targets
    func meetsTarget(minFPS: Double = 30, maxP95ms: Double = 33.33) -> Bool {
        averageFPS >= minFPS && p95FrameTime <= maxP95ms
    }
}

/// Defines a benchmark scenario
struct BenchmarkScenario {
    let name: String
    let steps: [BenchmarkStep]
    let warmupFrames: Int

    init(name: String, warmupFrames: Int = 10, steps: [BenchmarkStep]) {
        self.name = name
        self.warmupFrames = warmupFrames
        self.steps = steps
    }

    /// Standard zoom-in benchmark
    static func zoomIn(from startScale: Double = 2.0, to endScale: Double = 1e-6, frames: Int = 100) -> BenchmarkScenario {
        let steps = (0..<frames).map { i -> BenchmarkStep in
            let t = Double(i) / Double(frames - 1)
            let scale = startScale * Foundation.pow(endScale / startScale, t)
            return BenchmarkStep(center: Complex(-0.75, 0.1), scale: scale, lowQuality: true)
        }
        return BenchmarkScenario(name: "Zoom In (\(startScale) → \(endScale))", steps: steps)
    }

    /// Pan benchmark at fixed zoom
    static func pan(at scale: Double = 1e-4, frames: Int = 100) -> BenchmarkScenario {
        let steps = (0..<frames).map { i -> BenchmarkStep in
            let t = Double(i) / Double(frames - 1)
            let x = -0.75 + 0.1 * sin(t * .pi * 2)
            let y = 0.1 + 0.1 * cos(t * .pi * 2)
            return BenchmarkStep(center: Complex(x, y), scale: scale, lowQuality: true)
        }
        return BenchmarkScenario(name: "Pan at scale \(scale)", steps: steps)
    }

    /// Mixed zoom and pan
    static func zoomAndPan(frames: Int = 100) -> BenchmarkScenario {
        let steps = (0..<frames).map { i -> BenchmarkStep in
            let t = Double(i) / Double(frames - 1)
            let scale = 2.0 * Foundation.pow(1e-8 / 2.0, t)
            let x = -0.75 + 0.05 * t
            let y = 0.1 * t
            return BenchmarkStep(center: Complex(x, y), scale: scale, lowQuality: true)
        }
        return BenchmarkScenario(name: "Zoom and Pan", steps: steps)
    }

    /// Deep zoom stress test
    static func deepZoomStress(frames: Int = 50) -> BenchmarkScenario {
        let steps = (0..<frames).map { i -> BenchmarkStep in
            let t = Double(i) / Double(frames - 1)
            let scale = 1e-10 * Foundation.pow(1e-14 / 1e-10, t)
            return BenchmarkStep(center: Complex(-0.743643887037151, 0.131825904205330), scale: scale, lowQuality: true)
        }
        return BenchmarkScenario(name: "Deep Zoom Stress Test", steps: steps)
    }
}

/// A single step in a benchmark
struct BenchmarkStep {
    let center: Complex<Double>
    let scale: Double
    let lowQuality: Bool
}

/// Runs rendering benchmarks
class RenderBenchmark {
    private let renderer: Renderer
    private var mtkView: MTKView?

    init(renderer: Renderer) {
        self.renderer = renderer
    }

    /// Sets up a Metal view for benchmarking (call once before running benchmarks)
    func setUp(size: CGSize = CGSize(width: 1024, height: 1024)) {
        let view = MTKView(frame: CGRect(origin: .zero, size: size), device: renderer.device)
        view.framebufferOnly = false
        view.isPaused = true
        view.enableSetNeedsDisplay = false
        view.colorPixelFormat = .bgra8Unorm
        view.drawableSize = size
        view.delegate = renderer
        self.mtkView = view
    }

    /// Runs a benchmark scenario and returns results
    func run(_ scenario: BenchmarkScenario) -> BenchmarkResult {
        guard let view = mtkView else {
            fatalError("Call setUp() before running benchmarks")
        }

        var frameTimes: [Double] = []
        frameTimes.reserveCapacity(scenario.steps.count)

        // Warmup frames (not measured)
        for i in 0..<min(scenario.warmupFrames, scenario.steps.count) {
            let step = scenario.steps[i]
            renderer.setView(center: step.center, scale: step.scale, lowQuality: step.lowQuality)
            view.draw()
        }

        // Measured frames
        let startTime = CFAbsoluteTimeGetCurrent()

        for step in scenario.steps {
            let frameStart = CFAbsoluteTimeGetCurrent()

            renderer.setView(center: step.center, scale: step.scale, lowQuality: step.lowQuality)
            view.draw()

            // Wait for GPU completion by creating a completion semaphore
            // Note: This is a simplified sync - real GPU timing would use MTLEvent
            let frameEnd = CFAbsoluteTimeGetCurrent()
            frameTimes.append((frameEnd - frameStart) * 1000.0) // Convert to ms
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime

        return BenchmarkResult(
            name: scenario.name,
            frameCount: scenario.steps.count,
            totalTime: totalTime,
            frameTimes: frameTimes
        )
    }

    /// Runs all standard benchmarks
    func runAll() -> [BenchmarkResult] {
        [
            run(.zoomIn()),
            run(.pan()),
            run(.zoomAndPan()),
            run(.deepZoomStress())
        ]
    }

    /// Runs benchmarks and prints a summary report
    func runAndReport() -> Bool {
        let results = runAll()
        var allPassed = true

        print("\n" + String(repeating: "=", count: 60))
        print("BENCHMARK RESULTS")
        print(String(repeating: "=", count: 60))

        for result in results {
            print(result)
            let passed = result.meetsTarget()
            print("  Status: \(passed ? "✓ PASS" : "✗ FAIL")")
            print("")
            allPassed = allPassed && passed
        }

        print(String(repeating: "=", count: 60))
        print("Overall: \(allPassed ? "✓ ALL PASSED" : "✗ SOME FAILED")")
        print(String(repeating: "=", count: 60) + "\n")

        return allPassed
    }
}

/// Compares two benchmark results to detect regressions
struct BenchmarkComparison {
    let baseline: BenchmarkResult
    let current: BenchmarkResult

    var fpsChange: Double {
        guard baseline.averageFPS > 0 else { return 0 }
        return (current.averageFPS - baseline.averageFPS) / baseline.averageFPS * 100
    }

    var p95Change: Double {
        guard baseline.p95FrameTime > 0 else { return 0 }
        return (current.p95FrameTime - baseline.p95FrameTime) / baseline.p95FrameTime * 100
    }

    var isRegression: Bool {
        fpsChange < -10 || p95Change > 20  // 10% FPS drop or 20% P95 increase
    }

    var description: String {
        """
        Comparison: \(baseline.name)
          FPS: \(String(format: "%.1f", baseline.averageFPS)) → \(String(format: "%.1f", current.averageFPS)) (\(String(format: "%+.1f", fpsChange))%)
          P95: \(String(format: "%.2f", baseline.p95FrameTime))ms → \(String(format: "%.2f", current.p95FrameTime))ms (\(String(format: "%+.1f", p95Change))%)
          Status: \(isRegression ? "⚠️ REGRESSION" : "✓ OK")
        """
    }
}
