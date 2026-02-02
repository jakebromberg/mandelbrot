//
//  FPSTracker.swift
//  Mandelbrot
//
//  Tracks and exposes FPS metrics for SwiftUI observation.
//

import Foundation
import Observation

/// Tracks frame rate metrics for the Mandelbrot renderer.
/// Uses @Observable for automatic SwiftUI updates.
@Observable
class FPSTracker {
    /// Frames per second of the overall app (time between draw calls)
    var appFPS: Double = 0

    /// Frames per second based on GPU execution time
    var gpuFPS: Double = 0

    // Internal accumulators for averaging
    private var lastDrawTime: CFAbsoluteTime = 0
    private var appFPSAccumulator: Double = 0
    private var gpuFPSAccumulator: Double = 0
    private var frameCount: Int = 0

    /// Call at the start of each draw call to track app FPS.
    /// Returns the start time for GPU timing.
    func frameStarted() -> CFAbsoluteTime {
        let currentTime = CFAbsoluteTimeGetCurrent()
        if lastDrawTime > 0 {
            let frameDuration = currentTime - lastDrawTime
            if frameDuration > 0 {
                appFPSAccumulator += 1.0 / frameDuration
                frameCount += 1
            }
        }
        lastDrawTime = currentTime
        return currentTime
    }

    /// Call when GPU work completes to track GPU FPS.
    /// - Parameter startTime: The time returned from `frameStarted()`
    func frameCompleted(startTime: CFAbsoluteTime) {
        let gpuEndTime = CFAbsoluteTimeGetCurrent()
        let gpuDuration = gpuEndTime - startTime
        if gpuDuration > 0 {
            gpuFPSAccumulator += 1.0 / gpuDuration
        }

        // Update published FPS every 10 frames for smoother readings
        if frameCount >= 10 {
            appFPS = appFPSAccumulator / Double(frameCount)
            gpuFPS = gpuFPSAccumulator / Double(frameCount)
            appFPSAccumulator = 0
            gpuFPSAccumulator = 0
            frameCount = 0
        }
    }
}
