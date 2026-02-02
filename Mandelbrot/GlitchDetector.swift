//
//  GlitchDetector.swift
//  Mandelbrot
//
//  Glitch detection and re-referencing for perturbation theory rendering.
//  Detects numerical instability and selects new reference points for affected regions.
//

import Foundation
import Metal
import simd

// Using Complex from Numerics module (imported via swift-numerics dependency)
import Numerics

// MARK: - Glitch Data Structures

/// Represents a detected glitch at a specific pixel.
struct GlitchedPixel {
    let x: Int
    let y: Int
    let iteration: UInt32
}

/// Represents a cluster of glitched pixels that share a common new reference.
struct GlitchCluster {
    var pixels: [GlitchedPixel]
    var centroid: (x: Double, y: Double)
    var newReference: Complex<Double>?
}

// MARK: - Glitch Detector

/// Analyzes glitch buffer output from GPU and selects new reference points.
class GlitchDetector {
    /// Minimum number of glitched pixels to warrant a new reference point.
    private let minClusterSize: Int = 16

    /// Maximum number of reference points to compute per frame.
    private let maxNewReferences: Int = 4

    /// Threshold for clustering glitched pixels (in normalized screen coordinates).
    private let clusterRadius: Double = 0.15

    /// Extracts glitched pixel locations from the GPU output buffer.
    /// - Parameters:
    ///   - buffer: The Metal buffer containing glitch data (iteration where glitch detected, or 0 if none)
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Array of glitched pixels with their locations and iteration counts
    func extractGlitchedPixels(from buffer: MTLBuffer, width: Int, height: Int) -> [GlitchedPixel] {
        let pointer = buffer.contents().bindMemory(to: UInt32.self, capacity: width * height)
        var glitched: [GlitchedPixel] = []

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let iteration = pointer[idx]
                // Non-zero means glitch was detected at that iteration
                if iteration > 0 {
                    glitched.append(GlitchedPixel(x: x, y: y, iteration: iteration))
                }
            }
        }

        return glitched
    }

    /// Clusters glitched pixels into groups that can share a reference point.
    /// Uses simple grid-based clustering for efficiency.
    /// - Parameter pixels: Array of glitched pixels
    /// - Parameter width: Image width
    /// - Parameter height: Image height
    /// - Returns: Array of pixel clusters
    func clusterPixels(_ pixels: [GlitchedPixel], width: Int, height: Int) -> [GlitchCluster] {
        guard !pixels.isEmpty else { return [] }

        // Simple grid-based clustering
        let gridSize = 4  // 4x4 grid
        var grid: [[GlitchedPixel]] = Array(repeating: [], count: gridSize * gridSize)

        let cellWidth = Double(width) / Double(gridSize)
        let cellHeight = Double(height) / Double(gridSize)

        for pixel in pixels {
            let gridX = min(gridSize - 1, Int(Double(pixel.x) / cellWidth))
            let gridY = min(gridSize - 1, Int(Double(pixel.y) / cellHeight))
            let idx = gridY * gridSize + gridX
            grid[idx].append(pixel)
        }

        // Convert grid cells with enough pixels into clusters
        var clusters: [GlitchCluster] = []

        for (idx, cellPixels) in grid.enumerated() {
            guard cellPixels.count >= minClusterSize else { continue }

            // Compute centroid
            let sumX = cellPixels.reduce(0.0) { $0 + Double($1.x) }
            let sumY = cellPixels.reduce(0.0) { $0 + Double($1.y) }
            let centroid = (x: sumX / Double(cellPixels.count),
                          y: sumY / Double(cellPixels.count))

            clusters.append(GlitchCluster(pixels: cellPixels, centroid: centroid, newReference: nil))
        }

        // Sort by cluster size (largest first) and limit
        clusters.sort { $0.pixels.count > $1.pixels.count }
        return Array(clusters.prefix(maxNewReferences))
    }

    /// Selects new reference points for each cluster.
    /// - Parameters:
    ///   - clusters: Pixel clusters needing new references
    ///   - viewCenter: Current view center in complex plane
    ///   - scale: Current zoom scale
    ///   - aspect: Width/height aspect ratio
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Clusters with assigned reference points
    func selectReferencePoints(for clusters: inout [GlitchCluster],
                               viewCenter: Complex<Double>,
                               scale: Double,
                               aspect: Double,
                               width: Int,
                               height: Int) {
        for i in 0..<clusters.count {
            // Convert centroid from pixel coordinates to complex plane
            let normX = clusters[i].centroid.x / Double(width)
            let normY = clusters[i].centroid.y / Double(height)

            let realOffset = (normX - 0.5) * scale * aspect
            let imagOffset = (normY - 0.5) * scale

            let newRef = Complex(
                viewCenter.real + realOffset,
                viewCenter.imaginary + imagOffset
            )

            clusters[i].newReference = newRef
        }
    }

    /// Analyzes glitch buffer and returns reference points for re-rendering.
    /// - Parameters:
    ///   - buffer: GPU glitch buffer
    ///   - viewCenter: Current view center
    ///   - scale: Current zoom scale
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Array of new reference points to try
    func analyzeAndSelectReferences(buffer: MTLBuffer,
                                    viewCenter: Complex<Double>,
                                    scale: Double,
                                    width: Int,
                                    height: Int) -> [Complex<Double>] {
        let pixels = extractGlitchedPixels(from: buffer, width: width, height: height)
        guard !pixels.isEmpty else { return [] }

        var clusters = clusterPixels(pixels, width: width, height: height)
        guard !clusters.isEmpty else { return [] }

        let aspect = Double(width) / Double(height)
        selectReferencePoints(for: &clusters,
                             viewCenter: viewCenter,
                             scale: scale,
                             aspect: aspect,
                             width: width,
                             height: height)

        return clusters.compactMap { $0.newReference }
    }

    /// Returns the percentage of pixels that were glitched.
    func glitchPercentage(from buffer: MTLBuffer, width: Int, height: Int) -> Double {
        let pixels = extractGlitchedPixels(from: buffer, width: width, height: height)
        let total = width * height
        return Double(pixels.count) / Double(total) * 100.0
    }
}

// MARK: - Glitch Mask

/// Generates a mask of which pixels need re-rendering.
class GlitchMask {
    private var mask: [Bool]
    let width: Int
    let height: Int

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.mask = Array(repeating: false, count: width * height)
    }

    /// Marks pixels as needing re-render based on glitch buffer.
    func updateFromBuffer(_ buffer: MTLBuffer) {
        let pointer = buffer.contents().bindMemory(to: UInt32.self, capacity: width * height)
        for i in 0..<(width * height) {
            mask[i] = pointer[i] > 0
        }
    }

    /// Expands the mask by a given radius to catch edge artifacts.
    func expand(radius: Int) {
        guard radius > 0 else { return }

        var expanded = mask

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                guard mask[idx] else { continue }

                // Mark neighboring pixels
                for dy in -radius...radius {
                    for dx in -radius...radius {
                        let nx = x + dx
                        let ny = y + dy
                        if nx >= 0 && nx < width && ny >= 0 && ny < height {
                            expanded[ny * width + nx] = true
                        }
                    }
                }
            }
        }

        mask = expanded
    }

    /// Returns whether a pixel needs re-rendering.
    func needsRerender(x: Int, y: Int) -> Bool {
        guard x >= 0 && x < width && y >= 0 && y < height else { return false }
        return mask[y * width + x]
    }

    /// Returns the number of pixels needing re-render.
    var count: Int {
        mask.filter { $0 }.count
    }

    /// Resets the mask to all false.
    func reset() {
        mask = Array(repeating: false, count: width * height)
    }
}
