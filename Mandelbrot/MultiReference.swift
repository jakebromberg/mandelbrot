//
//  MultiReference.swift
//  Mandelbrot
//
//  Multiple reference point management for perturbation theory.
//  Partitions screen into regions with independent reference orbits
//  to reduce glitch frequency by 60-90%.
//

import Foundation
import Metal
import simd
import Accelerate

// Using Complex from Numerics module (imported via swift-numerics dependency)
import Numerics

// MARK: - Reference Region

/// Represents a screen region with its own reference orbit.
struct ReferenceRegion {
    /// Screen region bounds (normalized 0-1)
    let bounds: (minX: Double, minY: Double, maxX: Double, maxY: Double)

    /// Reference point in the complex plane
    let center: Complex<Double>

    /// Offset into the combined orbit buffer
    var orbitBufferOffset: Int

    /// Length of this region's orbit
    var orbitLength: Int

    /// Series approximation valid iterations for this region
    var skipIterations: Int
}

// MARK: - Multi-Reference Manager

/// Manages multiple reference points for reduced glitch artifacts.
class MultiReferenceManager {
    /// The reference regions
    private(set) var regions: [ReferenceRegion] = []

    /// Combined orbit buffer for all regions
    private(set) var combinedOrbitBuffer: MTLBuffer?

    /// Combined series A coefficients buffer
    private(set) var combinedSeriesABuffer: MTLBuffer?

    /// Combined series B coefficients buffer
    private(set) var combinedSeriesBBuffer: MTLBuffer?

    /// Region metadata packed for GPU
    private(set) var regionBoundsBuffer: MTLBuffer?
    private(set) var regionCentersBuffer: MTLBuffer?
    private(set) var orbitOffsetsBuffer: MTLBuffer?

    /// Grid size (NxN regions)
    private let gridSize: Int

    /// Maximum iterations per orbit
    private var maxIterations: Int = 0

    /// Device for buffer creation
    private weak var device: MTLDevice?

    /// Reference orbits for each region
    private var orbits: [ReferenceOrbit] = []

    init(gridSize: Int = 3) {
        self.gridSize = gridSize
    }

    /// Partitions the screen into regions and computes reference points.
    /// - Parameters:
    ///   - viewCenter: Current view center in complex plane
    ///   - scale: Current zoom scale
    ///   - aspect: Width/height aspect ratio
    ///   - maxIterations: Maximum iterations to compute
    ///   - device: Metal device for buffer creation
    func partition(viewCenter: Complex<Double>,
                   scale: Double,
                   aspect: Double,
                   maxIterations: Int,
                   device: MTLDevice) {
        self.device = device
        self.maxIterations = maxIterations
        regions.removeAll()
        orbits.removeAll()

        let regionWidth = 1.0 / Double(gridSize)
        let regionHeight = 1.0 / Double(gridSize)

        // Screen dimensions in complex plane
        let screenWidth = scale * aspect
        let screenHeight = scale

        // Compute screen diagonal squared for series validity
        let screenDiagonalSq = screenWidth * screenWidth + screenHeight * screenHeight
        // For a region, use the region diagonal instead
        let regionDiagonalSq = screenDiagonalSq / Double(gridSize * gridSize)

        for row in 0..<gridSize {
            for col in 0..<gridSize {
                let minX = Double(col) * regionWidth
                let minY = Double(row) * regionHeight
                let maxX = minX + regionWidth
                let maxY = minY + regionHeight

                // Region center in normalized coordinates
                let normX = (minX + maxX) / 2.0
                let normY = (minY + maxY) / 2.0

                // Convert to complex plane
                let realOffset = (normX - 0.5) * screenWidth
                let imagOffset = (normY - 0.5) * screenHeight
                let refCenter = Complex(
                    viewCenter.real + realOffset,
                    viewCenter.imaginary + imagOffset
                )

                let region = ReferenceRegion(
                    bounds: (minX, minY, maxX, maxY),
                    center: refCenter,
                    orbitBufferOffset: 0,
                    orbitLength: 0,
                    skipIterations: 0
                )
                regions.append(region)

                // Compute orbit for this region
                let orbit = ReferenceOrbit(center: refCenter)
                orbit.computeOrbitWithSeries(maxIterations: maxIterations,
                                            screenDiagonalSquared: regionDiagonalSq)
                orbits.append(orbit)
            }
        }

        // Update buffer offsets
        var offset = 0
        for i in 0..<regions.count {
            regions[i].orbitBufferOffset = offset
            regions[i].orbitLength = orbits[i].count
            regions[i].skipIterations = orbits[i].skipIterations
            offset += orbits[i].count
        }

        // Create combined buffers
        createCombinedBuffers(device: device)
    }

    /// Creates combined GPU buffers from all region orbits.
    private func createCombinedBuffers(device: MTLDevice) {
        // Pack all orbits into one buffer
        var allOrbitData: [Float] = []
        var allSeriesA: [Float] = []
        var allSeriesB: [Float] = []

        for orbit in orbits {
            allOrbitData.append(contentsOf: orbit.packForGPU())

            if let series = orbit.series {
                let (aPacked, bPacked) = series.packForGPU()
                allSeriesA.append(contentsOf: aPacked)
                allSeriesB.append(contentsOf: bPacked)
            }
        }

        // Create orbit buffer
        if !allOrbitData.isEmpty {
            let size = allOrbitData.count * MemoryLayout<Float>.stride
            combinedOrbitBuffer = device.makeBuffer(bytes: allOrbitData, length: size, options: .storageModeShared)
        }

        // Create series buffers
        if !allSeriesA.isEmpty {
            let sizeA = allSeriesA.count * MemoryLayout<Float>.stride
            combinedSeriesABuffer = device.makeBuffer(bytes: allSeriesA, length: sizeA, options: .storageModeShared)
        }
        if !allSeriesB.isEmpty {
            let sizeB = allSeriesB.count * MemoryLayout<Float>.stride
            combinedSeriesBBuffer = device.makeBuffer(bytes: allSeriesB, length: sizeB, options: .storageModeShared)
        }

        // Create region metadata buffers
        createRegionMetadataBuffers(device: device)
    }

    /// Creates GPU buffers for region metadata.
    private func createRegionMetadataBuffers(device: MTLDevice) {
        // Bounds: [minX, minY, maxX, maxY] as float4 per region
        var boundsData: [Float] = []
        for region in regions {
            boundsData.append(Float(region.bounds.minX))
            boundsData.append(Float(region.bounds.minY))
            boundsData.append(Float(region.bounds.maxX))
            boundsData.append(Float(region.bounds.maxY))
        }
        if !boundsData.isEmpty {
            let size = boundsData.count * MemoryLayout<Float>.stride
            regionBoundsBuffer = device.makeBuffer(bytes: boundsData, length: size, options: .storageModeShared)
        }

        // Centers: [real_hi, real_lo, imag_hi, imag_lo] as float4 per region
        var centersData: [Float] = []
        for region in regions {
            let realHi = Float(region.center.real)
            let realLo = Float(region.center.real - Double(realHi))
            let imagHi = Float(region.center.imaginary)
            let imagLo = Float(region.center.imaginary - Double(imagHi))
            centersData.append(realHi)
            centersData.append(realLo)
            centersData.append(imagHi)
            centersData.append(imagLo)
        }
        if !centersData.isEmpty {
            let size = centersData.count * MemoryLayout<Float>.stride
            regionCentersBuffer = device.makeBuffer(bytes: centersData, length: size, options: .storageModeShared)
        }

        // Orbit offsets and lengths: [offset, length, skipIterations, padding] as uint4 per region
        var offsetsData: [UInt32] = []
        for region in regions {
            offsetsData.append(UInt32(region.orbitBufferOffset))
            offsetsData.append(UInt32(region.orbitLength))
            offsetsData.append(UInt32(region.skipIterations))
            offsetsData.append(0)  // padding
        }
        if !offsetsData.isEmpty {
            let size = offsetsData.count * MemoryLayout<UInt32>.stride
            orbitOffsetsBuffer = device.makeBuffer(bytes: offsetsData, length: size, options: .storageModeShared)
        }
    }

    /// Returns the number of regions.
    var regionCount: Int {
        return regions.count
    }

    /// Returns the total orbit data size.
    var totalOrbitPoints: Int {
        return orbits.reduce(0) { $0 + $1.count }
    }

    /// Finds the region containing a given normalized screen coordinate.
    func regionIndex(at normalizedX: Double, normalizedY: Double) -> Int {
        for (i, region) in regions.enumerated() {
            if normalizedX >= region.bounds.minX && normalizedX < region.bounds.maxX &&
               normalizedY >= region.bounds.minY && normalizedY < region.bounds.maxY {
                return i
            }
        }
        return 0  // Default to first region
    }

    /// Returns the orbit for a specific region.
    func orbit(for regionIndex: Int) -> ReferenceOrbit? {
        guard regionIndex >= 0 && regionIndex < orbits.count else { return nil }
        return orbits[regionIndex]
    }

    /// Returns the region at the given index.
    func region(at index: Int) -> ReferenceRegion? {
        guard index >= 0 && index < regions.count else { return nil }
        return regions[index]
    }

    /// Clears all data.
    func clear() {
        regions.removeAll()
        orbits.removeAll()
        combinedOrbitBuffer = nil
        combinedSeriesABuffer = nil
        combinedSeriesBBuffer = nil
        regionBoundsBuffer = nil
        regionCentersBuffer = nil
        orbitOffsetsBuffer = nil
    }
}

// MARK: - Multi-Reference Params

/// Parameters for multi-reference rendering, matching the Metal struct.
struct MultiReferenceParams {
    var viewCenterHiLo: SIMD4<Float>       // (hi_x, lo_x, hi_y, lo_y)
    var scaleHiLo: SIMD2<Float>            // (hi, lo)
    var scaleAspectHiLo: SIMD2<Float>      // (hi, lo)
    var width: UInt32
    var height: UInt32
    var maxIterations: UInt32
    var regionCount: UInt32
    var colorMode: UInt32
    var padding1: UInt32 = 0
    var padding2: UInt32 = 0
    var padding3: UInt32 = 0

    static func splitDouble(_ value: Double) -> (hi: Float, lo: Float) {
        let hi = Float(value)
        let lo = Float(value - Double(hi))
        return (hi, lo)
    }

    init(viewCenter: Complex<Double>,
         scale: Double,
         scaleAspect: Double,
         width: UInt32,
         height: UInt32,
         maxIterations: UInt32,
         regionCount: UInt32,
         colorMode: UInt32) {

        let viewX = Self.splitDouble(viewCenter.real)
        let viewY = Self.splitDouble(viewCenter.imaginary)
        self.viewCenterHiLo = SIMD4(viewX.hi, viewX.lo, viewY.hi, viewY.lo)

        let scaleSplit = Self.splitDouble(scale)
        self.scaleHiLo = SIMD2(scaleSplit.hi, scaleSplit.lo)

        let scaleAspectSplit = Self.splitDouble(scaleAspect)
        self.scaleAspectHiLo = SIMD2(scaleAspectSplit.hi, scaleAspectSplit.lo)

        self.width = width
        self.height = height
        self.maxIterations = maxIterations
        self.regionCount = regionCount
        self.colorMode = colorMode
    }
}
