//
//  ReferenceOrbit.swift
//  Mandelbrot
//
//  Computes a high-precision reference orbit on CPU for perturbation theory rendering.
//

import Foundation
import Numerics
import simd
import Accelerate

/// Computes and stores a reference orbit for perturbation theory.
/// The reference orbit is computed at full Double precision on the CPU,
/// then packed into Float pairs for GPU consumption.
/// Uses simd_double2 for efficient complex arithmetic.
class ReferenceOrbit {
    /// The center point in the complex plane where the reference orbit was computed.
    let center: Complex<Double>

    /// The center as simd_double2 for efficient computation.
    let centerSimd: simd_double2

    /// The computed orbit points (z values at each iteration) using simd_double2.
    private(set) var orbitSimd: [simd_double2] = []

    /// The number of iterations the reference point took to escape (or maxIterations if it didn't).
    private(set) var escapeIteration: Int = 0

    /// Whether the reference point escaped (vs. being in the set).
    private(set) var didEscape: Bool = false

    /// Series approximation for skipping iterations.
    private(set) var series: SeriesApproximation?

    /// Maximum delta squared for the current view (for series validity).
    private var maxDeltaSquared: Double = 0

    init(center: Complex<Double>) {
        self.center = center
        self.centerSimd = simd_double2(center.real, center.imaginary)
    }

    /// Computes the reference orbit starting from c = center.
    /// - Parameter maxIterations: Maximum number of iterations to compute.
    /// - Returns: The number of iterations computed (may be less than maxIterations if point escapes).
    @discardableResult
    func computeOrbit(maxIterations: Int) -> Int {
        orbitSimd.removeAll()
        orbitSimd.reserveCapacity(maxIterations)

        var z = simd_double2.zero
        let c = centerSimd
        let escapeRadius: Double = 65536.0  // Same as shader for consistency

        for i in 0..<maxIterations {
            orbitSimd.append(z)

            // z = z^2 + c
            z = z.complexSquared() + c

            // Check for escape
            if simd_length_squared(z) > escapeRadius {
                escapeIteration = i + 1
                didEscape = true
                return orbitSimd.count
            }
        }

        escapeIteration = maxIterations
        didEscape = false
        return orbitSimd.count
    }

    /// Computes the reference orbit with series approximation coefficients.
    /// - Parameters:
    ///   - maxIterations: Maximum number of iterations to compute.
    ///   - screenDiagonalSquared: The maximum |δc|² for any pixel on screen.
    /// - Returns: The number of iterations computed.
    @discardableResult
    func computeOrbitWithSeries(maxIterations: Int, screenDiagonalSquared: Double) -> Int {
        maxDeltaSquared = screenDiagonalSquared

        // First compute the orbit
        let count = computeOrbit(maxIterations: maxIterations)

        // Then compute series coefficients
        let seriesApprox = SeriesApproximation()
        seriesApprox.computeCoefficients(orbit: orbitSimd, maxDeltaSquared: screenDiagonalSquared)
        series = seriesApprox

        return count
    }

    /// Packs the orbit into a GPU-friendly format using Accelerate.
    /// Returns an array of Float pairs: [real0, imag0, real1, imag1, ...]
    func packForGPU() -> [Float] {
        let count = orbitSimd.count
        guard count > 0 else { return [] }

        // Separate into real and imaginary arrays
        var reals = [Double](repeating: 0, count: count)
        var imags = [Double](repeating: 0, count: count)

        for (i, z) in orbitSimd.enumerated() {
            reals[i] = z.x
            imags[i] = z.y
        }

        // Convert double → float using vDSP (SIMD accelerated)
        var realsF = [Float](repeating: 0, count: count)
        var imagsF = [Float](repeating: 0, count: count)
        vDSP_vdpsp(reals, 1, &realsF, 1, vDSP_Length(count))
        vDSP_vdpsp(imags, 1, &imagsF, 1, vDSP_Length(count))

        // Interleave for GPU: [real0, imag0, real1, imag1, ...]
        var packed = [Float](repeating: 0, count: count * 2)
        cblas_scopy(Int32(count), realsF, 1, &packed, 2)      // Copy reals to even indices
        cblas_scopy(Int32(count), imagsF, 1, &packed[1], 2)   // Copy imags to odd indices

        return packed
    }

    /// Returns the number of iterations to skip using series approximation.
    var skipIterations: Int {
        return series?.validIterations ?? 0
    }

    /// Returns the number of points in the orbit.
    var count: Int {
        return orbitSimd.count
    }

    /// Legacy accessor for orbit as Complex<Double> array.
    var orbit: [Complex<Double>] {
        return orbitSimd.map { Complex($0.x, $0.y) }
    }
}

/// Rendering mode for the Mandelbrot renderer.
enum RenderingMode: String {
    case standard = "Standard"
    case perturbation = "Perturbation"
}

// MARK: - Complex extension for convenience

extension Complex where RealType == Double {
    /// The squared magnitude (avoids sqrt for escape checking).
    var lengthSquared: Double {
        return real * real + imaginary * imaginary
    }
}
