//
//  SeriesApproximation.swift
//  Mandelbrot
//
//  Series approximation for perturbation theory.
//  Taylor series: δₙ ≈ Aₙ·δc + Bₙ·δc²
//  Allows skipping 50-90% of GPU iterations by precomputing polynomial coefficients.
//

import Foundation
import simd
import Accelerate

// MARK: - Complex simd_double2 Operations

extension simd_double2 {
    /// Complex multiplication: (a.x + a.y*i) * (b.x + b.y*i)
    static func complexMul(_ a: Self, _ b: Self) -> Self {
        return simd_double2(
            a.x * b.x - a.y * b.y,
            a.x * b.y + a.y * b.x
        )
    }

    /// Complex squared: z * z
    func complexSquared() -> simd_double2 {
        return simd_double2(x * x - y * y, 2 * x * y)
    }
}

// MARK: - Series Approximation

/// Computes series approximation coefficients alongside the reference orbit.
/// The series δₙ ≈ Aₙ·δc + Bₙ·δc² allows the GPU to skip early iterations.
class SeriesApproximation {
    /// The A coefficients (linear term)
    private(set) var aCoefficients: [simd_double2] = []

    /// The B coefficients (quadratic term)
    private(set) var bCoefficients: [simd_double2] = []

    /// Number of iterations where series approximation is valid
    private(set) var validIterations: Int = 0

    /// Computes series coefficients alongside orbit computation.
    /// - Parameters:
    ///   - orbit: The orbit points (z values at each iteration)
    ///   - maxDeltaSquared: Maximum |δc|² for the screen (determines validity)
    /// - Returns: The number of valid iterations for series approximation
    @discardableResult
    func computeCoefficients(orbit: [simd_double2], maxDeltaSquared: Double) -> Int {
        aCoefficients.removeAll()
        bCoefficients.removeAll()
        aCoefficients.reserveCapacity(orbit.count)
        bCoefficients.reserveCapacity(orbit.count)

        guard !orbit.isEmpty else {
            validIterations = 0
            return 0
        }

        var A = simd_double2(1, 0)  // A₀ = 1
        var B = simd_double2.zero   // B₀ = 0

        validIterations = orbit.count

        for i in 0..<orbit.count {
            aCoefficients.append(A)
            bCoefficients.append(B)

            let z = orbit[i]

            // Check series validity: |Bₙ·δc²| should be small relative to |Aₙ·δc|
            // This ensures the quadratic term doesn't dominate
            let aMagSq = simd_length_squared(A)
            let bMagSq = simd_length_squared(B)

            // When |B|·|δc|² > ε·|A|·|δc|, series becomes unreliable
            // Simplifies to: |B|·|δc| > ε·|A|
            if bMagSq * maxDeltaSquared > 1e-6 * aMagSq && validIterations == orbit.count {
                validIterations = i
            }

            // Recurrence relations:
            // Aₙ₊₁ = 2·zₙ·Aₙ + 1
            // Bₙ₊₁ = 2·zₙ·Bₙ + Aₙ²
            let twoZ = 2.0 * z
            let newA = simd_double2.complexMul(twoZ, A) + simd_double2(1, 0)
            let newB = simd_double2.complexMul(twoZ, B) + simd_double2.complexMul(A, A)

            A = newA
            B = newB
        }

        // Ensure we have at least some valid iterations (minimum 1 to avoid starting at 0)
        if validIterations == 0 {
            validIterations = min(1, orbit.count)
        }

        return validIterations
    }

    /// Packs series coefficients for GPU using Accelerate for optimized conversion.
    /// Returns interleaved Float pairs: [A0.x, A0.y, A1.x, A1.y, ...] and same for B
    func packForGPU() -> (aBuffer: [Float], bBuffer: [Float]) {
        let count = validIterations
        guard count > 0 else {
            return ([], [])
        }

        // Use Accelerate for double→float conversion
        let aPacked = packCoefficients(Array(aCoefficients.prefix(count)))
        let bPacked = packCoefficients(Array(bCoefficients.prefix(count)))

        return (aPacked, bPacked)
    }

    /// Packs a coefficient array using Accelerate's vDSP for optimized conversion.
    private func packCoefficients(_ coefficients: [simd_double2]) -> [Float] {
        let count = coefficients.count
        guard count > 0 else { return [] }

        // Separate into real and imaginary arrays
        var reals = [Double](repeating: 0, count: count)
        var imags = [Double](repeating: 0, count: count)

        for (i, z) in coefficients.enumerated() {
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
}

// MARK: - Orbit with Series Integration

/// Extended reference orbit computation that includes series coefficients.
class ReferenceOrbitWithSeries {
    /// The center point in the complex plane.
    let center: simd_double2

    /// The computed orbit points (z values) using simd_double2 for efficiency.
    private(set) var orbit: [simd_double2] = []

    /// Series approximation coefficients
    let series: SeriesApproximation

    /// The number of iterations the reference point took to escape.
    private(set) var escapeIteration: Int = 0

    /// Whether the reference point escaped.
    private(set) var didEscape: Bool = false

    /// Maximum |δc|² for series validity computation (screen diagonal squared).
    var maxDeltaSquared: Double = 0

    init(center: simd_double2) {
        self.center = center
        self.series = SeriesApproximation()
    }

    /// Convenience initializer from Complex<Double>.
    convenience init(centerReal: Double, centerImag: Double) {
        self.init(center: simd_double2(centerReal, centerImag))
    }

    /// Computes the reference orbit and series coefficients in a single pass.
    /// - Parameters:
    ///   - maxIterations: Maximum number of iterations.
    ///   - screenDiagonalSquared: The maximum |δc|² for any pixel on screen.
    /// - Returns: The number of iterations computed.
    @discardableResult
    func computeOrbitWithSeries(maxIterations: Int, screenDiagonalSquared: Double) -> Int {
        orbit.removeAll()
        orbit.reserveCapacity(maxIterations)

        maxDeltaSquared = screenDiagonalSquared

        var z = simd_double2.zero
        let c = center
        let escapeRadius: Double = 65536.0

        for i in 0..<maxIterations {
            orbit.append(z)

            // z = z² + c
            z = z.complexSquared() + c

            // Check for escape
            if simd_length_squared(z) > escapeRadius {
                escapeIteration = i + 1
                didEscape = true
                break
            }
        }

        if !didEscape {
            escapeIteration = maxIterations
        }

        // Compute series coefficients
        series.computeCoefficients(orbit: orbit, maxDeltaSquared: screenDiagonalSquared)

        return orbit.count
    }

    /// Packs the orbit into a GPU-friendly format using Accelerate.
    func packOrbitForGPU() -> [Float] {
        let count = orbit.count
        guard count > 0 else { return [] }

        // Separate into real and imaginary arrays
        var reals = [Double](repeating: 0, count: count)
        var imags = [Double](repeating: 0, count: count)

        for (i, z) in orbit.enumerated() {
            reals[i] = z.x
            imags[i] = z.y
        }

        // Convert double → float using vDSP
        var realsF = [Float](repeating: 0, count: count)
        var imagsF = [Float](repeating: 0, count: count)
        vDSP_vdpsp(reals, 1, &realsF, 1, vDSP_Length(count))
        vDSP_vdpsp(imags, 1, &imagsF, 1, vDSP_Length(count))

        // Interleave for GPU
        var packed = [Float](repeating: 0, count: count * 2)
        cblas_scopy(Int32(count), realsF, 1, &packed, 2)
        cblas_scopy(Int32(count), imagsF, 1, &packed[1], 2)

        return packed
    }

    /// Returns the number of iterations to skip using series approximation.
    var skipIterations: Int {
        return series.validIterations
    }

    /// Returns the total orbit length.
    var count: Int {
        return orbit.count
    }
}
