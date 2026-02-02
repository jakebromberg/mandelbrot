//
//  ReferenceOrbit.swift
//  Mandelbrot
//
//  Computes a high-precision reference orbit on CPU for perturbation theory rendering.
//

import Foundation
import Numerics

/// Computes and stores a reference orbit for perturbation theory.
/// The reference orbit is computed at full Double precision on the CPU,
/// then packed into Float pairs for GPU consumption.
class ReferenceOrbit {
    /// The center point in the complex plane where the reference orbit was computed.
    let center: Complex<Double>

    /// The computed orbit points (z values at each iteration).
    private(set) var orbit: [Complex<Double>] = []

    /// The number of iterations the reference point took to escape (or maxIterations if it didn't).
    private(set) var escapeIteration: Int = 0

    /// Whether the reference point escaped (vs. being in the set).
    private(set) var didEscape: Bool = false

    init(center: Complex<Double>) {
        self.center = center
    }

    /// Computes the reference orbit starting from c = center.
    /// - Parameter maxIterations: Maximum number of iterations to compute.
    /// - Returns: The number of iterations computed (may be less than maxIterations if point escapes).
    @discardableResult
    func computeOrbit(maxIterations: Int) -> Int {
        orbit.removeAll()
        orbit.reserveCapacity(maxIterations)

        var z = Complex<Double>.zero
        let c = center
        let escapeRadius: Double = 65536.0  // Same as shader for consistency

        for i in 0..<maxIterations {
            orbit.append(z)

            // z = z^2 + c
            z = z * z + c

            // Check for escape
            if z.lengthSquared > escapeRadius {
                escapeIteration = i + 1
                didEscape = true
                return orbit.count
            }
        }

        escapeIteration = maxIterations
        didEscape = false
        return orbit.count
    }

    /// Packs the orbit into a GPU-friendly format.
    /// Returns an array of Float pairs: [real0, imag0, real1, imag1, ...]
    func packForGPU() -> [Float] {
        var packed = [Float]()
        packed.reserveCapacity(orbit.count * 2)

        for z in orbit {
            packed.append(Float(z.real))
            packed.append(Float(z.imaginary))
        }

        return packed
    }

    /// Returns the number of points in the orbit.
    var count: Int {
        return orbit.count
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
