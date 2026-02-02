//
//  ComplexDD.swift
//  Mandelbrot
//
//  Complex number arithmetic using double-double precision.
//  Enables Mandelbrot zooms to approximately 10^28.
//

import Foundation

// Using Complex from Numerics module (imported via swift-numerics dependency)
import Numerics

/// Complex number with double-double precision components.
struct ComplexDD: AdditiveArithmetic, Equatable, CustomStringConvertible {
    var real: DoubleDouble
    var imag: DoubleDouble

    // MARK: - Initialization

    init(real: DoubleDouble, imag: DoubleDouble) {
        self.real = real
        self.imag = imag
    }

    init(_ real: Double, _ imag: Double) {
        self.real = DoubleDouble(real)
        self.imag = DoubleDouble(imag)
    }

    init(from complex: Complex<Double>) {
        self.real = DoubleDouble(complex.real)
        self.imag = DoubleDouble(complex.imaginary)
    }

    // MARK: - AdditiveArithmetic

    static let zero = ComplexDD(real: .zero, imag: .zero)

    static func +(a: ComplexDD, b: ComplexDD) -> ComplexDD {
        ComplexDD(real: a.real + b.real, imag: a.imag + b.imag)
    }

    static func -(a: ComplexDD, b: ComplexDD) -> ComplexDD {
        ComplexDD(real: a.real - b.real, imag: a.imag - b.imag)
    }

    static func +=(a: inout ComplexDD, b: ComplexDD) {
        a = a + b
    }

    static func -=(a: inout ComplexDD, b: ComplexDD) {
        a = a - b
    }

    // MARK: - Multiplication

    static func *(a: ComplexDD, b: ComplexDD) -> ComplexDD {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        ComplexDD(
            real: a.real * b.real - a.imag * b.imag,
            imag: a.real * b.imag + a.imag * b.real
        )
    }

    static func *=(a: inout ComplexDD, b: ComplexDD) {
        a = a * b
    }

    // MARK: - Convenience

    static let one = ComplexDD(real: .one, imag: .zero)

    /// Negation
    prefix static func -(a: ComplexDD) -> ComplexDD {
        ComplexDD(real: -a.real, imag: -a.imag)
    }

    /// Squared value: z * z
    func squared() -> ComplexDD {
        // (a + bi)² = (a² - b²) + 2abi
        ComplexDD(
            real: real * real - imag * imag,
            imag: 2.0 * real * imag
        )
    }

    /// Squared magnitude: |z|² = a² + b²
    var magnitudeSquared: DoubleDouble {
        real * real + imag * imag
    }

    /// Magnitude: |z|
    var magnitude: DoubleDouble {
        magnitudeSquared.sqrt()
    }

    /// Convert to standard Complex<Double> (loses precision)
    var asComplex: Complex<Double> {
        Complex(real.doubleValue, imag.doubleValue)
    }

    var description: String {
        "(\(real.doubleValue), \(imag.doubleValue)i)"
    }
}

// MARK: - Scalar Multiplication

extension ComplexDD {
    static func *(a: Double, b: ComplexDD) -> ComplexDD {
        ComplexDD(real: a * b.real, imag: a * b.imag)
    }

    static func *(a: ComplexDD, b: Double) -> ComplexDD {
        b * a
    }

    static func *(a: DoubleDouble, b: ComplexDD) -> ComplexDD {
        ComplexDD(real: a * b.real, imag: a * b.imag)
    }

    static func *(a: ComplexDD, b: DoubleDouble) -> ComplexDD {
        b * a
    }
}

// MARK: - Precision Level

/// Determines the required precision level based on zoom depth.
enum PrecisionLevel: String, CaseIterable {
    case double       // 10^0 - 10^14
    case doubleDouble // 10^14 - 10^28

    /// The human-readable name
    var displayName: String {
        switch self {
        case .double: return "Double"
        case .doubleDouble: return "Double-Double"
        }
    }

    /// Maximum zoom depth (as exponent) supported by this precision level.
    var maxZoomExponent: Double {
        switch self {
        case .double: return 14
        case .doubleDouble: return 28
        }
    }

    /// Determines the required precision level for a given scale.
    /// - Parameter scale: The current zoom scale (smaller = deeper zoom)
    /// - Returns: The appropriate precision level
    static func required(for scale: Double) -> PrecisionLevel {
        let digits = -log10(max(scale, 1e-30)) + 3  // +3 safety margin
        if digits < 14 { return .double }
        return .doubleDouble
    }
}

// MARK: - Conversion Helpers

extension Complex where RealType == Double {
    /// Convert to ComplexDD (extends precision)
    var asComplexDD: ComplexDD {
        ComplexDD(real, imaginary)
    }
}
