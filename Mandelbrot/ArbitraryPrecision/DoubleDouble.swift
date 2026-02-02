//
//  DoubleDouble.swift
//  Mandelbrot
//
//  Double-double arithmetic for extended precision (~30 decimal digits).
//  Represents a value as hi + lo where |lo| << |hi|.
//  Enables zooms to approximately 10^28.
//

import Foundation

/// Double-double arithmetic: represents value as hi + lo where |lo| << |hi|
/// This gives approximately 30 decimal digits of precision.
/// Conforms to standard Swift protocols for ergonomic use.
struct DoubleDouble: AdditiveArithmetic, Comparable, ExpressibleByFloatLiteral,
                     ExpressibleByIntegerLiteral, CustomStringConvertible {
    var hi: Double
    var lo: Double

    // MARK: - Initialization

    init(hi: Double, lo: Double) {
        self.hi = hi
        self.lo = lo
    }

    init(_ value: Double) {
        self.hi = value
        self.lo = 0
    }

    init(floatLiteral value: Double) {
        self.init(value)
    }

    init(integerLiteral value: Int) {
        self.init(Double(value))
    }

    // MARK: - AdditiveArithmetic

    static let zero = DoubleDouble(hi: 0, lo: 0)

    static func +(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
        let (s1, e1) = twoSum(a.hi, b.hi)
        let (s2, e2) = twoSum(a.lo, b.lo)
        let e3 = e1 + s2
        let (s3, e4) = twoSum(s1, e3)
        return DoubleDouble(hi: s3, lo: e2 + e4)
    }

    static func -(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
        a + DoubleDouble(hi: -b.hi, lo: -b.lo)
    }

    static func +=(a: inout DoubleDouble, b: DoubleDouble) {
        a = a + b
    }

    static func -=(a: inout DoubleDouble, b: DoubleDouble) {
        a = a - b
    }

    // MARK: - Multiplication

    static func *(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
        let (p, e) = twoProd(a.hi, b.hi)
        let e2 = e + a.hi * b.lo + a.lo * b.hi
        return DoubleDouble(hi: p + e2, lo: (p - (p + e2)) + e2)
    }

    static func *=(a: inout DoubleDouble, b: DoubleDouble) {
        a = a * b
    }

    // MARK: - Division

    static func /(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
        // Newton-Raphson division: a/b = a * (1/b)
        // First approximation
        let q1 = a.hi / b.hi

        // Compute remainder: r = a - q1 * b
        let (p, e) = twoProd(q1, b.hi)
        let r = DoubleDouble(hi: a.hi - p, lo: a.lo - e - q1 * b.lo)

        // Refine
        let q2 = r.hi / b.hi
        return DoubleDouble(hi: q1 + q2, lo: (q1 - (q1 + q2)) + q2)
    }

    static func /=(a: inout DoubleDouble, b: DoubleDouble) {
        a = a / b
    }

    // MARK: - Comparable

    static func <(a: DoubleDouble, b: DoubleDouble) -> Bool {
        a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo)
    }

    static func ==(a: DoubleDouble, b: DoubleDouble) -> Bool {
        a.hi == b.hi && a.lo == b.lo
    }

    // MARK: - Convenience

    static let one = DoubleDouble(hi: 1, lo: 0)

    /// Negation
    prefix static func -(a: DoubleDouble) -> DoubleDouble {
        DoubleDouble(hi: -a.hi, lo: -a.lo)
    }

    /// Absolute value
    var magnitude: DoubleDouble {
        hi < 0 ? -self : self
    }

    /// Convert to Double (loses precision)
    var doubleValue: Double { hi + lo }

    var description: String {
        "DoubleDouble(\(hi) + \(lo))"
    }

    // MARK: - Error-Free Algorithms

    /// Two-sum: computes s = a + b and e = error such that a + b = s + e exactly
    private static func twoSum(_ a: Double, _ b: Double) -> (s: Double, e: Double) {
        let s = a + b
        let v = s - a
        let e = (a - (s - v)) + (b - v)
        return (s, e)
    }

    /// Two-prod: computes p = a * b and e = error using FMA
    private static func twoProd(_ a: Double, _ b: Double) -> (p: Double, e: Double) {
        let p = a * b
        let e = fma(a, b, -p)
        return (p, e)
    }
}

// MARK: - Scalar Multiplication

extension DoubleDouble {
    static func *(a: Double, b: DoubleDouble) -> DoubleDouble {
        let p = a * b.hi
        let e = fma(a, b.hi, -p)
        return DoubleDouble(hi: p, lo: e + a * b.lo)
    }

    static func *(a: DoubleDouble, b: Double) -> DoubleDouble {
        b * a
    }
}

// MARK: - Math Functions

extension DoubleDouble {
    /// Square of the value
    func squared() -> DoubleDouble {
        self * self
    }

    /// Square root using Newton-Raphson
    func sqrt() -> DoubleDouble {
        guard hi > 0 else { return .zero }

        // Initial approximation
        let x0 = Foundation.sqrt(hi)

        // One Newton-Raphson iteration: x1 = (x0 + self/x0) / 2
        let dd_x0 = DoubleDouble(x0)
        let x1 = 0.5 * (dd_x0 + self / dd_x0)

        // Another iteration for more precision
        return 0.5 * (x1 + self / x1)
    }
}
