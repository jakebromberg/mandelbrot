//
//  MandelbrotView.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/8/25.
//

import SwiftUI
import MetalKit
import Numerics

struct MandelbrotView: UIViewRepresentable {
    let renderer: Renderer
    let size: CGSize

    var drawableSize: CGSize {
        let dimension = max(size.width, size.height)
        return CGSize(width: dimension, height: dimension)
    }

    class Coordinator: NSObject {
        let renderer: Renderer

        // Gesture state
        private var initialScale: Double = 2.0
        private var initialCenter: Complex<Double> = Complex(-0.75, 0.0)
        private var pinchCenter: CGPoint = .zero

        // Resolution management
        weak var mtkView: MTKView?
        var fullDrawableSize: CGSize = .zero
        let lowResFactor: CGFloat = 0.5

        init(renderer: Renderer) {
            self.renderer = renderer
            super.init()
        }

        private func recommendedMaxIterations(scale: Double, lowQuality: Bool) -> UInt32 {
            let depth = max(1.0, log2(1.0 / max(1e-14, scale)) + 1.0)
            let base: Double = 256.0
            var iters = base * depth * 2.0
            if lowQuality { iters *= 0.33 }
            return UInt32(max(64.0, min(16384.0, iters)))
        }

        @objc func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            guard let view = recognizer.view else { return }
            let location = recognizer.location(in: view)
            let viewSize = view.bounds.size
            let aspect = Double(viewSize.width / viewSize.height)

            switch recognizer.state {
            case .began:
                initialScale = renderer.scale
                initialCenter = renderer.center
                pinchCenter = location
                if let mtkView = mtkView {
                    fullDrawableSize = mtkView.drawableSize
                    mtkView.drawableSize = CGSize(
                        width: fullDrawableSize.width * lowResFactor,
                        height: fullDrawableSize.height * lowResFactor
                    )
                }

            case .changed:
                let newScale = initialScale / Double(recognizer.scale)
                let pNormX = Double(pinchCenter.x / viewSize.width)
                let pNormY = Double(pinchCenter.y / viewSize.height)

                let fractalX = (pNormX - 0.5) * initialScale * aspect + initialCenter.real
                let fractalY = (pNormY - 0.5) * initialScale + initialCenter.imaginary

                let newCenterX = fractalX - (pNormX - 0.5) * newScale * aspect
                let newCenterY = fractalY - (pNormY - 0.5) * newScale

                renderer.scale = newScale
                renderer.center = Complex(newCenterX, newCenterY)
                renderer.params.maxIterations = recommendedMaxIterations(scale: newScale, lowQuality: true)
                mtkView?.draw()

            case .ended, .cancelled:
                initialScale = renderer.scale
                initialCenter = renderer.center
                if let mtkView = mtkView, fullDrawableSize != .zero {
                    mtkView.drawableSize = fullDrawableSize
                }
                renderer.params.maxIterations = recommendedMaxIterations(scale: renderer.scale, lowQuality: false)
                mtkView?.draw()

            default:
                break
            }
        }

        @objc func handlePan(_ recognizer: UIPanGestureRecognizer) {
            guard let view = recognizer.view else { return }
            let translation = recognizer.translation(in: view)
            let viewSize = view.bounds.size
            let aspect = Double(viewSize.width / viewSize.height)

            switch recognizer.state {
            case .began:
                initialCenter = renderer.center
                if let mtkView = mtkView {
                    fullDrawableSize = mtkView.drawableSize
                    mtkView.drawableSize = CGSize(
                        width: fullDrawableSize.width * lowResFactor,
                        height: fullDrawableSize.height * lowResFactor
                    )
                }

            case .changed:
                let dx = Double(translation.x) / Double(viewSize.width) * (renderer.scale * aspect)
                let dy = Double(translation.y) / Double(viewSize.height) * renderer.scale
                renderer.center = Complex(initialCenter.real - dx, initialCenter.imaginary - dy)
                renderer.params.maxIterations = recommendedMaxIterations(scale: renderer.scale, lowQuality: true)
                mtkView?.draw()

            case .ended, .cancelled:
                initialCenter = renderer.center
                if let mtkView = mtkView, fullDrawableSize != .zero {
                    mtkView.drawableSize = fullDrawableSize
                }
                renderer.params.maxIterations = recommendedMaxIterations(scale: renderer.scale, lowQuality: false)
                mtkView?.draw()

            default:
                break
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: CGRect(origin: .zero, size: drawableSize), device: renderer.device)
        mtkView.framebufferOnly = false
        mtkView.isPaused = true
        mtkView.enableSetNeedsDisplay = true
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.drawableSize = drawableSize
        mtkView.delegate = renderer

        context.coordinator.mtkView = mtkView
        context.coordinator.fullDrawableSize = drawableSize

        let pinchRecognizer = UIPinchGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePinch(_:))
        )
        mtkView.addGestureRecognizer(pinchRecognizer)

        let panRecognizer = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePan(_:))
        )
        mtkView.addGestureRecognizer(panRecognizer)

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        uiView.draw()
    }
}
