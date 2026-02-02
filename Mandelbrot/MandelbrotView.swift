//
//  MandelbrotView.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/8/25.
//

import SwiftUI
import MetalKit

struct MandelbrotView: UIViewRepresentable {
    let size: CGSize
    var drawableSize: CGSize {
        let dimension = max(size.width, size.height)
        return CGSize(width: dimension, height: dimension)
    }
    var scale: Float
    var center: SIMD2<Float>
    var colorMode: UInt32 = 0
    
    class Coordinator: NSObject {
        // These values hold the initial parameters when a gesture begins.
        var initialScale: Float = 2.0
        var initialCenter: SIMD2<Float> = SIMD2<Float>(-0.75, 0.0)
        // Will be updated as gestures change.
        var scale: Float = 2.0
        var center: SIMD2<Float> = SIMD2<Float>(-0.75, 0.0)
        var colorMode: UInt32 = 0
        
        // Stores the pinch’s center in view coordinates at the beginning of the gesture.
        var pinchCenter: CGPoint = .zero
        
        // A weak reference to the MTKView (if needed).
        weak var mtkView: MTKView?
        var renderer: Renderer?
        var fullDrawableSize: CGSize = .zero
        let lowResFactor: CGFloat = 0.5

        private func recommendedMaxIterations(scale: Float, lowQuality: Bool) -> UInt32 {
            // Empirical mapping: grows with zoom depth; cap to reasonable bounds
            let depth = max(1.0, log2f(1.0 / max(1e-6, scale)) + 1.0)
            let base: Float = 256.0
            var iters = base * depth * 2.0
            if lowQuality { iters *= 0.33 }
            let clamped = max(64.0, min(8192.0, iters))
            return UInt32(clamped)
        }

        // Callbacks from gesture recognizers will update our scale and center.
        @objc func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            guard let view = recognizer.view else { return }
            let location = recognizer.location(in: view)
            let viewSize = view.bounds.size
            // Calculate the aspect ratio (width / height)
            let aspect = Float(viewSize.width / viewSize.height)
            
            switch recognizer.state {
            case .began:
                // Record the starting scale, center, and pinch location.
                initialScale = scale
                initialCenter = center
                pinchCenter = location
                // Lower resolution for interactive responsiveness
                if let mtkView = mtkView {
                    fullDrawableSize = mtkView.drawableSize
                    let low = CGSize(width: fullDrawableSize.width * lowResFactor, height: fullDrawableSize.height * lowResFactor)
                    mtkView.drawableSize = low
                }
            case .changed:
                // Compute the new scale. (A pinch-out, where recognizer.scale > 1, should zoom in, so new scale becomes smaller.)
                let newScale = initialScale / Float(recognizer.scale)
                
                // Get normalized coordinates of the pinch center (values in [0,1]).
                let pNorm = CGPoint(x: pinchCenter.x / viewSize.width,
                                    y: pinchCenter.y / viewSize.height)
                
                // Compute the fractal coordinate at the pinch point using the initial parameters.
                let fractalX = (Float(pNorm.x) - 0.5) * initialScale * aspect + initialCenter.x
                let fractalY = (Float(pNorm.y) - 0.5) * initialScale + initialCenter.y
                
                // Now compute the new center so that the fractal coordinate at the pinch point remains the same.
                let newCenterX = fractalX - (Float(pNorm.x) - 0.5) * newScale * aspect
                let newCenterY = fractalY - (Float(pNorm.y) - 0.5) * newScale
                
                // Update our stored values.
                scale = newScale
                center = SIMD2<Float>(newCenterX, newCenterY)
                renderer?.params.center = center
                renderer?.params.scale = scale
                if let renderer = renderer {
                    renderer.params.maxIterations = recommendedMaxIterations(scale: scale, lowQuality: true)
                    renderer.params.colorMode = colorMode
                }
                mtkView?.draw()
            case .ended, .cancelled:
                // Gesture finished—update the initial values.
                initialScale = scale
                initialCenter = center
                // Restore full resolution and higher iterations, then draw
                if let mtkView = mtkView, fullDrawableSize != .zero {
                    mtkView.drawableSize = fullDrawableSize
                }
                if let renderer = renderer {
                    renderer.params.maxIterations = recommendedMaxIterations(scale: scale, lowQuality: false)
                }
                mtkView?.draw()
            default:
                break
            }
        }
        
        @objc func handlePan(_ recognizer: UIPanGestureRecognizer) {
            guard let view = recognizer.view else { return }
            let translation = recognizer.translation(in: view)
            let viewSize = view.bounds.size
            let aspect = Float(viewSize.width / viewSize.height)
            
            switch recognizer.state {
            case .began:
                initialCenter = center
                // Lower resolution at pan start
                if let mtkView = mtkView {
                    fullDrawableSize = mtkView.drawableSize
                    let low = CGSize(width: fullDrawableSize.width * lowResFactor, height: fullDrawableSize.height * lowResFactor)
                    mtkView.drawableSize = low
                }
            case .changed:
                // Map the translation (in pixels) to fractal-space translation.
                let dx = Float(translation.x) / Float(viewSize.width) * (scale * aspect)
                let dy = Float(translation.y) / Float(viewSize.height) * scale
                // Update the center (dragging right moves the fractal left, so subtract dx).
                center = initialCenter - SIMD2<Float>(dx, dy)
                renderer?.params.center = center
                if let renderer = renderer {
                    renderer.params.maxIterations = recommendedMaxIterations(scale: scale, lowQuality: true)
                    renderer.params.colorMode = colorMode
                }
                mtkView?.draw()
            case .ended, .cancelled:
                initialCenter = center
                // Restore resolution and draw high quality
                if let mtkView = mtkView, fullDrawableSize != .zero {
                    mtkView.drawableSize = fullDrawableSize
                }
                if let renderer = renderer {
                    renderer.params.maxIterations = recommendedMaxIterations(scale: scale, lowQuality: false)
                }
                mtkView?.draw()
            default:
                break
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator()
    }
    
    func makeUIView(context: Context) -> MTKView {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        // Create the MTKView with the provided size.
        let mtkView = MTKView(frame: CGRect(origin: .zero, size: drawableSize), device: device)
        mtkView.framebufferOnly = false  // So we can use the drawable as a blit target.
        mtkView.isPaused = true
        mtkView.enableSetNeedsDisplay = true
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.drawableSize = drawableSize
        
        // Create your Metal renderer (assumed implemented elsewhere).
        guard let renderer = Renderer(mtkView: mtkView) else {
            fatalError("Renderer failed to initialize")
        }
        mtkView.delegate = renderer
        context.coordinator.renderer = renderer
        
        // Save the view reference in the coordinator.
        context.coordinator.mtkView = mtkView
        context.coordinator.fullDrawableSize = drawableSize
        context.coordinator.scale = scale
        context.coordinator.center = center
        context.coordinator.colorMode = colorMode
        
        // Add the UIPinchGestureRecognizer.
        let pinchRecognizer = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        mtkView.addGestureRecognizer(pinchRecognizer)
        
        // Add the UIPanGestureRecognizer for panning.
        let panRecognizer = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        mtkView.addGestureRecognizer(panRecognizer)
        
        return mtkView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        // Pass the updated scale and center to your renderer.
        if let renderer = uiView.delegate as? Renderer {
            renderer.params.scale = scale
            renderer.params.center = center
            renderer.params.colorMode = colorMode
            context.coordinator.colorMode = colorMode
        }
        uiView.draw()
    }
}
