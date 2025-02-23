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
    
    class Coordinator: NSObject {
        // These values hold the initial parameters when a gesture begins.
        var initialScale: Float = 2.0
        var initialCenter: SIMD2<Float> = SIMD2<Float>(-0.75, 0.0)
        // Will be updated as gestures change.
        var scale: Float = 2.0
        var center: SIMD2<Float> = SIMD2<Float>(-0.75, 0.0)
        
        // Stores the pinch’s center in view coordinates at the beginning of the gesture.
        var pinchCenter: CGPoint = .zero
        
        // A weak reference to the MTKView (if needed).
        weak var mtkView: MTKView?
        var renderer: Renderer?

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
                renderer?.params.maxIterations = UInt32(1000 / scale)
                mtkView?.setNeedsDisplay()
            case .ended, .cancelled:
                // Gesture finished—update the initial values.
                initialScale = scale
                initialCenter = center
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
            case .changed:
                // Map the translation (in pixels) to fractal-space translation.
                let dx = Float(translation.x) / Float(viewSize.width) * (scale * aspect)
                let dy = Float(translation.y) / Float(viewSize.height) * scale
                // Update the center (dragging right moves the fractal left, so subtract dx).
                center = initialCenter - SIMD2<Float>(dx, dy)
                renderer?.params.center = center
                mtkView?.setNeedsDisplay()
            case .ended, .cancelled:
                initialCenter = center
                mtkView?.setNeedsDisplay()
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
        mtkView.drawableSize = drawableSize
        
        // Create your Metal renderer (assumed implemented elsewhere).
        guard let renderer = Renderer(mtkView: mtkView) else {
            fatalError("Renderer failed to initialize")
        }
        mtkView.delegate = renderer
        context.coordinator.renderer = renderer
        
        // Save the view reference in the coordinator.
        context.coordinator.mtkView = mtkView
        
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
            print("scale \(scale)")
            renderer.params.scale = scale
            renderer.params.center = center
        }
        uiView.setNeedsDisplay()
    }
}
