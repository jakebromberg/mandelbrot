//
//  Renderer.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/8/25.
//

import MetalKit

// The Swift definition of the parameters, matching the Metal struct.
struct MandelbrotParams {
    var scale: Float          // Region size in the complex plane.
    var center: SIMD2<Float>  // Center of the region.
    var width: UInt32         // Image width (in pixels).
    var height: UInt32        // Image height (in pixels).
    var maxIterations: UInt32 // Maximum iterations to test for escape.
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var computePipelineState: MTLComputePipelineState
    var params: MandelbrotParams {
        didSet {
            print("params \(params)")
        }
    }
    var outputTexture: MTLTexture?
    
    init?(mtkView: MTKView) {
        // Ensure a Metal device is available.
        guard let device = mtkView.device else { return nil }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        self.commandQueue = commandQueue
        
        // Load the default Metal library and locate the compute function.
        guard let library = device.makeDefaultLibrary(),
              let kernelFunction = library.makeFunction(name: "mandelbrotKernel") else {
            return nil
        }
        
        do {
            computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
        } catch {
            print("Error creating compute pipeline state: \(error)")
            return nil
        }
        
        // Set default parameters (adjust scale/center to view the desired portion).
        let drawableSize = mtkView.drawableSize
        params = MandelbrotParams(
            scale: 1, // Height of the displayed region in the complex plane.
            center: SIMD2<Float>(0.0, 0.0), // Centering the fractal.
            width: UInt32(drawableSize.width),
            height: UInt32(drawableSize.height),
            maxIterations: UInt32(1000 / 5.5)
        )
        
        super.init()
        
        // Create an output texture to hold the computed image.
        createOutputTexture(size: drawableSize)
    }
    
    func createOutputTexture(size: CGSize) {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: Int(size.width),
            height: Int(size.height),
            mipmapped: false
        )
        // Allow the shader to write to and read from this texture.
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        outputTexture = device.makeTexture(descriptor: textureDescriptor)
    }
    
    // MTKViewDelegate: Called when the view’s drawable size changes.
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        if size.width < 1.0  ||
           size.height < 1.0 ||
           size.width.isNaN  ||
           size.height.isNaN {
            return
        }
        params.width = UInt32(size.width)
        params.height = UInt32(size.height)
        createOutputTexture(size: size)
    }
    
    // MTKViewDelegate: Called each frame.
    func draw(in view: MTKView) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let outputTexture = outputTexture,
              let drawable = view.currentDrawable else {
            fatalError()
        }
        
        computeEncoder.setComputePipelineState(computePipelineState)
        // Bind the output texture (index 0 in the shader).
        computeEncoder.setTexture(outputTexture, index: 0)
        // Pass in the Mandelbrot parameters (index 0 for the constant buffer).
        var currentParams = params
        computeEncoder.setBytes(&currentParams, length: MemoryLayout<MandelbrotParams>.stride, index: 0)
        
        // Set up thread groups based on the pipeline state.
        // Compute the threadgroup dimensions.
        let threadGroupWidth = computePipelineState.threadExecutionWidth
        let threadGroupHeight = computePipelineState.maxTotalThreadsPerThreadgroup / threadGroupWidth
        let threadsPerGroup = MTLSize(width: threadGroupWidth, height: threadGroupHeight, depth: 1)

        // Calculate the number of threadgroups needed by rounding up.
        let groupsX = (Int(params.width) + threadGroupWidth - 1) / threadGroupWidth
        let groupsY = (Int(params.height) + threadGroupHeight - 1) / threadGroupHeight
        let threadgroupsPerGrid = MTLSize(width: groupsX, height: groupsY, depth: 1)

        // Dispatch threadgroups.
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        // Use a blit pass to copy the output texture into the drawable’s texture.
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            let origin = MTLOrigin(x: 0, y: 0, z: 0)
            let size = MTLSize(width: Int(params.width), height: Int(params.height), depth: 1)
            blitEncoder.copy(
                from: outputTexture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: origin,
                sourceSize: size,
                to: drawable.texture,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: origin
            )
            blitEncoder.endEncoding()
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
