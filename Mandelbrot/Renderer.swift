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
    var colorMode: UInt32     // 0 = HSV, 1 = Palette
    var options: UInt32       // bit 0: periodicity check
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var computePipelineState: MTLComputePipelineState
    var params: MandelbrotParams
    var outputTexture: MTLTexture?
    var paletteTexture: MTLTexture?
    var paletteSampler: MTLSamplerState?
    
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
            scale: 1,
            center: SIMD2<Float>(0.0, 0.0),
            width: UInt32(drawableSize.width),
            height: UInt32(drawableSize.height),
            maxIterations: 512,
            colorMode: 0,
            options: 1 // enable periodicity
        )
        
        super.init()
        
        // Create an output texture to hold the computed image.
        createOutputTexture(size: drawableSize)

        // Create a default palette and sampler.
        createPaletteTexture()
        createPaletteSampler()
    }
    
    func createOutputTexture(size: CGSize) {
        if size.width < 1.0 || size.height < 1.0 || size.width.isNaN || size.height.isNaN {
            outputTexture = nil
            return
        }
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: Int(size.width),
            height: Int(size.height),
            mipmapped: false
        )
        // Allow the shader to write to and read from this texture.
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        outputTexture = device.makeTexture(descriptor: textureDescriptor)
    }

    private func createPaletteSampler() {
        let desc = MTLSamplerDescriptor()
        desc.minFilter = .linear
        desc.magFilter = .linear
        desc.mipFilter = .notMipmapped
        desc.sAddressMode = .clampToEdge
        desc.tAddressMode = .clampToEdge
        paletteSampler = device.makeSamplerState(descriptor: desc)
    }

    private func createPaletteTexture() {
        // Create a simple gradient palette (e.g., HSV-like rainbow) as width x 1.
        let width = 1024
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: 1,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else { return }

        var data = [UInt8](repeating: 0, count: width * 4)
        for i in 0..<width {
            let t = Float(i) / Float(width - 1)
            // Simple gradient: convert t to RGB via a piecewise rainbow.
            // Red → Yellow → Green → Cyan → Blue → Magenta → Red
            let x = t * 6.0
            let k = Int(x)
            let f = x - Float(k)
            var r: Float = 0, g: Float = 0, b: Float = 0
            switch k {
            case 0: r = 1; g = f; b = 0
            case 1: r = 1 - f; g = 1; b = 0
            case 2: r = 0; g = 1; b = f
            case 3: r = 0; g = 1 - f; b = 1
            case 4: r = f; g = 0; b = 1
            default: r = 1; g = 0; b = 1 - f
            }
            data[i*4+0] = UInt8(max(0, min(255, Int(r * 255))))
            data[i*4+1] = UInt8(max(0, min(255, Int(g * 255))))
            data[i*4+2] = UInt8(max(0, min(255, Int(b * 255))))
            data[i*4+3] = 255
        }
        data.withUnsafeBytes { rawPtr in
            let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: 1, depth: 1))
            tex.replace(region: region, mipmapLevel: 0, withBytes: rawPtr.baseAddress!, bytesPerRow: width * 4)
        }
        self.paletteTexture = tex
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
              let outputTexture = outputTexture,
              let drawable = view.currentDrawable,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(computePipelineState)
        // Bind the output texture (index 0 in the shader).
        computeEncoder.setTexture(outputTexture, index: 0)
        // Bind palette texture (index 1) if available
        if let paletteTexture = paletteTexture {
            computeEncoder.setTexture(paletteTexture, index: 1)
        }
        // Pass in the Mandelbrot parameters (index 0 for the constant buffer).
        var currentParams = params
        computeEncoder.setBytes(&currentParams, length: MemoryLayout<MandelbrotParams>.stride, index: 0)
        // Sampler at index 0
        if let sampler = paletteSampler {
            computeEncoder.setSamplerState(sampler, index: 0)
        }
        
        // Use fixed threadgroup size and dispatchThreadgroups with rounded-up group counts
        let tgWidth = 16
        let tgHeight = 8
        let threadsPerGroup = MTLSize(width: tgWidth, height: tgHeight, depth: 1)
        let groupsX = (Int(params.width) + tgWidth - 1) / tgWidth
        let groupsY = (Int(params.height) + tgHeight - 1) / tgHeight
        let threadgroupsPerGrid = MTLSize(width: groupsX, height: groupsY, depth: 1)
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
