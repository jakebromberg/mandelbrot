//
//  Renderer.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/8/25.
//

import MetalKit
import Numerics

// The Swift definition of the parameters, matching the Metal struct.
struct MandelbrotParams {
    var scale: Float          // Region size in the complex plane.
    var scaleAspect: Float    // Precomputed scale * aspect ratio.
    var center: SIMD2<Float>  // Center of the region.
    var width: UInt32         // Image width (in pixels).
    var height: UInt32        // Image height (in pixels).
    var maxIterations: UInt32 // Maximum iterations to test for escape.
    var colorMode: UInt32     // 0 = HSV, 1 = Palette
    var options: UInt32       // bit 0: periodicity check
}

// Parameters for perturbation rendering, matching the Metal struct.
// Double-precision values are split into (hi, lo) float pairs for extended precision.
struct PerturbationParams {
    var referenceCenterHiLo: SIMD4<Float>  // (hi_x, lo_x, hi_y, lo_y)
    var viewCenterHiLo: SIMD4<Float>       // (hi_x, lo_x, hi_y, lo_y)
    var scaleHiLo: SIMD2<Float>            // (hi, lo)
    var scaleAspectHiLo: SIMD2<Float>      // (hi, lo)
    var width: UInt32
    var height: UInt32
    var maxIterations: UInt32
    var orbitLength: UInt32
    var colorMode: UInt32
    var padding: UInt32 = 0

    /// Splits a Double into high and low Float components.
    /// hi contains the value rounded to float precision.
    /// lo contains the error (original - hi), also as float.
    static func splitDouble(_ value: Double) -> (hi: Float, lo: Float) {
        let hi = Float(value)
        let lo = Float(value - Double(hi))
        return (hi, lo)
    }

    init(referenceCenter: Complex<Double>,
         viewCenter: Complex<Double>,
         scale: Double,
         scaleAspect: Double,
         width: UInt32,
         height: UInt32,
         maxIterations: UInt32,
         orbitLength: UInt32,
         colorMode: UInt32) {

        let refX = Self.splitDouble(referenceCenter.real)
        let refY = Self.splitDouble(referenceCenter.imaginary)
        self.referenceCenterHiLo = SIMD4(refX.hi, refX.lo, refY.hi, refY.lo)

        let viewX = Self.splitDouble(viewCenter.real)
        let viewY = Self.splitDouble(viewCenter.imaginary)
        self.viewCenterHiLo = SIMD4(viewX.hi, viewX.lo, viewY.hi, viewY.lo)

        let scaleSplit = Self.splitDouble(scale)
        self.scaleHiLo = SIMD2(scaleSplit.hi, scaleSplit.lo)

        let scaleAspectSplit = Self.splitDouble(scaleAspect)
        self.scaleAspectHiLo = SIMD2(scaleAspectSplit.hi, scaleAspectSplit.lo)

        self.width = width
        self.height = height
        self.maxIterations = maxIterations
        self.orbitLength = orbitLength
        self.colorMode = colorMode
        self.padding = 0
    }
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var computePipelineState: MTLComputePipelineState
    var perturbationPipelineState: MTLComputePipelineState?
    var params: MandelbrotParams
    var outputTexture: MTLTexture?
    var paletteTexture: MTLTexture?
    var paletteSampler: MTLSamplerState?

    // Perturbation theory support
    private(set) var renderingMode: RenderingMode = .standard
    private var referenceOrbit: ReferenceOrbit?
    private var referenceOrbitBuffer: MTLBuffer?
    private var centerDouble: Complex<Double> = Complex(0.0, 0.0)
    private var scaleDouble: Double = 1.0

    /// Scale threshold below which perturbation mode is used.
    /// At scales smaller than this, single-precision floats lose accuracy.
    private let perturbationThreshold: Double = 1e-6

    /// How much the center can drift (as fraction of scale) before recomputing reference orbit.
    private let referenceDriftThreshold: Double = 0.1
    
    init?(mtkView: MTKView) {
        // Ensure a Metal device is available.
        guard let device = mtkView.device else { return nil }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else { return nil }
        self.commandQueue = commandQueue

        // Load the default Metal library and locate the compute functions.
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

        // Create perturbation pipeline if the kernel exists
        if let perturbationFunction = library.makeFunction(name: "mandelbrotPerturbationKernel") {
            do {
                perturbationPipelineState = try device.makeComputePipelineState(function: perturbationFunction)
            } catch {
                print("Warning: Could not create perturbation pipeline: \(error)")
            }
        }

        // Set default parameters (adjust scale/center to view the desired portion).
        let drawableSize = mtkView.drawableSize
        let aspect = Float(drawableSize.width) / Float(drawableSize.height)
        params = MandelbrotParams(
            scale: 1,
            scaleAspect: 1 * aspect,
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

    // MARK: - Center and Scale Management

    /// Sets the center and scale, automatically choosing rendering mode and updating reference orbit as needed.
    func setCenter(_ center: Complex<Double>, scale: Double) {
        centerDouble = center
        scaleDouble = scale

        // Update single-precision params for standard rendering
        params.center = SIMD2<Float>(Float(center.real), Float(center.imaginary))
        params.scale = Float(scale)

        // Determine rendering mode based on scale
        let shouldUsePerturbation = scale < perturbationThreshold && perturbationPipelineState != nil

        if shouldUsePerturbation {
            renderingMode = .perturbation
            updateReferenceOrbitIfNeeded()
        } else {
            renderingMode = .standard
        }
    }

    /// Checks if reference orbit needs recomputation and updates it if necessary.
    private func updateReferenceOrbitIfNeeded() {
        guard renderingMode == .perturbation else { return }

        let needsRecompute: Bool
        if let existing = referenceOrbit {
            // Check if center has drifted too far from reference
            let drift = Complex(centerDouble.real - existing.center.real,
                               centerDouble.imaginary - existing.center.imaginary)
            let driftMagnitude = sqrt(drift.real * drift.real + drift.imaginary * drift.imaginary)
            needsRecompute = driftMagnitude > scaleDouble * referenceDriftThreshold
        } else {
            needsRecompute = true
        }

        if needsRecompute {
            computeReferenceOrbit()
        }
    }

    /// Computes a new reference orbit at the current center.
    private func computeReferenceOrbit() {
        let orbit = ReferenceOrbit(center: centerDouble)
        orbit.computeOrbit(maxIterations: Int(params.maxIterations))

        // Pack orbit data for GPU
        let packedData = orbit.packForGPU()

        // Create or update the Metal buffer
        if !packedData.isEmpty {
            let bufferSize = packedData.count * MemoryLayout<Float>.stride
            referenceOrbitBuffer = device.makeBuffer(bytes: packedData, length: bufferSize, options: .storageModeShared)
        } else {
            referenceOrbitBuffer = nil
        }

        referenceOrbit = orbit
    }

    // MARK: - MTKViewDelegate

    // MTKViewDelegate: Called when the view's drawable size changes.
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

        // Choose rendering path based on mode
        if renderingMode == .perturbation,
           let perturbationPipeline = perturbationPipelineState,
           let orbitBuffer = referenceOrbitBuffer,
           let orbit = referenceOrbit {
            // Perturbation rendering
            drawPerturbation(encoder: computeEncoder,
                           pipeline: perturbationPipeline,
                           orbitBuffer: orbitBuffer,
                           orbit: orbit,
                           outputTexture: outputTexture)
        } else {
            // Standard rendering
            drawStandard(encoder: computeEncoder, outputTexture: outputTexture)
        }

        computeEncoder.endEncoding()

        // Use a blit pass to copy the output texture into the drawable's texture.
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

    private func drawStandard(encoder: MTLComputeCommandEncoder, outputTexture: MTLTexture) {
        encoder.setComputePipelineState(computePipelineState)
        encoder.setTexture(outputTexture, index: 0)
        if let paletteTexture = paletteTexture {
            encoder.setTexture(paletteTexture, index: 1)
        }

        var currentParams = params
        currentParams.scaleAspect = params.scale * Float(params.width) / Float(params.height)
        encoder.setBytes(&currentParams, length: MemoryLayout<MandelbrotParams>.stride, index: 0)

        if let sampler = paletteSampler {
            encoder.setSamplerState(sampler, index: 0)
        }

        dispatchThreadgroups(encoder: encoder)
    }

    private func drawPerturbation(encoder: MTLComputeCommandEncoder,
                                  pipeline: MTLComputePipelineState,
                                  orbitBuffer: MTLBuffer,
                                  orbit: ReferenceOrbit,
                                  outputTexture: MTLTexture) {
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(outputTexture, index: 0)
        if let paletteTexture = paletteTexture {
            encoder.setTexture(paletteTexture, index: 1)
        }

        let aspect = Double(params.width) / Double(params.height)
        var perturbParams = PerturbationParams(
            referenceCenter: orbit.center,
            viewCenter: centerDouble,
            scale: scaleDouble,
            scaleAspect: scaleDouble * aspect,
            width: params.width,
            height: params.height,
            maxIterations: params.maxIterations,
            orbitLength: UInt32(orbit.count),
            colorMode: params.colorMode
        )

        encoder.setBytes(&perturbParams, length: MemoryLayout<PerturbationParams>.stride, index: 0)
        encoder.setBuffer(orbitBuffer, offset: 0, index: 1)

        if let sampler = paletteSampler {
            encoder.setSamplerState(sampler, index: 0)
        }

        dispatchThreadgroups(encoder: encoder)
    }

    private func dispatchThreadgroups(encoder: MTLComputeCommandEncoder) {
        let tgWidth = 16
        let tgHeight = 8
        let threadsPerGroup = MTLSize(width: tgWidth, height: tgHeight, depth: 1)
        let groupsX = (Int(params.width) + tgWidth - 1) / tgWidth
        let groupsY = (Int(params.height) + tgHeight - 1) / tgHeight
        let threadgroupsPerGrid = MTLSize(width: groupsX, height: groupsY, depth: 1)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }
}
