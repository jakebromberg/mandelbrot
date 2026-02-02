//
//  Renderer.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/8/25.
//

import MetalKit
import Numerics
import Observation

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
    var skipIterations: UInt32             // Number of iterations to skip via series approximation

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
         colorMode: UInt32,
         skipIterations: UInt32 = 0) {

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
        self.skipIterations = skipIterations
    }
}

@Observable
class Renderer: NSObject, MTKViewDelegate {
    // MARK: - Observable State (Single Source of Truth)

    /// Current scale in the complex plane (smaller = deeper zoom)
    /// Note: Use displayScale for UI to avoid triggering updates during gestures
    var scale: Double = 2.0

    /// Current center point in the complex plane
    /// Note: Use displayCenter for UI to avoid triggering updates during gestures
    var center: Complex<Double> = Complex(-0.75, 0.0)

    /// Scale value for UI display - only updates on gesture end
    private(set) var displayScale: Double = 2.0

    /// Center value for UI display - only updates on gesture end
    private(set) var displayCenter: Complex<Double> = Complex(-0.75, 0.0)

    /// Color mode: 0 = HSV, 1 = Palette
    var colorMode: UInt32 = 0

    /// Current rendering mode (read-only for UI)
    private(set) var renderingMode: RenderingMode = .standard

    /// Current precision level (read-only for UI)
    private(set) var precisionLevel: PrecisionLevel = .double

    /// App frames per second (time between draw calls)
    private(set) var appFPS: Double = 0

    /// GPU frames per second (shader execution time)
    private(set) var gpuFPS: Double = 0

    // MARK: - Metal Resources

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var computePipelineState: MTLComputePipelineState
    var perturbationPipelineState: MTLComputePipelineState?
    var perturbationWithSeriesPipelineState: MTLComputePipelineState?
    var perturbationWithGlitchPipelineState: MTLComputePipelineState?
    var perturbationWithSeriesAndGlitchPipelineState: MTLComputePipelineState?
    var params: MandelbrotParams
    var outputTexture: MTLTexture?
    var paletteTexture: MTLTexture?
    var paletteSampler: MTLSamplerState?

    // MARK: - Perturbation Theory Support

    private var referenceOrbit: ReferenceOrbit?
    private var referenceOrbitBuffer: MTLBuffer?

    // Series approximation buffers
    private var seriesABuffer: MTLBuffer?
    private var seriesBBuffer: MTLBuffer?
    private var useSeriesApproximation: Bool = false

    // Glitch detection support
    private var glitchBuffer: MTLBuffer?
    private var glitchDetector: GlitchDetector?
    private var enableGlitchDetection: Bool = false

    // Multi-reference support
    var multiReferencePipelineState: MTLComputePipelineState?
    private var multiReferenceManager: MultiReferenceManager?
    private var useMultiReference: Bool = false

    // Precision level support
    private var referenceOrbitDD: ReferenceOrbitDD?

    // MARK: - FPS Tracking (Internal)

    private var lastDrawTime: CFAbsoluteTime = 0
    private var appFPSAccumulator: Double = 0
    private var gpuFPSAccumulator: Double = 0
    private var fpsFrameCount: Int = 0

    /// Scale threshold below which perturbation mode is used.
    /// At scales smaller than this, single-precision floats lose accuracy.
    private let perturbationThreshold: Double = 1e-6

    /// Scale threshold below which series approximation provides significant benefit.
    /// Deep zooms benefit most from skipping iterations.
    private let seriesApproximationThreshold: Double = 1e-7

    /// Scale threshold below which glitch detection is enabled.
    /// Very deep zooms are more prone to numerical instability.
    private let glitchDetectionThreshold: Double = 1e-8

    /// Scale threshold below which multiple reference points are used.
    /// Deep zooms benefit from regional references to reduce glitches.
    private let multiReferenceThreshold: Double = 1e-7

    /// How much the center can drift (as fraction of scale) before recomputing reference orbit.
    private let referenceDriftThreshold: Double = 0.1
    
    init?(device: MTLDevice? = nil) {
        // Use provided device or create system default
        guard let device = device ?? MTLCreateSystemDefaultDevice() else { return nil }
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

        // Create perturbation with series approximation pipeline
        if let perturbationSeriesFunction = library.makeFunction(name: "mandelbrotPerturbationWithSeriesKernel") {
            do {
                perturbationWithSeriesPipelineState = try device.makeComputePipelineState(function: perturbationSeriesFunction)
            } catch {
                print("Warning: Could not create perturbation with series pipeline: \(error)")
            }
        }

        // Create perturbation with glitch detection pipeline
        if let perturbationGlitchFunction = library.makeFunction(name: "mandelbrotPerturbationWithGlitchDetectionKernel") {
            do {
                perturbationWithGlitchPipelineState = try device.makeComputePipelineState(function: perturbationGlitchFunction)
            } catch {
                print("Warning: Could not create perturbation with glitch pipeline: \(error)")
            }
        }

        // Create perturbation with series and glitch detection pipeline
        if let perturbationSeriesGlitchFunction = library.makeFunction(name: "mandelbrotPerturbationWithSeriesAndGlitchKernel") {
            do {
                perturbationWithSeriesAndGlitchPipelineState = try device.makeComputePipelineState(function: perturbationSeriesGlitchFunction)
            } catch {
                print("Warning: Could not create perturbation with series and glitch pipeline: \(error)")
            }
        }

        // Initialize glitch detector
        glitchDetector = GlitchDetector()

        // Create multi-reference pipeline
        if let multiRefFunction = library.makeFunction(name: "mandelbrotMultiReferenceKernel") {
            do {
                multiReferencePipelineState = try device.makeComputePipelineState(function: multiRefFunction)
            } catch {
                print("Warning: Could not create multi-reference pipeline: \(error)")
            }
        }

        // Initialize multi-reference manager
        multiReferenceManager = MultiReferenceManager(gridSize: 3)

        // Set default parameters (will be updated when view size is known)
        params = MandelbrotParams(
            scale: 2.0,
            scaleAspect: 2.0,
            center: SIMD2<Float>(-0.75, 0.0),
            width: 1,
            height: 1,
            maxIterations: 512,
            colorMode: 0,
            options: 1 // enable periodicity
        )

        super.init()

        // Output texture will be created when drawable size is known (mtkView:drawableSizeWillChange)

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

    /// Updates rendering state based on current center and scale.
    /// - Parameter skipExpensiveUpdates: If true, skips reference orbit recomputation (for gesture performance)
    private func updateRenderingState(skipExpensiveUpdates: Bool = false) {
        // Update single-precision params for standard rendering
        params.center = SIMD2<Float>(Float(center.real), Float(center.imaginary))
        params.scale = Float(scale)

        // Determine rendering mode based on scale
        let shouldUsePerturbation = scale < perturbationThreshold && perturbationPipelineState != nil

        // Determine required precision level
        precisionLevel = PrecisionLevel.required(for: scale)

        if shouldUsePerturbation {
            renderingMode = .perturbation
            // Use series approximation for deeper zooms where it provides benefit
            useSeriesApproximation = scale < seriesApproximationThreshold && perturbationWithSeriesPipelineState != nil
            // Enable glitch detection for very deep zooms
            enableGlitchDetection = scale < glitchDetectionThreshold &&
                                   (perturbationWithGlitchPipelineState != nil || perturbationWithSeriesAndGlitchPipelineState != nil)
            // Use multiple reference points to reduce glitch artifacts
            useMultiReference = scale < multiReferenceThreshold && multiReferencePipelineState != nil

            // Skip expensive updates during gestures for better responsiveness
            if !skipExpensiveUpdates {
                updateReferenceOrbitIfNeeded()
                updateGlitchBufferIfNeeded()
                updateMultiReferenceIfNeeded()
            }
        } else {
            renderingMode = .standard
            useSeriesApproximation = false
            enableGlitchDetection = false
            useMultiReference = false
        }
    }

    /// Sets center and scale together, updating rendering state once.
    /// - Parameter lowQuality: If true, skips expensive updates and doesn't update display properties (use during gestures)
    func setView(center newCenter: Complex<Double>, scale newScale: Double, lowQuality: Bool = false) {
        center = newCenter
        scale = newScale
        updateRenderingState(skipExpensiveUpdates: lowQuality)

        // Only update display properties on gesture end to avoid SwiftUI churn
        if !lowQuality {
            displayScale = newScale
            displayCenter = newCenter
        }
    }

    /// Creates or updates the glitch buffer based on current image size.
    private func updateGlitchBufferIfNeeded() {
        guard enableGlitchDetection else {
            glitchBuffer = nil
            return
        }

        let requiredSize = Int(params.width) * Int(params.height) * MemoryLayout<UInt32>.stride
        if glitchBuffer == nil || glitchBuffer!.length < requiredSize {
            glitchBuffer = device.makeBuffer(length: requiredSize, options: .storageModeShared)
        }
    }

    /// Updates multi-reference manager when scale or center changes significantly.
    private func updateMultiReferenceIfNeeded() {
        guard useMultiReference, let manager = multiReferenceManager else {
            return
        }

        let aspect = Double(params.width) / Double(params.height)
        manager.partition(viewCenter: center,
                         scale: scale,
                         aspect: aspect,
                         maxIterations: Int(params.maxIterations),
                         device: device)
    }

    /// Checks if reference orbit needs recomputation and updates it if necessary.
    private func updateReferenceOrbitIfNeeded() {
        guard renderingMode == .perturbation else { return }

        let needsRecompute: Bool
        if let existing = referenceOrbit {
            // Check if center has drifted too far from reference
            let drift = Complex(center.real - existing.center.real,
                               center.imaginary - existing.center.imaginary)
            let driftMagnitude = sqrt(drift.real * drift.real + drift.imaginary * drift.imaginary)
            needsRecompute = driftMagnitude > scale * referenceDriftThreshold
        } else {
            needsRecompute = true
        }

        if needsRecompute {
            computeReferenceOrbit()
        }
    }

    /// Computes a new reference orbit at the current center.
    private func computeReferenceOrbit() {
        // Compute screen diagonal squared for series validity
        let aspect = Double(params.width) / Double(params.height)
        let screenWidth = scale * aspect
        let screenHeight = scale
        let screenDiagonalSquared = screenWidth * screenWidth + screenHeight * screenHeight

        // Choose precision based on zoom depth
        let orbit: ReferenceOrbit

        if precisionLevel == .doubleDouble {
            // Use double-double precision for ultra-deep zooms
            let orbitDD = ReferenceOrbitDD(center: center)
            orbitDD.computeOrbit(maxIterations: Int(params.maxIterations))
            referenceOrbitDD = orbitDD

            // Create a standard orbit wrapper from the DD orbit for compatibility
            orbit = ReferenceOrbit(center: center)
            // Copy the orbit data (converting back to simd_double2)
            orbit.orbitSimd = orbitDD.orbitAsSimd
        } else {
            // Standard double precision
            orbit = ReferenceOrbit(center: center)
            referenceOrbitDD = nil

            // Compute orbit with series approximation if enabled
            if useSeriesApproximation {
                orbit.computeOrbitWithSeries(maxIterations: Int(params.maxIterations),
                                             screenDiagonalSquared: screenDiagonalSquared)
            } else {
                orbit.computeOrbit(maxIterations: Int(params.maxIterations))
            }
        }

        // For DD precision, also compute series if enabled
        if precisionLevel == .doubleDouble && useSeriesApproximation {
            let series = SeriesApproximation()
            series.computeCoefficients(orbit: orbit.orbitSimd, maxDeltaSquared: screenDiagonalSquared)
            // Note: We'd need to expose a setter for series on ReferenceOrbit
            // For now, series is computed in the standard path
        }

        // Pack orbit data for GPU
        let packedData = orbit.packForGPU()

        // Create or update the Metal buffer
        if !packedData.isEmpty {
            let bufferSize = packedData.count * MemoryLayout<Float>.stride
            referenceOrbitBuffer = device.makeBuffer(bytes: packedData, length: bufferSize, options: .storageModeShared)
        } else {
            referenceOrbitBuffer = nil
        }

        // Create series approximation buffers if available
        if useSeriesApproximation, let series = orbit.series {
            let (aPacked, bPacked) = series.packForGPU()

            if !aPacked.isEmpty {
                let aBufferSize = aPacked.count * MemoryLayout<Float>.stride
                seriesABuffer = device.makeBuffer(bytes: aPacked, length: aBufferSize, options: .storageModeShared)
            } else {
                seriesABuffer = nil
            }

            if !bPacked.isEmpty {
                let bBufferSize = bPacked.count * MemoryLayout<Float>.stride
                seriesBBuffer = device.makeBuffer(bytes: bPacked, length: bBufferSize, options: .storageModeShared)
            } else {
                seriesBBuffer = nil
            }
        } else {
            seriesABuffer = nil
            seriesBBuffer = nil
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
        // Track app FPS (time between draw calls)
        let currentTime = CFAbsoluteTimeGetCurrent()
        if lastDrawTime > 0 {
            let frameDuration = currentTime - lastDrawTime
            if frameDuration > 0 {
                appFPSAccumulator += 1.0 / frameDuration
                fpsFrameCount += 1
            }
        }
        lastDrawTime = currentTime
        let gpuStartTime = currentTime

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let outputTexture = outputTexture,
              let drawable = view.currentDrawable,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        // Choose rendering path based on mode
        if renderingMode == .perturbation {
            // Check for multi-reference mode first
            if useMultiReference,
               let multiRefPipeline = multiReferencePipelineState,
               let manager = multiReferenceManager,
               let orbitBuffer = manager.combinedOrbitBuffer,
               let boundsBuffer = manager.regionBoundsBuffer,
               let centersBuffer = manager.regionCentersBuffer,
               let offsetsBuffer = manager.orbitOffsetsBuffer {
                // Multi-reference rendering
                drawMultiReference(encoder: computeEncoder,
                                  pipeline: multiRefPipeline,
                                  manager: manager,
                                  orbitBuffer: orbitBuffer,
                                  boundsBuffer: boundsBuffer,
                                  centersBuffer: centersBuffer,
                                  offsetsBuffer: offsetsBuffer,
                                  outputTexture: outputTexture)
            } else if let perturbationPipeline = perturbationPipelineState,
                      let orbitBuffer = referenceOrbitBuffer,
                      let orbit = referenceOrbit {
                // Single-reference perturbation rendering
                drawPerturbation(encoder: computeEncoder,
                               pipeline: perturbationPipeline,
                               orbitBuffer: orbitBuffer,
                               orbit: orbit,
                               outputTexture: outputTexture)
            } else {
                // Fallback to standard
                drawStandard(encoder: computeEncoder, outputTexture: outputTexture)
            }
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

        // Track GPU FPS via completion handler
        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            let gpuEndTime = CFAbsoluteTimeGetCurrent()
            let gpuDuration = gpuEndTime - gpuStartTime
            if gpuDuration > 0 {
                self.gpuFPSAccumulator += 1.0 / gpuDuration
            }

            // Update published FPS every 10 frames for smoother readings
            if self.fpsFrameCount >= 10 {
                DispatchQueue.main.async {
                    self.appFPS = self.appFPSAccumulator / Double(self.fpsFrameCount)
                    self.gpuFPS = self.gpuFPSAccumulator / Double(self.fpsFrameCount)
                    self.appFPSAccumulator = 0
                    self.gpuFPSAccumulator = 0
                    self.fpsFrameCount = 0
                }
            }
        }

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
        // Determine feature flags
        let canUseSeries = useSeriesApproximation &&
                          perturbationWithSeriesPipelineState != nil &&
                          seriesABuffer != nil &&
                          seriesBBuffer != nil &&
                          orbit.skipIterations > 1

        let canUseGlitch = enableGlitchDetection && glitchBuffer != nil

        // Select the appropriate pipeline based on enabled features
        let activePipeline: MTLComputePipelineState
        var glitchBufferIndex: Int = 2  // Default for no-series case

        if canUseSeries && canUseGlitch {
            if let seriesGlitchPipeline = perturbationWithSeriesAndGlitchPipelineState {
                activePipeline = seriesGlitchPipeline
                glitchBufferIndex = 4  // After seriesA (2), seriesB (3)
            } else if let seriesPipeline = perturbationWithSeriesPipelineState {
                activePipeline = seriesPipeline
            } else {
                activePipeline = pipeline
            }
        } else if canUseSeries {
            if let seriesPipeline = perturbationWithSeriesPipelineState {
                activePipeline = seriesPipeline
            } else {
                activePipeline = pipeline
            }
        } else if canUseGlitch {
            if let glitchPipeline = perturbationWithGlitchPipelineState {
                activePipeline = glitchPipeline
                glitchBufferIndex = 2
            } else {
                activePipeline = pipeline
            }
        } else {
            activePipeline = pipeline
        }

        encoder.setComputePipelineState(activePipeline)
        encoder.setTexture(outputTexture, index: 0)
        if let paletteTexture = paletteTexture {
            encoder.setTexture(paletteTexture, index: 1)
        }

        let aspect = Double(params.width) / Double(params.height)
        let skipIter = canUseSeries ? UInt32(orbit.skipIterations) : 0

        var perturbParams = PerturbationParams(
            referenceCenter: orbit.center,
            viewCenter: center,
            scale: scale,
            scaleAspect: scale * aspect,
            width: params.width,
            height: params.height,
            maxIterations: params.maxIterations,
            orbitLength: UInt32(orbit.count),
            colorMode: params.colorMode,
            skipIterations: skipIter
        )

        encoder.setBytes(&perturbParams, length: MemoryLayout<PerturbationParams>.stride, index: 0)
        encoder.setBuffer(orbitBuffer, offset: 0, index: 1)

        // Set series buffers if using series approximation
        if canUseSeries {
            encoder.setBuffer(seriesABuffer, offset: 0, index: 2)
            encoder.setBuffer(seriesBBuffer, offset: 0, index: 3)
        }

        // Set glitch buffer if using glitch detection
        if canUseGlitch && (activePipeline === perturbationWithGlitchPipelineState ||
                           activePipeline === perturbationWithSeriesAndGlitchPipelineState) {
            encoder.setBuffer(glitchBuffer, offset: 0, index: glitchBufferIndex)
        }

        if let sampler = paletteSampler {
            encoder.setSamplerState(sampler, index: 0)
        }

        dispatchThreadgroups(encoder: encoder)
    }

    private func drawMultiReference(encoder: MTLComputeCommandEncoder,
                                    pipeline: MTLComputePipelineState,
                                    manager: MultiReferenceManager,
                                    orbitBuffer: MTLBuffer,
                                    boundsBuffer: MTLBuffer,
                                    centersBuffer: MTLBuffer,
                                    offsetsBuffer: MTLBuffer,
                                    outputTexture: MTLTexture) {
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(outputTexture, index: 0)
        if let paletteTexture = paletteTexture {
            encoder.setTexture(paletteTexture, index: 1)
        }

        let aspect = Double(params.width) / Double(params.height)
        var multiRefParams = MultiReferenceParams(
            viewCenter: center,
            scale: scale,
            scaleAspect: scale * aspect,
            width: params.width,
            height: params.height,
            maxIterations: params.maxIterations,
            regionCount: UInt32(manager.regionCount),
            colorMode: params.colorMode
        )

        encoder.setBytes(&multiRefParams, length: MemoryLayout<MultiReferenceParams>.stride, index: 0)
        encoder.setBuffer(orbitBuffer, offset: 0, index: 1)
        encoder.setBuffer(boundsBuffer, offset: 0, index: 2)
        encoder.setBuffer(centersBuffer, offset: 0, index: 3)
        encoder.setBuffer(offsetsBuffer, offset: 0, index: 4)

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
