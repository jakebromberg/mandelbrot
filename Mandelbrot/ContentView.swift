//
//  ContentView.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/7/25.
//

import SwiftUI

struct ContentView: View {
    @State private var colorMode: UInt32 = 0 // 0=HSV, 1=Palette
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topTrailing) {
                MandelbrotView(
                    size: geometry.size,
                    scale: 2.0,
                    center: SIMD2<Float>(-0.75, 0.0),
                    colorMode: colorMode
                )
                .edgesIgnoringSafeArea(.all)

                Picker("Color", selection: $colorMode) {
                    Text("HSV").tag(UInt32(0))
                    Text("Palette").tag(UInt32(1))
                }
                .pickerStyle(.segmented)
                .padding(12)
                .background(.ultraThinMaterial, in: Capsule())
                .padding([.top, .trailing], 16)
            }
        }
    }
}

#Preview {
    ContentView()
}

