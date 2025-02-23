//
//  ContentView.swift
//  Mandelbrot
//
//  Created by Jake Bromberg on 2/7/25.
//

import SwiftUI
import CoreGraphics

struct ContentView: View {
    var body: some View {
        GeometryReader { geometry in
            MandelbrotView(
                size: geometry.size,
                scale: 2.0,
                center: SIMD2<Float>(-0.75, 0.0)
            )
            .edgesIgnoringSafeArea(.all)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

#Preview {
    ContentView()
}

