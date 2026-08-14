import CoreGraphics
import Foundation
import ScreenCaptureKit

enum CaptureFailure: Error {
    case arguments
    case display
    case context
}

func rgbBytes(from image: CGImage) throws -> Data {
    let width = image.width
    let height = image.height
    var rgba = Data(count: width * height * 4)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let rendered = rgba.withUnsafeMutableBytes { rawBuffer -> Bool in
        guard let address = rawBuffer.baseAddress else { return false }
        guard let context = CGContext(
            data: address,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    if !rendered { throw CaptureFailure.context }
    var rgb = Data(count: width * height * 3)
    rgba.withUnsafeBytes { rgbaBuffer in
        rgb.withUnsafeMutableBytes { rgbBuffer in
            let source = rgbaBuffer.bindMemory(to: UInt8.self)
            let destination = rgbBuffer.bindMemory(to: UInt8.self)
            for pixel in 0..<(width * height) {
                destination[pixel * 3] = source[pixel * 4]
                destination[pixel * 3 + 1] = source[pixel * 4 + 1]
                destination[pixel * 3 + 2] = source[pixel * 4 + 2]
            }
        }
    }
    return rgb
}

@main
struct CaptureMain {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 5,
              let x = Double(arguments[1]),
              let y = Double(arguments[2]),
              let width = Double(arguments[3]),
              let height = Double(arguments[4]),
              width > 0,
              height > 0 else {
            throw CaptureFailure.arguments
        }
        let requested = CGRect(x: x, y: y, width: width, height: height)
        let content = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        )
        guard let display = content.displays.first(where: {
            $0.frame.intersects(requested)
        }) else {
            throw CaptureFailure.display
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let configuration = SCStreamConfiguration()
        configuration.sourceRect = CGRect(
            x: requested.minX - display.frame.minX,
            y: requested.minY - display.frame.minY,
            width: requested.width,
            height: requested.height
        )
        // The learner frontend immediately reduces this public crop to its
        // configured image size. Capturing logical-window resolution avoids
        // transporting and converting four times as many Retina pixels.
        configuration.width = Int(requested.width)
        configuration.height = Int(requested.height)
        configuration.showsCursor = false
        configuration.scalesToFit = true
        let input = FileHandle.standardInput
        let output = FileHandle.standardOutput
        while !input.availableData.isEmpty {
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter,
                configuration: configuration
            )
            var header = [UInt32(image.width), UInt32(image.height)]
            output.write(Data(bytes: &header, count: MemoryLayout<UInt32>.size * 2))
            output.write(try rgbBytes(from: image))
        }
    }
}
