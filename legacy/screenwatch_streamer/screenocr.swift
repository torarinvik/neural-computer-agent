// screenocr — positioned text extraction from a screen image via the Vision framework.
//
// Part of the screen-understanding orchestra (see SPEC.md): the "guitar" member. A pure
// function from pixels to positioned strings; no state, no daemon.
//
// Usage: screenocr <image> [--crop x,y,w,h] [--min-conf 0.4]
// Output: one line per recognized string:  <x> <y> <w> <h> <conf> <text>
//         Coordinates are pixels in the FULL image's space (top-left origin), even when
//         cropped, so grid->pixel evidence pointers stay valid. conf is 0-1.
// Exit: 0 on success, 1 on unreadable image / bad args.

import Foundation
import Vision
import CoreImage

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

var imagePath: String? = nil
var crop: CGRect? = nil
var minConf: Float = 0.4

var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--crop":
        guard !args.isEmpty else { fail("--crop needs x,y,w,h") }
        let parts = args.removeFirst().split(separator: ",").compactMap { Double($0) }
        guard parts.count == 4 else { fail("--crop needs x,y,w,h") }
        crop = CGRect(x: parts[0], y: parts[1], width: parts[2], height: parts[3])
    case "--min-conf":
        guard !args.isEmpty, let v = Float(args.removeFirst()) else { fail("--min-conf needs a number") }
        minConf = v
    default:
        if imagePath == nil { imagePath = a } else { fail("unexpected argument: \(a)") }
    }
}

guard let path = imagePath else { fail("usage: screenocr <image> [--crop x,y,w,h] [--min-conf 0.4]") }
guard let ciImage = CIImage(contentsOf: URL(fileURLWithPath: path)) else { fail("unreadable image: \(path)") }

let fullW = ciImage.extent.width
let fullH = ciImage.extent.height

// CIImage origin is bottom-left; our crop rect is top-left-origin pixel space.
var work = ciImage
var offX: CGFloat = 0
var offY: CGFloat = 0   // top-left-space y offset of the crop
if let c = crop {
    let flipped = CGRect(x: c.minX, y: fullH - c.maxY, width: c.width, height: c.height)
    work = ciImage.cropped(to: flipped.intersection(ciImage.extent))
    offX = c.minX
    offY = c.minY
}
let workW = work.extent.width
let workH = work.extent.height
guard workW > 1, workH > 1 else { fail("empty crop") }

let ctx = CIContext()
guard let cg = ctx.createCGImage(work, from: work.extent) else { fail("render failed") }

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do { try handler.perform([request]) } catch { fail("vision error: \(error)") }

for obs in request.results ?? [] {
    guard let cand = obs.topCandidates(1).first, cand.confidence >= minConf else { continue }
    // bounding box is normalized, bottom-left origin, in the WORK image's space
    let bb = obs.boundingBox
    let x = offX + bb.minX * workW
    let yTop = offY + (1 - bb.maxY) * workH
    let w = bb.width * workW
    let h = bb.height * workH
    let text = cand.string.replacingOccurrences(of: "\n", with: " ")
    print("\(Int(x)) \(Int(yTop)) \(Int(w)) \(Int(h)) \(String(format: "%.2f", cand.confidence)) \(text)")
}
