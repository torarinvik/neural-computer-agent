import AppKit
import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Darwin
import Foundation
import ScreenCaptureKit

enum CaptureFailure: Error {
    case arguments
    case window
    case stream
    case format
}

private let magic = Data([0x4E, 0x43, 0x41, 0x31]) // NCA1
private let lookbackSamples = 19_200 // 400 ms at 48 kHz
private let ringCapacity = 48_000

private final class SampleRing {
    private var samples: [Int16]
    private var writeIndex = 0
    private var count = 0

    init(capacity: Int) {
        samples = Array(repeating: 0, count: capacity)
    }

    func append(_ values: [Int16]) {
        guard !values.isEmpty else { return }
        for value in values {
            samples[writeIndex] = value
            writeIndex = (writeIndex + 1) % samples.count
            if count < samples.count {
                count += 1
            }
        }
    }

    func last(_ wanted: Int) -> [Int16] {
        let take = min(wanted, count)
        guard take > 0 else { return [] }
        var result = Array(repeating: Int16(0), count: take)
        var index = (writeIndex - take + samples.count) % samples.count
        for offset in 0..<take {
            result[offset] = samples[index]
            index = (index + 1) % samples.count
        }
        return result
    }
}

private final class WindowAVSession: NSObject, SCStreamOutput, SCStreamDelegate {
    private let lock = NSLock()
    private var latestRGB = Data()
    private var width: UInt32 = 0
    private var height: UInt32 = 0
    private var audioActive = false
    private let ring = SampleRing(capacity: ringCapacity)
    private var stream: SCStream?

    func start(window: SCWindow) async throws {
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let configuration = SCStreamConfiguration()
        configuration.width = max(2, Int(window.frame.width.rounded()))
        configuration.height = max(2, Int(window.frame.height.rounded()))
        configuration.showsCursor = false
        configuration.scalesToFit = true
        configuration.queueDepth = 3
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 12)
        configuration.pixelFormat = kCVPixelFormatType_32BGRA
        configuration.capturesAudio = true
        configuration.sampleRate = 48_000
        configuration.channelCount = 2
        configuration.excludesCurrentProcessAudio = true
        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: DispatchQueue(label: "nca.sck.video"))
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: DispatchQueue(label: "nca.sck.audio"))
        try await stream.startCapture()
        lock.lock()
        audioActive = true
        lock.unlock()
        self.stream = stream
    }

    func snapshot() -> (
        rgb: Data,
        width: UInt32,
        height: UInt32,
        pcm: Data,
        audioActive: Bool
    ) {
        lock.lock()
        defer { lock.unlock() }
        let pcmSamples = ring.last(lookbackSamples)
        var pcm = Data(count: pcmSamples.count * 2)
        pcm.withUnsafeMutableBytes { raw in
            let dest = raw.bindMemory(to: Int16.self)
            for index in pcmSamples.indices {
                dest[index] = pcmSamples[index]
            }
        }
        return (latestRGB, width, height, pcm, audioActive)
    }

    func stop() async {
        if let stream {
            try? await stream.stopCapture()
        }
        stream = nil
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        if type == .screen {
            storeVideo(sampleBuffer)
        } else if type == .audio {
            storeAudio(sampleBuffer)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write(Data("sck stream stopped: \(error)\n".utf8))
    }

    private func storeVideo(_ sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = sampleBuffer.imageBuffer,
              let rgb = rgbFromPixelBuffer(pixelBuffer) else { return }
        lock.lock()
        latestRGB = rgb.data
        width = UInt32(rgb.width)
        height = UInt32(rgb.height)
        lock.unlock()
    }

    private func storeAudio(_ sampleBuffer: CMSampleBuffer) {
        guard let converted = monoInt16(from: sampleBuffer) else { return }
        lock.lock()
        audioActive = true
        ring.append(converted)
        lock.unlock()
    }
}

private func rgbFromPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> (data: Data, width: Int, height: Int)? {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
    var rgb = Data(count: width * height * 3)
    let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
    rgb.withUnsafeMutableBytes { destRaw in
        let dest = destRaw.bindMemory(to: UInt8.self)
        let source = base.assumingMemoryBound(to: UInt8.self)
        for row in 0..<height {
            let line = source.advanced(by: row * stride)
            for column in 0..<width {
                let src = column * 4
                let out = (row * width + column) * 3
                if format == kCVPixelFormatType_32ARGB {
                    dest[out] = line[src + 1]
                    dest[out + 1] = line[src + 2]
                    dest[out + 2] = line[src + 3]
                } else {
                    dest[out] = line[src + 2]
                    dest[out + 1] = line[src + 1]
                    dest[out + 2] = line[src]
                }
            }
        }
    }
    return (rgb, width, height)
}

private func monoInt16(from sampleBuffer: CMSampleBuffer) -> [Int16]? {
    var sizeNeeded = 0
    var status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: &sizeNeeded,
        bufferListOut: nil,
        bufferListSize: 0,
        blockBufferAllocator: kCFAllocatorDefault,
        blockBufferMemoryAllocator: kCFAllocatorDefault,
        flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
        blockBufferOut: nil
    )
    if status != noErr || sizeNeeded < MemoryLayout<AudioBufferList>.size {
        return nil
    }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: sizeNeeded, alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { raw.deallocate() }
    let list = raw.bindMemory(to: AudioBufferList.self, capacity: 1)
    var blockBuffer: CMBlockBuffer?
    status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: nil,
        bufferListOut: list,
        bufferListSize: sizeNeeded,
        blockBufferAllocator: kCFAllocatorDefault,
        blockBufferMemoryAllocator: kCFAllocatorDefault,
        flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
        blockBufferOut: &blockBuffer
    )
    if status != noErr {
        return nil
    }
    guard let asbd = CMSampleBufferGetFormatDescription(sampleBuffer).flatMap({
        CMAudioFormatDescriptionGetStreamBasicDescription($0)?.pointee
    }) else { return nil }
    let buffers = UnsafeMutableAudioBufferListPointer(list)
    let frames = Int(CMSampleBufferGetNumSamples(sampleBuffer))
    guard frames > 0 else { return [] }
    var mono = Array(repeating: Float(0), count: frames)
    if asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0 {
        for buffer in buffers {
            guard let pointer = buffer.mData?.assumingMemoryBound(to: Float.self) else { continue }
            let channelCount = max(Int(buffer.mNumberChannels), 1)
            for frame in 0..<frames {
                var sum = Float(0)
                for channel in 0..<channelCount {
                    sum += pointer[frame * channelCount + channel]
                }
                mono[frame] += sum / Float(channelCount)
            }
        }
        if buffers.count > 1 {
            let scale = 1.0 / Float(buffers.count)
            for frame in 0..<frames {
                mono[frame] *= scale
            }
        }
    } else if asbd.mBitsPerChannel == 16 {
        for buffer in buffers {
            guard let pointer = buffer.mData?.assumingMemoryBound(to: Int16.self) else { continue }
            let channelCount = max(Int(buffer.mNumberChannels), 1)
            for frame in 0..<frames {
                var sum = Float(0)
                for channel in 0..<channelCount {
                    sum += Float(pointer[frame * channelCount + channel]) / 32768.0
                }
                mono[frame] += sum / Float(channelCount)
            }
        }
        if buffers.count > 1 {
            let scale = 1.0 / Float(buffers.count)
            for frame in 0..<frames {
                mono[frame] *= scale
            }
        }
    } else {
        return nil
    }
    return mono.map { value in
        let clipped = max(-1.0, min(1.0, value))
        return Int16((clipped * 32767.0).rounded())
    }
}

private func chooseWindow(pid: pid_t) async throws -> SCWindow {
    let content = try await SCShareableContent.excludingDesktopWindows(
        false,
        onScreenWindowsOnly: true
    )
    let matches = content.windows.filter { window in
        window.owningApplication?.processID == pid
            && window.isOnScreen
            && window.frame.width >= 32
            && window.frame.height >= 32
    }
    guard let window = matches.max(by: {
        $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height
    }) else {
        throw CaptureFailure.window
    }
    return window
}

private func listWindows(pid: pid_t?) async throws {
    let content = try await SCShareableContent.excludingDesktopWindows(
        false,
        onScreenWindowsOnly: true
    )
    var rows: [[String: Any]] = []
    for window in content.windows {
        guard window.isOnScreen,
              window.frame.width >= 32,
              window.frame.height >= 32 else { continue }
        let application = window.owningApplication
        let windowPid = application?.processID ?? 0
        if let pid, windowPid != pid {
            continue
        }
        rows.append(
            [
                "pid": Int(windowPid),
                "title": window.title ?? "",
                "application": application?.applicationName ?? "",
                "bundle": application?.bundleIdentifier ?? "",
                "x": window.frame.origin.x,
                "y": window.frame.origin.y,
                "width": window.frame.width,
                "height": window.frame.height,
            ]
        )
    }
    let data = try JSONSerialization.data(withJSONObject: rows)
    try FileHandle.standardOutput.write(contentsOf: data)
    try FileHandle.standardOutput.write(contentsOf: Data([0x0A]))
}

private func writeSnapshot(_ snapshot: (rgb: Data, width: UInt32, height: UInt32, pcm: Data, audioActive: Bool)) throws {
    if snapshot.rgb.isEmpty || snapshot.width == 0 || snapshot.height == 0 {
        throw CaptureFailure.format
    }
    var version = UInt32(1).littleEndian
    var width = snapshot.width.littleEndian
    var height = snapshot.height.littleEndian
    var rgbCount = UInt32(snapshot.rgb.count).littleEndian
    var rate = UInt32(48_000).littleEndian
    var channels = UInt16(1).littleEndian
    var sampleWidth = UInt16(2).littleEndian
    var pcmCount = UInt32(snapshot.pcm.count).littleEndian
    var flags = UInt32(snapshot.audioActive ? 1 : 0).littleEndian
    var message = Data()
    message.append(magic)
    message.append(Data(bytes: &version, count: 4))
    message.append(Data(bytes: &width, count: 4))
    message.append(Data(bytes: &height, count: 4))
    message.append(Data(bytes: &rgbCount, count: 4))
    message.append(Data(bytes: &rate, count: 4))
    message.append(Data(bytes: &channels, count: 2))
    message.append(Data(bytes: &sampleWidth, count: 2))
    message.append(Data(bytes: &pcmCount, count: 4))
    message.append(Data(bytes: &flags, count: 4))
    message.append(snapshot.rgb)
    message.append(snapshot.pcm)
    try FileHandle.standardOutput.write(contentsOf: message)
}

@main
struct CaptureMain {
    static func main() async {
        do {
            let arguments = CommandLine.arguments
            if arguments.count >= 2 && arguments[1] == "--query" {
                let pid: pid_t? = arguments.count >= 3 ? Int32(arguments[2]) : nil
                try await listWindows(pid: pid)
                return
            }
            _ = NSApplication.shared
            guard arguments.count == 6, let pid = Int32(arguments[5]), pid > 0 else {
                FileHandle.standardError.write(
                    Data("usage: macos_window_av_capture x y width height pid\n       macos_window_av_capture --query [pid]\n".utf8)
                )
                throw CaptureFailure.arguments
            }
            _ = arguments[1...4]
            let window = try await chooseWindow(pid: pid)
            let session = WindowAVSession()
            try await session.start(window: window)
            let deadline = Date().addingTimeInterval(2.0)
            while Date() < deadline {
                let snapshot = session.snapshot()
                if !snapshot.rgb.isEmpty {
                    break
                }
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            var byte: UInt8 = 0
            while Darwin.read(STDIN_FILENO, &byte, 1) == 1 {
                var snapshot = session.snapshot()
                var waits = 0
                while snapshot.rgb.isEmpty && waits < 50 {
                    try await Task.sleep(nanoseconds: 20_000_000)
                    snapshot = session.snapshot()
                    waits += 1
                }
                try writeSnapshot(snapshot)
            }
            await session.stop()
        } catch {
            FileHandle.standardError.write(Data("macos window av capture failed: \(error)\n".utf8))
            exit(1)
        }
    }
}
