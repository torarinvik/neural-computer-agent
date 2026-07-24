// audiocap.swift — system-audio capture into a lossless PCM ring, the audio twin of the frame_dump
// video path (orchestra_v2 plan M4). Taps system audio via ScreenCaptureKit (capturesAudio, no
// microphone — same permission the recorder already holds), resamples to 16 kHz mono s16le, and
// appends to <dir>/aud_<epoch>.pcm with a checkpoint index <dir>/aud_<epoch>.idx. Timestamps come
// from CLOCK_MONOTONIC (the same clock frame_dump stamps), so audio<->video correlation is a
// subtraction, not a calibration project. The cymbal (audiotriage) and sax (screenaud) read this ring.
//
// USAGE: audiocap [out_dir]        (default /tmp/screen_batches)   Ctrl-C to stop.
// Ring format:
//   aud_<epoch>.pcm   raw s16le, 16 kHz mono, continuous
//   aud_<epoch>.idx   text checkpoints "<t_ms> <sample_offset>" ~2/s (map a time span to bytes)
//
// LIVE-VERIFICATION PENDING: this compiles and follows the proven screenasr.swift capture pattern,
// but a real capture run needs Screen-Recording permission + live audio (the same gate as the M4
// co-run). The deterministic direct-PCM path (audiogen WAV -> audiotriage/screenaud) is what the
// trap suites exercise; this is the production source that feeds the same detectors live.

import AVFoundation
import Foundation
import ScreenCaptureKit

func monoNowMs() -> Int64 {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Int64(ts.tv_sec) * 1000 + Int64(ts.tv_nsec) / 1_000_000
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write("audiocap: \(msg)\n".data(using: .utf8)!)
    exit(1)
}

let SR: Double = 16000

// Appends resampled s16 samples to the .pcm file and writes periodic index checkpoints.
final class PcmRing {
    private let pcm: FileHandle
    private let idx: FileHandle
    private var sampleOffset: Int64 = 0
    private var lastIdxSample: Int64 = -100000
    let anchorMonoMs: Int64

    init(dir: String, anchorMonoMs: Int64) {
        self.anchorMonoMs = anchorMonoMs
        let epoch = anchorMonoMs
        let pcmPath = "\(dir)/aud_\(epoch).pcm"
        let idxPath = "\(dir)/aud_\(epoch).idx"
        FileManager.default.createFile(atPath: pcmPath, contents: nil)
        FileManager.default.createFile(atPath: idxPath, contents: nil)
        guard let p = FileHandle(forWritingAtPath: pcmPath),
              let i = FileHandle(forWritingAtPath: idxPath) else {
            fail("cannot open ring files in \(dir)")
        }
        pcm = p
        idx = i
    }

    // append n s16 samples; write a "<t_ms> <sample_offset>" checkpoint about twice a second
    func append(_ samples: [Int16]) {
        if samples.isEmpty { return }
        samples.withUnsafeBufferPointer { buf in
            pcm.write(Data(buffer: buf))
        }
        if sampleOffset - lastIdxSample >= Int64(SR / 2) {
            let tMs = anchorMonoMs + Int64(Double(sampleOffset) / SR * 1000)
            idx.write("\(tMs) \(sampleOffset)\n".data(using: .utf8)!)
            lastIdxSample = sampleOffset
        }
        sampleOffset += Int64(samples.count)
    }
}

// Bridges SCStream audio buffers -> 16 kHz mono s16 -> the ring.
final class AudioSink: NSObject, SCStreamOutput, SCStreamDelegate {
    let ring: PcmRing
    private var converter: AVAudioConverter? = nil
    private let outFormat: AVAudioFormat

    init(ring: PcmRing) {
        self.ring = ring
        self.outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: SR,
                                       channels: 1, interleaved: true)!
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        guard let fmtDesc = sampleBuffer.formatDescription else { return }
        let inFormat = AVAudioFormat(cmAudioFormatDescription: fmtDesc)
        var inPcm: AVAudioPCMBuffer? = nil
        try? sampleBuffer.withAudioBufferList { abl, _ in
            inPcm = AVAudioPCMBuffer(pcmFormat: inFormat, bufferListNoCopy: abl.unsafePointer)
        }
        guard let input = inPcm else { return }

        if converter == nil {
            converter = AVAudioConverter(from: inFormat, to: outFormat)
        }
        guard let conv = converter else { return }

        let ratio = SR / inFormat.sampleRate
        let outCap = AVAudioFrameCount(Double(input.frameLength) * ratio) + 16
        guard let output = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCap) else { return }

        var fed = false
        var err: NSError? = nil
        conv.convert(to: output, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return input
        }
        if err != nil { return }

        let n = Int(output.frameLength)
        guard n > 0, let ch = output.int16ChannelData else { return }
        var samples = [Int16](repeating: 0, count: n)
        let p = ch[0]
        for i in 0..<n { samples[i] = p[i] }
        ring.append(samples)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("audiocap: stream stopped: \(error)\n".data(using: .utf8)!)
    }
}

func run() async throws {
    let args = CommandLine.arguments
    let outDir = args.count > 1 ? args[1] : "/tmp/screen_batches"
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
    guard let display = content.displays.first else { fail("no display (Screen Recording permission?)") }
    let filter = SCContentFilter(display: display, excludingWindows: [])
    let config = SCStreamConfiguration()
    config.capturesAudio = true
    config.excludesCurrentProcessAudio = true
    config.sampleRate = 48000
    config.channelCount = 2
    config.width = 2                 // minimal video plane; no video output attached
    config.height = 2

    let ring = PcmRing(dir: outDir, anchorMonoMs: monoNowMs())
    let sink = AudioSink(ring: ring)
    let stream = SCStream(filter: filter, configuration: config, delegate: sink)
    try stream.addStreamOutput(sink, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audiocap.audio"))
    try await stream.startCapture()
    FileHandle.standardError.write("audiocap: capturing system audio -> \(outDir)/aud_*.pcm (16kHz mono)\n".data(using: .utf8)!)
}

signal(SIGINT) { _ in exit(0) }
Task {
    do { try await run() } catch { fail("\(error)") }
}
dispatchMain()
