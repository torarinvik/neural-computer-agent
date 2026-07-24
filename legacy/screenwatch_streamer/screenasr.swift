// screenasr — system-audio speech transcription for the screen orchestra (SPEC.md).
//
// A symbolic member (like screenocr): captures SYSTEM AUDIO via ScreenCaptureKit (the same
// capturesAudio path the recorder uses; no microphone) and streams it through the on-device
// SpeechAnalyzer/SpeechTranscriber (macOS 26), appending finalized utterances to an append-only
// log the watcher can read by time window:
//
//   ASR u0=<ms> u1=<ms> <utterance text>
//   ASRW <ms>:<word> <ms>:<word> ...          (per-word start times, same utterance)
//
// Timebase: u0/u1 are raw CLOCK_MONOTONIC milliseconds — the same clock frame_dump's
// screencap_now_ms() uses. The batch header carries `mono0=<recorder start>` so a reader aligns:
// batch-relative t = u0 - mono0.
//
// Build:  swiftc -O screenasr.swift -o screenasr
// Usage:  screenasr [out_file=/tmp/screen_batches/asr.log] [locale=en_US]
//
// Needs the Screen Recording TCC permission (ScreenCaptureKit prompts on first run). The speech
// model for the locale downloads on first use (AssetInventory).

import AVFoundation
import Foundation
import ScreenCaptureKit
import Speech

func monoNowMs() -> Int64 {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Int64(ts.tv_sec) * 1000 + Int64(ts.tv_nsec) / 1_000_000
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("screenasr: " + msg + "\n").data(using: .utf8)!)
    exit(1)
}

// Append-only, single-writer log (blackboard rule: every write is a single append).
final class AsrLog {
    private let handle: FileHandle
    init(path: String) {
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let h = FileHandle(forWritingAtPath: path) else { fail("cannot open \(path)") }
        h.seekToEndOfFile()
        handle = h
    }
    func append(_ line: String) {
        handle.write((line + "\n").data(using: .utf8)!)
    }
}

// Bridges SCStream audio buffers into an AsyncStream of AnalyzerInput. The first buffer anchors
// the stream's CMTime timeline to CLOCK_MONOTONIC; every buffer carries its stream-relative
// start time so transcription ranges stay exact even if a buffer is dropped.
final class AudioTap: NSObject, SCStreamOutput, SCStreamDelegate {
    let continuation: AsyncStream<AnalyzerInput>.Continuation
    private(set) var anchorMonoMs: Int64 = 0        // monotonic ms at stream time zero
    private var firstPTS: CMTime? = nil

    init(continuation: AsyncStream<AnalyzerInput>.Continuation) {
        self.continuation = continuation
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        guard let fmtDesc = sampleBuffer.formatDescription else { return }
        let format = AVAudioFormat(cmAudioFormatDescription: fmtDesc)
        let pts = sampleBuffer.presentationTimeStamp
        if firstPTS == nil {
            firstPTS = pts
            anchorMonoMs = monoNowMs() // buffer just arrived: "now" ~= its capture end; close enough at 10ms buffers
        }
        let rel = CMTimeSubtract(pts, firstPTS!)
        var pcm: AVAudioPCMBuffer? = nil
        try? sampleBuffer.withAudioBufferList { audioBufferList, _ in
            pcm = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: audioBufferList.unsafePointer)
        }
        guard let buffer = pcm else { return }
        continuation.yield(AnalyzerInput(buffer: buffer, bufferStartTime: rel))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("screenasr: stream stopped: \(error)\n".data(using: .utf8)!)
        continuation.finish()
    }
}

func run() async throws {
    let args = CommandLine.arguments
    let outPath = args.count > 1 ? args[1] : "/tmp/screen_batches/asr.log"
    let localeID = args.count > 2 ? args[2] : "en_US"
    let locale = Locale(identifier: localeID)

    guard SpeechTranscriber.isAvailable else { fail("SpeechTranscriber is not available on this system") }
    guard let supported = await SpeechTranscriber.supportedLocale(equivalentTo: locale) else {
        fail("locale \(localeID) is not supported for transcription")
    }

    let transcriber = SpeechTranscriber(
        locale: supported,
        transcriptionOptions: [],
        reportingOptions: [],                    // finalized results only — the log is append-only
        attributeOptions: [.audioTimeRange])     // per-word timing runs

    // Ensure the on-device model for this locale is installed (downloads on first use).
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        FileHandle.standardError.write("screenasr: downloading speech model for \(supported.identifier)...\n".data(using: .utf8)!)
        try await request.downloadAndInstall()
    }

    let (inputSequence, inputBuilder) = AsyncStream<AnalyzerInput>.makeStream()
    let analyzer = SpeechAnalyzer(modules: [transcriber])

    // System-audio-only SCStream (a display filter is still required; video output is not added).
    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
    guard let display = content.displays.first else { fail("no display found (Screen Recording permission?)") }
    let filter = SCContentFilter(display: display, excludingWindows: [])
    let config = SCStreamConfiguration()
    config.capturesAudio = true
    config.excludesCurrentProcessAudio = true
    config.width = 2   // minimal video plane; no video output is attached
    config.height = 2

    let tap = AudioTap(continuation: inputBuilder)
    let stream = SCStream(filter: filter, configuration: config, delegate: tap)
    try stream.addStreamOutput(tap, type: .audio, sampleHandlerQueue: DispatchQueue(label: "screenasr.audio"))
    try await stream.startCapture()

    let log = AsrLog(path: outPath)
    FileHandle.standardError.write("screenasr: transcribing system audio -> \(outPath) (locale \(supported.identifier))\n".data(using: .utf8)!)

    try await analyzer.start(inputSequence: inputSequence)

    for try await result in transcriber.results {
        let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { continue }
        let anchor = tap.anchorMonoMs
        let u0 = anchor + Int64(result.range.start.seconds * 1000)
        let u1 = anchor + Int64(result.range.end.seconds * 1000)
        log.append("ASR u0=\(u0) u1=\(u1) \(text)")
        // Per-word start times from the audioTimeRange attribute runs.
        var words: [String] = []
        for runElement in result.text.runs {
            guard let timeRange = runElement.audioTimeRange else { continue }
            let word = String(result.text[runElement.range].characters).trimmingCharacters(in: .whitespacesAndNewlines)
            if word.isEmpty { continue }
            words.append("\(anchor + Int64(timeRange.start.seconds * 1000)):\(word)")
        }
        if !words.isEmpty {
            log.append("ASRW " + words.joined(separator: " "))
        }
    }
}

signal(SIGINT) { _ in exit(0) }
Task {
    do { try await run() } catch { fail("\(error)") }
}
dispatchMain()
