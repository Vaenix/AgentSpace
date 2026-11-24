import Foundation
import SoundAnalysis
import AVFoundation

// Input: an uncompressed audio file. Output: start, duration, class, confidence (TSV).
final class Observer: NSObject, SNResultsObserving {
    private(set) var error: Error?

    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let result = result as? SNClassificationResult else { return }
        let start = CMTimeGetSeconds(result.timeRange.start)
        let duration = CMTimeGetSeconds(result.timeRange.duration)
        for label in result.classifications.prefix(3) {
            print(String(format: "%.2f\t%.2f\t%@\t%.4f",
                         start, duration, label.identifier, label.confidence))
        }
    }

    func request(_ request: SNRequest, didFailWithError error: Error) {
        self.error = error
    }
}

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: classify_sound path-to-pcm.wav\n", stderr)
    exit(2)
}
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let analyzer = try SNAudioFileAnalyzer(url: url)
let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
let observer = Observer()
try analyzer.add(request, withObserver: observer)
analyzer.analyze()
if let error = observer.error { throw error }
