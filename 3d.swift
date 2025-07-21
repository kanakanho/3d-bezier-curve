import Foundation
import Accelerate
import simd

// MARK: - データ構造

struct BezierPoint3D: Codable {
    var x: Double
    var y: Double
    var z: Double
    
    init(_ point: SIMD3<Double>) {
        self.x = point.x
        self.y = point.y
        self.z = point.z
    }

    var simd: SIMD3<Double> {
        SIMD3(x, y, z)
    }
}

struct CubicBezierSegment3D: Codable {
    var p0: BezierPoint3D
    var p1: BezierPoint3D
    var p2: BezierPoint3D
    var p3: BezierPoint3D

    init(p0: SIMD3<Double>, p1: SIMD3<Double>, p2: SIMD3<Double>, p3: SIMD3<Double>) {
        self.p0 = BezierPoint3D(p0)
        self.p1 = BezierPoint3D(p1)
        self.p2 = BezierPoint3D(p2)
        self.p3 = BezierPoint3D(p3)
    }

    var controlPointsSIMD: [SIMD3<Double>] {
        [p0.simd, p1.simd, p2.simd, p3.simd]
    }
}

func convertToSegments3D(curves: [[SIMD3<Double>]]) -> [CubicBezierSegment3D] {
    curves.map { controlPoints in
        CubicBezierSegment3D(
            p0: controlPoints[0],
            p1: controlPoints[1],
            p2: controlPoints[2],
            p3: controlPoints[3]
        )
    }
}

// MARK: - ベジェ点評価

func cubicBezierPoint3D(p0: SIMD3<Double>, p1: SIMD3<Double>, p2: SIMD3<Double>, p3: SIMD3<Double>, t: Double) -> SIMD3<Double> {
    let u = 1 - t
    return u * u * u * p0
         + 3 * u * u * t * p1
         + 3 * u * t * t * p2
         + t * t * t * p3
}

// MARK: - 曲率推定

func estimateCurvature3D(points: [SIMD3<Double>]) -> [Double] {
    guard points.count >= 3 else { return Array(repeating: 0.0, count: points.count) }
    var curvatures: [Double] = Array(repeating: 0.0, count: points.count)
    for i in 1..<points.count - 1 {
        let t = (points[i + 1] - points[i - 1]) / 2.0
        let n = points[i + 1] - 2.0 * points[i] + points[i - 1]
        curvatures[i] = simd_length(n) / (pow(simd_length(t), 2.0) + 1e-9)
    }
    return curvatures
}

// MARK: - 曲率による分割

func segmentByCurvature(points: [SIMD3<Double>], curvatureThreshold: Double, minSegmentSize: Int = 10) -> [[SIMD3<Double>]] {
    let curvatures = estimateCurvature3D(points: points)
    var segments: [[SIMD3<Double>]] = []
    var current: [SIMD3<Double>] = []

    for i in 0..<points.count {
        current.append(points[i])
        if i > 1 && curvatures[i] > curvatureThreshold && current.count >= minSegmentSize {
            segments.append(current)
            current = [points[i]]
        }
    }
    if !current.isEmpty {
        segments.append(current)
    }
    return segments.filter { $0.count >= 4 }
}

// MARK: - 単一セグメントへのベジェフィット

func fitCubicBezier3D(points: [SIMD3<Double>]) -> [SIMD3<Double>] {
    guard let p0 = points.first, let p3 = points.last else { return [] }
    var p1 = p0 + 0.3 * (p3 - p0)
    var p2 = p0 + 0.7 * (p3 - p0)
    let n = points.count
    let tValues = (0..<n).map { Double($0) / Double(n - 1) }

    func error(_ p1: SIMD3<Double>, _ p2: SIMD3<Double>) -> Double {
        let curve = tValues.map { cubicBezierPoint3D(p0: p0, p1: p1, p2: p2, p3: p3, t: $0) }
        return zip(curve, points).map { simd_length_squared($0 - $1) }.reduce(0, +)
    }

    var bestP1 = p1, bestP2 = p2
    var minError = error(p1, p2)
    let delta: Double = 0.01
    for dx in -1...1 {
        for dy in -1...1 {
            for dz in -1...1 {
                for dx2 in -1...1 {
                    for dy2 in -1...1 {
                        for dz2 in -1...1 {
                            let candidateP1 = p1 + SIMD3<Double>(Double(dx) * delta, Double(dy) * delta, Double(dz) * delta)
                            let candidateP2 = p2 + SIMD3<Double>(Double(dx2) * delta, Double(dy2) * delta, Double(dz2) * delta)
                            let e = error(candidateP1, candidateP2)
                            if e < minError {
                                bestP1 = candidateP1
                                bestP2 = candidateP2
                                minError = e
                            }
                        }
                    }
                }
            }
        }
    }

    return [p0, bestP1, bestP2, p3]
}

// MARK: - 各セグメントにベジェ曲線を適用

func fitSegmentsToBeziers(segments: [[SIMD3<Double>]]) -> [[SIMD3<Double>]] {
    var fitted: [[SIMD3<Double>]] = []
    for segment in segments {
        let bezier = fitCubicBezier3D(points: segment)
        fitted.append(bezier)
    }
    return fitted
}

// MARK: - C¹連続性の調整（任意）

func enforceC1Continuity(curves: inout [[SIMD3<Double>]]) {
    guard curves.count >= 2 else { return }
    for i in 1..<curves.count {
        let prev = curves[i - 1]
        var curr = curves[i]
        let shared = curr[0]
        let v1 = prev[3] - prev[2]
        let v2 = curr[1] - curr[0]
        let avg = 0.5 * (v1 + v2)
        curves[i - 1][2] = shared - avg
        curves[i][1] = shared + avg
    }
}


// --- 3次元用分割フィット ---
func segmentedCubicBezierFit3D(points: [SIMD3<Double>], segments: Int) -> [[SIMD3<Double>]] {
    let xs = points.map { $0.x }
    let minX = xs.min()!
    let maxX = xs.max()!
    let splitXs = stride(from: minX, through: maxX, by: (maxX - minX) / Double(segments)).map { $0 }
    let splitPoints: [SIMD3<Double>] = splitXs.map { x in
        points.min(by: { abs($0.x - x) < abs($1.x - x) })!
    }
    var curves: [[SIMD3<Double>]] = []
    for i in 0..<segments {
        if i+1 >= splitXs.count || i+1 >= splitPoints.count { break }
        let xStart = splitXs[i]
        let xEnd = splitXs[i+1]
        let subPoints = points.filter {
            if i == 0 {
                return xStart <= $0.x && $0.x <= xEnd
            } else {
                return xStart < $0.x && $0.x <= xEnd
            }
        }
        guard subPoints.count >= 4 else { continue }
        var bezier = fitCubicBezier3D(points: subPoints)
        bezier[0] = splitPoints[i]
        bezier[3] = splitPoints[i+1]
        curves.append(bezier)
    }
    if curves.count >= 2 {
        for i in 1..<curves.count {
            let prev = curves[i - 1]
            var curr = curves[i]
            let sharedPoint = curr[0]
            let v1 = prev[3] - prev[2]
            let v2 = curr[1] - curr[0]
            let avg = 0.5 * (v1 + v2)
            curves[i - 1][2] = sharedPoint - avg
            curves[i][1] = sharedPoint + avg
        }
    }
    return curves
}

// --- 3次元用誤差計算 ---
func totalFittingError3D(curves: [[SIMD3<Double>]], points: [SIMD3<Double>]) -> Double {
    var errorSum = 0.0
    let n = points.count
    for curve in curves {
        let x0 = curve[0].x
        let x3 = curve[3].x
        let segPoints = points.filter { $0.x >= min(x0, x3) && $0.x <= max(x0, x3) }
        for pt in segPoints {
            var minDist = Double.infinity
            for t in stride(from: 0.0, through: 1.0, by: 0.05) {
                let bez = cubicBezierPoint3D(p0: curve[0], p1: curve[1], p2: curve[2], p3: curve[3], t: t)
                let dist = simd_distance(bez, pt)
                if dist < minDist { minDist = dist }
            }
            errorSum += minDist * minDist
        }
    }
    return errorSum
}

func findBestSegments3D(points: [SIMD3<Double>], maxSegments: Int, errorThreshold: Double) -> (Int, [[SIMD3<Double>]]) {
    var currentCurves: [[SIMD3<Double>]] = []
    for segs in 2...maxSegments {
        let curves = segmentedCubicBezierFit3D(points: points, segments: segs)
        if segs == maxSegments {
            // 最後のセグメントであれば、現在のカーブを保存
            currentCurves = curves
        }
        let err = totalFittingError3D(curves: curves, points: points)
        if err / Double(points.count) < errorThreshold {
            return (segs, curves)
        }
    }
    // 最適なセグメントが見つからなかった場合、maxSegmentsとnilを返す
    return (maxSegments, currentCurves)
}


// MARK: - メイン処理

let xs3 = Array(stride(from: -2.0, through: 2.0, by: 0.002))
let input3D: [SIMD3<Double>] = xs3.map { x in
    let y = x * x
    // let y = x
    // let y = 0.01 * x * sin(x * 24) * 0.05
    // let z = x
    // let z = x * x * x
    let z = sin(x * 15) * 0.01 + x * 0.02 * cos(x * 24)  // 例: xに対してzをsin-cos関数で変化させる
    return SIMD3(x, y, z)
}

// original の 保存
let originalPoints3D: [BezierPoint3D] = input3D.map { BezierPoint3D($0) }
let jsonEncoder3DOriginal = JSONEncoder()
jsonEncoder3DOriginal.outputFormatting = [.prettyPrinted]
do {
    let jsonDataOriginal = try jsonEncoder3DOriginal.encode(originalPoints3D)
    let fileURLOriginal = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("original_points_3d.json")
    try jsonDataOriginal.write(to: fileURLOriginal)
    print("[3D] 元のデータをJSONファイルに保存しました: \(fileURLOriginal.path)")
} catch {
    print("[3D] 元のデータのJSON保存失敗: \(error)")
}


// 曲率しきい値を設定して分割
let curvatureThreshold = 1.4
let segmented:[[SIMD3<Double>]] = segmentByCurvature(points: input3D, curvatureThreshold: curvatureThreshold)
print("[3D] 曲率ベースで分割されたセグメント数: \(segmented.count)")
for (i, segment) in segmented.enumerated() {
    print("[3D] セグメント \(i): \(segment.count) 点")
}

// let segmented = [input3D]

// 各セグメントにベジェ曲線をフィット
var fittedCurves = fitSegmentsToBeziers(segments: segmented)

var cubicCurves3Ds: [[SIMD3<Double>]] = []
for points in segmented {
    let (bestSegments3D, cubicCurves3D) = findBestSegments3D(points: points, maxSegments: 100, errorThreshold: 0.001)
    cubicCurves3Ds.append(contentsOf: cubicCurves3D)
    print("[3D] 最適なセグメント数: \(bestSegments3D), 曲線数: \(cubicCurves3Ds.count)")
}

// C¹連続性を確保（任意）
// enforceC1Continuity(curves: &fittedCurves)

// ベジェ曲線セグメントを保存用に変換
// let segments3D = convertToSegments3D(curves: fittedCurves)
let segments3D = convertToSegments3D(curves: cubicCurves3Ds)

// JSON 保存
let jsonEncoder = JSONEncoder()
jsonEncoder.outputFormatting = [.prettyPrinted]

do {
    let json = try jsonEncoder.encode(segments3D)
    let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("cubic_segments_3d_curvature.json")
    try json.write(to: fileURL)
    print("[3D] 曲率ベース分割のベジェ曲線をJSONで保存しました: \(fileURL.path)")
} catch {
    print("[3D] JSON保存失敗: \(error)")
}
