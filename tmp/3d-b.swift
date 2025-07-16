import Foundation
import Accelerate
import simd

// --- 3次元対応: 型定義 (変更なし) ---
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

// --- 3次元用ベジェ点計算 (変更なし) ---
func cubicBezierPoint3D(p0: SIMD3<Double>, p1: SIMD3<Double>, p2: SIMD3<Double>, p3: SIMD3<Double>, t: Double) -> SIMD3<Double> {
    let u = 1 - t
    return u * u * u * p0
         + 3 * u * u * t * p1
         + 3 * u * t * t * p2
         + t * t * t * p3
}

// --- 3次元用フィッティング (変更なし) ---
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

func fitCubicBezier3D(points: [SIMD3<Double>], start: SIMD3<Double>, end: SIMD3<Double>) -> [SIMD3<Double>] {
    let n = points.count
    guard n >= 4 else {
        return [start, start, end, end]
    }

    let ts = (0..<n).map { Double($0) / Double(n - 1) }
    let P0 = start
    let P3 = end

    var A1 = SIMD3<Double>(repeating: 0)
    var A2 = SIMD3<Double>(repeating: 0)
    var b = SIMD3<Double>(repeating: 0)
    var c = SIMD3<Double>(repeating: 0)

    for i in 0..<n {
        let t = ts[i]
        let b1 = 3 * pow(1 - t, 2) * t
        let b2 = 3 * (1 - t) * pow(t, 2)
        let rhs = points[i] - pow(1 - t, 3) * P0 - pow(t, 3) * P3
        A1 += b1 * b1
        A2 += b1 * b2
        b += b1 * rhs
        c += b2 * rhs
    }

    let denom = A1 * A1 - A2 * A2
    var alpha = SIMD3<Double>(repeating: 0)
    var beta = SIMD3<Double>(repeating: 0)
    for i in 0..<3 {
        if abs(denom[i]) > 1e-8 {
            alpha[i] = (A1[i] * b[i] - A2[i] * c[i]) / denom[i]
            beta[i]  = (c[i] - A2[i] * alpha[i]) / A1[i]
        } else {
            alpha[i] = 0
            beta[i] = 0
        }
    }

    let P1 = P0 + alpha
    let P2 = P3 - beta

    return [P0, P1, P2, P3]
}


// --- 新しい3次元用分割フィット (曲率ベース) ---
func segmentedCubicBezierFit3DCurvature(points: [SIMD3<Double>], curvatureThreshold: Double, minSegmentPoints: Int) -> [[SIMD3<Double>]] {
    guard points.count >= minSegmentPoints else {
        if points.count >= 4 {
            return [fitCubicBezier3D(points: points, start: points.first!, end: points.last!)]
        }
        return []
    }

    var curves: [[SIMD3<Double>]] = []
    var currentSegmentPoints: [SIMD3<Double>] = [points[0]]

    func angleBetweenVectors(p1: SIMD3<Double>, p2: SIMD3<Double>, p3: SIMD3<Double>) -> Double {
        let v1 = p2 - p1
        let v2 = p3 - p2
        let dotProduct = simd_dot(simd_normalize(v1), simd_normalize(v2))
        return acos(max(-1.0, min(1.0, dotProduct)))
    }

    for i in 1..<points.count {
        currentSegmentPoints.append(points[i])
        
        if currentSegmentPoints.count >= 3 {
            let p_prev = currentSegmentPoints[currentSegmentPoints.count - 3]
            let p_curr = currentSegmentPoints[currentSegmentPoints.count - 2]
            let p_next = currentSegmentPoints[currentSegmentPoints.count - 1]
            
            let angle = angleBetweenVectors(p1: p_prev, p2: p_curr, p3: p_next)
            
            if angle > curvatureThreshold && currentSegmentPoints.count - 1 >= minSegmentPoints {
                let (_, cubicCurves3D) = findBestSegments3D(points: currentSegmentPoints, maxSegments: 4, errorThreshold: 0.01)
                curves.append(contentsOf: cubicCurves3D)
            }
        }
    }

    if currentSegmentPoints.count >= 1 {
        let start = currentSegmentPoints.first!
        let end = currentSegmentPoints.last!
        curves.append(fitCubicBezier3D(points: [start, end], start: start, end: end))
    }

    // C1連続性調整（変えずに使用）
    if curves.count >= 2 {
        for i in 1..<curves.count {
            let prev = curves[i - 1]
            var curr = curves[i]
            let sharedPoint = curr[0]

            if prev[3] != sharedPoint {
                curves[i - 1][3] = sharedPoint
            }

            let v1 = prev[3] - prev[2]
            let v2 = curr[1] - curr[0]
            let avg = 0.5 * (v1 + v2)
            curves[i - 1][2] = sharedPoint - avg
            curves[i][1] = sharedPoint + avg
        }
    }

    return curves
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


// --- 3次元用誤差計算 (変更なし) ---
func totalFittingError3D(curves: [[SIMD3<Double>]], points: [SIMD3<Double>]) -> Double {
    var errorSum = 0.0
    // 全てのポイントが何らかの曲線にフィットしていることを前提とする
    // この誤差計算は、元のポイントとフィッティングされた曲線との距離を測る
    for pt in points {
        var minDistSquared = Double.infinity
        var foundCurveForPoint = false

        for curve in curves {
            // 各曲線セグメントがカバーするX軸の範囲を考慮する（必要であれば）
            // 現状の実装では、全ての曲線を調べて最も近い距離を見つける
            for t in stride(from: 0.0, through: 1.0, by: 0.05) { // 細かくサンプリング
                let bez = cubicBezierPoint3D(p0: curve[0], p1: curve[1], p2: curve[2], p3: curve[3], t: t)
                let distSquared = simd_length_squared(bez - pt)
                if distSquared < minDistSquared {
                    minDistSquared = distSquared
                }
            }
            foundCurveForPoint = true // ポイントが少なくとも1つの曲線に近づけられた
        }
        if foundCurveForPoint {
            errorSum += minDistSquared
        }
    }
    return errorSum
}

// --- 3次元サンプルデータ・保存例 ---
// 例: 3次元のsinカーブ
let xs3 = Array(stride(from: -2.0, through: 2.0, by: 0.001))
let input3D: [SIMD3<Double>] = xs3.map { x in
    // let y = 0.01 * x * sin(x * 24) * 0.05
    let y = x * x * x
    let z = sin(x * 15) * 0.01 + x * 0.02 * cos(x * 24)
    // let z = x * x
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

// 曲率ベースの分割を実行
// curvatureThreshold: 曲率変化の閾値 (ラジアン単位, 0からπ)。値を小さくすると分割が増える。
// minSegmentPoints: 1つのセグメントに含める最小の点数。少なすぎると不安定になる可能性がある。
let cubicCurves3D = segmentedCubicBezierFit3DCurvature(points: input3D, curvatureThreshold: 0.005, minSegmentPoints: 4)
print("[3D] 動的に決定されたセグメント数: \(cubicCurves3D.count)")

// for (i, curve) in cubicCurves3D.enumerated() {
//     print("[3D] セグメント\(i): \(curve)")
// }

let segments3D = convertToSegments3D(curves: cubicCurves3D)
let jsonEncoder3D = JSONEncoder()
jsonEncoder3D.outputFormatting = [.prettyPrinted]
do {
    let jsonData = try jsonEncoder3D.encode(segments3D)
    let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("cubic_segments_3d.json")
    try jsonData.write(to: fileURL)
    print("[3D] JSONファイルを保存しました: \(fileURL.path)")
} catch {
    print("[3D] JSON保存失敗: \(error)")
}

// 全体のフィッティング誤差を計算 (オプション)
let totalError = totalFittingError3D(curves: cubicCurves3D, points: input3D)
print("[3D] 全体の平均二乗誤差: \(totalError / Double(input3D.count))")