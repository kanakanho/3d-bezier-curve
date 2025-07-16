import Foundation
import Accelerate
import simd


struct BezierPoint: Codable {
    var x: Double
    var y: Double
    
    init(_ point: SIMD2<Double>) {
        self.x = point.x
        self.y = point.y
    }

    var simd: SIMD2<Double> {
        SIMD2(x, y)
    }
}

struct CubicBezierSegment: Codable {
    var p0: BezierPoint
    var p1: BezierPoint
    var p2: BezierPoint
    var p3: BezierPoint

    init(p0: SIMD2<Double>, p1: SIMD2<Double>, p2: SIMD2<Double>, p3: SIMD2<Double>) {
        self.p0 = BezierPoint(p0)
        self.p1 = BezierPoint(p1)
        self.p2 = BezierPoint(p2)
        self.p3 = BezierPoint(p3)
    }

    var controlPointsSIMD: [SIMD2<Double>] {
        [p0.simd, p1.simd, p2.simd, p3.simd]
    }
}

func convertToSegments(curves: [[SIMD2<Double>]]) -> [CubicBezierSegment] {
    curves.map { controlPoints in
        CubicBezierSegment(
            p0: controlPoints[0],
            p1: controlPoints[1],
            p2: controlPoints[2],
            p3: controlPoints[3]
        )
    }
}

func binomial(_ n: Int, _ k: Int) -> Double {
    if k < 0 || k > n { return 0 }
    var result = 1.0
    for i in 1...k {
        result *= Double(n - i + 1) / Double(i)
    }
    return result
}

func bezierGeneral(t: Double, coeffs: [Double]) -> Double {
    let degree = coeffs.count - 1
    var value = 0.0
    for (j, coeff) in coeffs.enumerated() {
        let basis = binomial(degree, j) * pow(1 - t, Double(degree - j)) * pow(t, Double(j))
        value += basis * coeff
    }
    return value
}

func leastSquaresBezier(points: [SIMD2<Double>], degree: Int) -> [Double] {
    let n = points.count
    let m = degree + 1
    var A = [Double](repeating: 0.0, count: n * m)
    var b = [Double](repeating: 0.0, count: n)

    for i in 0..<n {
        let t = Double(i) / Double(n - 1)
        b[i] = points[i].y
        for j in 0...degree {
            A[i * m + j] = binomial(degree, j) * pow(1 - t, Double(degree - j)) * pow(t, Double(j))
        }
    }

    // A^T A
    var ata = [Double](repeating: 0.0, count: m * m)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                Int32(m), Int32(m), Int32(n),
                1.0, A, Int32(m),
                A, Int32(m),
                0.0, &ata, Int32(m))

    // A^T b
    var atb = [Double](repeating: 0.0, count: m)
    cblas_dgemv(CblasRowMajor, CblasTrans,
                Int32(n), Int32(m),
                1.0, A, Int32(m),
                b, 1,
                0.0, &atb, 1)

    // 解く
    var ipiv = [__CLPK_integer](repeating: 0, count: m)
    var info: __CLPK_integer = 0
    var nrhs: __CLPK_integer = 1
    var m_clpk = __CLPK_integer(m)
    var m_clpk1 = m_clpk
    var m_clpk2 = m_clpk
    var m_clpk3 = m_clpk
    var ata_copy = ata
    var atb_copy = atb

    dgesv_(&m_clpk1, &nrhs, &ata_copy, &m_clpk2, &ipiv, &atb_copy, &m_clpk3, &info)

    if info != 0 {
        print("線形方程式の解法に失敗: info = \(info)")
        return []
    }

    return atb_copy
}

func bicScore(rss: Double, n: Int, k: Int) -> Double {
    return Double(n) * log(rss / Double(n)) + Double(k) * log(Double(n))
}

func selectBestBezier(points: [SIMD2<Double>], minDegree: Int = 3, maxDegree: Int = 12) -> (degree: Int, coeffs: [Double])? {
    var bestBIC = Double.infinity
    var bestDegree: Int = -1
    var bestCoeffs: [Double] = []

    let n = points.count
    let yTrue = points.map { $0.y }

    for deg in minDegree...maxDegree {
        let coeffs = leastSquaresBezier(points: points, degree: deg)
        if coeffs.isEmpty { continue }

        let tValues = (0..<n).map { Double($0) / Double(n - 1) }
        let yPred = tValues.map { bezierGeneral(t: $0, coeffs: coeffs) }
        let rss = zip(yTrue, yPred).map { pow($0 - $1, 2) }.reduce(0, +)
        let bic = bicScore(rss: rss, n: n, k: deg + 1)

        if bic < bestBIC {
            bestBIC = bic
            bestDegree = deg
            bestCoeffs = coeffs
        }
    }

    return bestDegree >= 0 ? (bestDegree, bestCoeffs) : nil
}

func cubicBezierPoint(p0: SIMD2<Double>, p1: SIMD2<Double>, p2: SIMD2<Double>, p3: SIMD2<Double>, t: Double) -> SIMD2<Double> {
    let u = 1 - t
    return u * u * u * p0
         + 3 * u * u * t * p1
         + 3 * u * t * t * p2
         + t * t * t * p3
}

func fitCubicBezier(points: [SIMD2<Double>]) -> [SIMD2<Double>] {
    guard let p0 = points.first, let p3 = points.last else { return [] }
    // 初期制御点を p0-p3 の線形補間で配置
    var p1 = p0 + 0.3 * (p3 - p0)
    var p2 = p0 + 0.7 * (p3 - p0)

    let n = points.count
    let tValues = (0..<n).map { Double($0) / Double(n - 1) }

    func error(_ p1: SIMD2<Double>, _ p2: SIMD2<Double>) -> Double {
        let curve = tValues.map { cubicBezierPoint(p0: p0, p1: p1, p2: p2, p3: p3, t: $0) }
        return zip(curve, points).map { simd_length_squared($0 - $1) }.reduce(0, +)
    }

    // 簡単な微調整 (本格的な最適化アルゴリズムの代用)
    var bestP1 = p1, bestP2 = p2
    var minError = error(p1, p2)
    let delta: Double = 0.01

    for dx in -1...1 {
        for dy in -1...1 {
            for dx2 in -1...1 {
                for dy2 in -1...1 {
                    let candidateP1 = p1 + SIMD2<Double>(Double(dx) * delta, Double(dy) * delta)
                    let candidateP2 = p2 + SIMD2<Double>(Double(dx2) * delta, Double(dy2) * delta)
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

    return [p0, bestP1, bestP2, p3]
}

func segmentedCubicBezierFit(points: [SIMD2<Double>], segments: Int) -> [[SIMD2<Double>]] {
    let xs = points.map { $0.x }
    let minX = xs.min()!
    let maxX = xs.max()!
    let splitXs = stride(from: minX, through: maxX, by: (maxX - minX) / Double(segments)).map { $0 }

    // 分割点に一番近い元データ点を取得
    let splitPoints: [SIMD2<Double>] = splitXs.map { x in
        points.min(by: { abs($0.x - x) < abs($1.x - x) })!
    }

    var curves: [[SIMD2<Double>]] = []

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

        var bezier = fitCubicBezier(points: subPoints)

        // 始点・終点を強制的に分割点に合わせる
        bezier[0] = splitPoints[i]
        bezier[3] = splitPoints[i+1]

        curves.append(bezier)
    }

    // C1連続性を保つようにP1/P2を調整
    if curves.count >= 2 {
        for i in 1..<curves.count {
            let prev = curves[i - 1]
            var curr = curves[i]

            let sharedPoint = curr[0]  // = prev[3]
            let v1 = prev[3] - prev[2]
            let v2 = curr[1] - curr[0]
            let avg = 0.5 * (v1 + v2)

            curves[i - 1][2] = sharedPoint - avg
            curves[i][1] = sharedPoint + avg
        }
    }

    return curves
}

// --- セグメント数自動決定ロジック ---
func totalFittingError(curves: [[SIMD2<Double>]], points: [SIMD2<Double>]) -> Double {
    // 各セグメントごとに、元データ点のうちその区間に該当する点との距離の合計を計算
    var errorSum = 0.0
    let n = points.count
    for curve in curves {
        let x0 = curve[0].x
        let x3 = curve[3].x
        let segPoints = points.filter { $0.x >= min(x0, x3) && $0.x <= max(x0, x3) }
        for pt in segPoints {
            // 最近傍tを粗く探索
            var minDist = Double.infinity
            for t in stride(from: 0.0, through: 1.0, by: 0.05) {
                let bez = cubicBezierPoint(p0: curve[0], p1: curve[1], p2: curve[2], p3: curve[3], t: t)
                let dist = simd_distance(bez, pt)
                if dist < minDist { minDist = dist }
            }
            errorSum += minDist * minDist
        }
    }
    return errorSum
}

func findBestSegments(points: [SIMD2<Double>], maxSegments: Int = 10, errorThreshold: Double = 0.01) -> Int {
    for segs in 2...maxSegments {
        let curves = segmentedCubicBezierFit(points: points, segments: segs)
        let err = totalFittingError(curves: curves, points: points)
        if err / Double(points.count) < errorThreshold {
            return segs
        }
    }
    return maxSegments
}

// --- ここから自動決定を利用 ---
let xs = Array(stride(from: -5.0, through: 5.0, by: 0.01))
let input: [SIMD2<Double>] = xs.map { x in
    let y = sin(x) * 2 * (1 - 1 / x)
    return SIMD2(x, y)
}

let bestSegments = findBestSegments(points: input, maxSegments: 10, errorThreshold: 0.0005)
let cubicCurves = segmentedCubicBezierFit(points: input, segments: bestSegments)
print("自動決定されたセグメント数: \(bestSegments)")
for (i, curve) in cubicCurves.enumerated() {
    print("セグメント\(i): \(curve)")
}

// cubicCurves を json形式で ファイルに保存
let segments = convertToSegments(curves: cubicCurves)
let jsonEncoder = JSONEncoder()
jsonEncoder.outputFormatting = [.prettyPrinted]

do {
    let jsonData = try jsonEncoder.encode(segments)
    let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("cubic_segments.json")
    try jsonData.write(to: fileURL)
    print("JSONファイルを保存しました: \(fileURL.path)")
} catch {
    print("JSON保存失敗: \(error)")
}
