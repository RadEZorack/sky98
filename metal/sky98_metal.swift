import Foundation
import Metal

enum Mode {
    case compute
    case verify
}

struct ComputeArgs {
    let outputPath: String
    let seed: UInt64
    let nonce: UInt64
    let matrixSize: Int
    let rounds: Int
}

struct VerifyArgs {
    let tracePath: String
    let seed: UInt64
    let nonce: UInt64
    let checks: Int
    let verifierSecret: UInt64
}

func parseMode() throws -> Mode {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        throw NSError(domain: "sky98-metal", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "usage: sky98-metal-helper <compute|verify> ..."
        ])
    }

    switch args[1] {
    case "compute":
        return .compute
    case "verify":
        return .verify
    default:
        throw NSError(domain: "sky98-metal", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "usage: sky98-metal-helper <compute|verify> ..."
        ])
    }
}

func parseComputeArgs() throws -> ComputeArgs {
    let args = CommandLine.arguments
    guard args.count == 7 else {
        throw NSError(domain: "sky98-metal", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "usage: sky98-metal-helper compute <output-path> <seed> <nonce> <matrix-size> <rounds>"
        ])
    }

    guard
        let seed = UInt64(args[3]),
        let nonce = UInt64(args[4]),
        let matrixSize = Int(args[5]),
        let rounds = Int(args[6])
    else {
        throw NSError(domain: "sky98-metal", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "invalid numeric arguments"
        ])
    }

    return ComputeArgs(
        outputPath: args[2],
        seed: seed,
        nonce: nonce,
        matrixSize: matrixSize,
        rounds: rounds
    )
}

func parseVerifyArgs() throws -> VerifyArgs {
    let args = CommandLine.arguments
    guard args.count == 7 else {
        throw NSError(domain: "sky98-metal", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "usage: sky98-metal-helper verify <trace-path> <seed> <nonce> <checks> <verifier-secret>"
        ])
    }

    guard
        let seed = UInt64(args[3]),
        let nonce = UInt64(args[4]),
        let checks = Int(args[5]),
        let verifierSecret = UInt64(args[6])
    else {
        throw NSError(domain: "sky98-metal", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "invalid numeric arguments"
        ])
    }

    return VerifyArgs(
        tracePath: args[2],
        seed: seed,
        nonce: nonce,
        checks: checks,
        verifierSecret: verifierSecret
    )
}

@inline(__always)
func mix(_ x: UInt64) -> UInt64 {
    var value = x
    value ^= value << 13
    value ^= value >> 7
    value ^= value << 17
    return value
}

@inline(__always)
func rotateLeft(_ x: UInt64, _ count: Int) -> UInt64 {
    let c = UInt64(count & 63)
    return (x << c) | (x >> (64 - c))
}

@inline(__always)
func roundSeed(_ seed: UInt64, _ round: Int) -> UInt64 {
    seed ^ UInt64(round).multipliedReportingOverflow(by: 0x9E3779B97F4A7C15).partialValue
}

@inline(__always)
func roundNonceSeed(_ seed: UInt64, _ round: Int, _ nonce: UInt64) -> UInt64 {
    mix(roundSeed(seed, round) ^ rotateLeft(nonce, round % 63) ^ 0xD6E8FEB86659FD93)
}

func seedToMatrices(seed: UInt64, nonce: UInt64, n: Int) -> ([UInt32], [UInt32]) {
    var a = [UInt32](repeating: 0, count: n * n)
    var b = [UInt32](repeating: 0, count: n * n)
    var state = seed ^ nonce

    for i in 0..<(n * n) {
        state = mix(state)
        a[i] = UInt32(truncatingIfNeeded: state)
        state = mix(state)
        b[i] = UInt32(truncatingIfNeeded: state)
    }

    return (a, b)
}

func permute(_ matrix: [UInt32], n: Int) -> [UInt32] {
    var out = [UInt32](repeating: 0, count: n * n)
    for i in 0..<n {
        for j in 0..<n {
            let srcJ = (j + 1) % n
            out[i * n + j] = matrix[i * n + srcJ]
        }
    }
    return out
}

func writeTrace(
    outputPath: String,
    n: Int,
    rounds: Int,
    a0: [UInt32],
    b0: [UInt32],
    roundOutputs: [[UInt32]],
    roundCommitments: [UInt64]
) throws {
    var data = Data()

    func appendU32(_ value: UInt32) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    func appendU64(_ value: UInt64) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    appendU32(0x5339384D) // "M89S" little-endian tag
    appendU32(UInt32(n))
    appendU32(UInt32(rounds))
    for commitment in roundCommitments {
        appendU64(commitment)
    }

    for value in a0 {
        appendU32(value)
    }
    for value in b0 {
        appendU32(value)
    }
    for round in roundOutputs {
        for value in round {
            appendU32(value)
        }
    }

    try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
}

func readTrace(path: String) throws -> (Int, Int, [UInt32], [UInt32], [[UInt32]]) {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    var offset = 0

    func readU32() throws -> UInt32 {
        guard offset + 4 <= data.count else {
            throw NSError(domain: "sky98-metal", code: 9, userInfo: [
                NSLocalizedDescriptionKey: "unexpected end of trace file"
            ])
        }
        let value = data[offset..<(offset + 4)].withUnsafeBytes {
            $0.load(as: UInt32.self)
        }
        offset += 4
        return UInt32(littleEndian: value)
    }

    let magic = try readU32()
    guard magic == 0x5339384D else {
        throw NSError(domain: "sky98-metal", code: 10, userInfo: [
            NSLocalizedDescriptionKey: "invalid trace magic"
        ])
    }

    let n = Int(try readU32())
    let rounds = Int(try readU32())
    let matrixLen = n * n

    func readMatrix() throws -> [UInt32] {
        var matrix = [UInt32](repeating: 0, count: matrixLen)
        for i in 0..<matrixLen {
            matrix[i] = try readU32()
        }
        return matrix
    }

    let a0 = try readMatrix()
    let b0 = try readMatrix()
    var outputs = [[UInt32]]()
    outputs.reserveCapacity(rounds)

    for _ in 0..<rounds {
        outputs.append(try readMatrix())
    }

    return (n, rounds, a0, b0, outputs)
}

func commitMatrix(_ matrix: [UInt32], n: Int) -> UInt64 {
    var state = 0x6A09E667F3BCC909 ^ UInt64(n)
    for value in matrix {
        state ^= UInt64(value)
        state = mix((state << 9) | (state >> 55) &* 0x9E3779B97F4A7C15)
    }
    return state
}

func deriveChallengeSeed(roundSeed: UInt64, commitment: UInt64, verifierSecret: UInt64) -> UInt64 {
    mix(roundSeed ^ rotateLeft(commitment, 17) ^ verifierSecret)
}

func sampleIndices(n: Int, checks: Int, challengeSeed: UInt64) -> ([UInt32], [UInt32]) {
    var state = challengeSeed
    var rows = [UInt32]()
    var cols = [UInt32]()
    rows.reserveCapacity(checks)
    cols.reserveCapacity(checks)

    for _ in 0..<checks {
        state = mix(state)
        rows.append(UInt32(state % UInt64(n)))
        state = mix(state)
        cols.append(UInt32(state % UInt64(n)))
    }

    return (rows, cols)
}

let shaderSource = #"""
#include <metal_stdlib>
using namespace metal;

inline ulong mix64(ulong x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

inline uint rotl32(uint x, uint count) {
    return (x << count) | (x >> (32u - count));
}

inline uint rotr32(uint x, uint count) {
    return (x >> count) | (x << (32u - count));
}

inline uint sigma32(uint x) {
    uint x1 = x ^ rotl32(x, 13u) ^ 0x9E3779B9u;
    uint x2 = x1 * 0x85EBCA6Bu;
    uint x3 = x2 ^ rotr32(x2, 11u);
    return (x3 * 0xC2B2AE35u) ^ rotr32(x, 7u);
}

kernel void matmul_sigma_mask(
    device const uint *a [[buffer(0)]],
    device const uint *b [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant ulong &roundSeed [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= n || gid.y >= n) {
        return;
    }

    uint row = gid.y;
    uint col = gid.x;
    uint acc = 0;

    for (uint k = 0; k < n; ++k) {
        acc += a[row * n + k] * b[k * n + col];
    }

    ulong idx = ulong(row) * ulong(n) + ulong(col);
    ulong maskState = mix64(roundSeed ^ idx);
    uint value = (maskState & 1ul) ? sigma32(acc) : 0u;
    out[row * n + col] = value;
}

kernel void verify_samples(
    device const uint *a [[buffer(0)]],
    device const uint *b [[buffer(1)]],
    device const uint *c [[buffer(2)]],
    device const uint *sampleRows [[buffer(3)]],
    device const uint *sampleCols [[buffer(4)]],
    device uint *results [[buffer(5)]],
    constant uint &n [[buffer(6)]],
    constant ulong &roundSeed [[buffer(7)]],
    constant uint &checks [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= checks) {
        return;
    }

    uint row = sampleRows[gid];
    uint col = sampleCols[gid];
    uint acc = 0;

    for (uint k = 0; k < n; ++k) {
        acc += a[row * n + k] * b[k * n + col];
    }

    ulong idx = ulong(row) * ulong(n) + ulong(col);
    ulong maskState = mix64(roundSeed ^ idx);
    uint expected = (maskState & 1ul) ? sigma32(acc) : 0u;
    results[gid] = (expected == c[row * n + col]) ? 1u : 0u;
}
"""#

func makeDeviceAndQueue() throws -> (MTLDevice, MTLCommandQueue) {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw NSError(domain: "sky98-metal", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "no Metal device available"
        ])
    }

    guard let queue = device.makeCommandQueue() else {
        throw NSError(domain: "sky98-metal", code: 4, userInfo: [
            NSLocalizedDescriptionKey: "failed to create command queue"
        ])
    }

    return (device, queue)
}

func makeLibraryAndFunctions(device: MTLDevice) throws -> (MTLComputePipelineState, MTLComputePipelineState) {
    let library = try device.makeLibrary(source: shaderSource, options: nil)
    guard
        let computeFunction = library.makeFunction(name: "matmul_sigma_mask"),
        let verifyFunction = library.makeFunction(name: "verify_samples")
    else {
        throw NSError(domain: "sky98-metal", code: 5, userInfo: [
            NSLocalizedDescriptionKey: "failed to load Metal functions"
        ])
    }

    return (
        try device.makeComputePipelineState(function: computeFunction),
        try device.makeComputePipelineState(function: verifyFunction)
    )
}

func runCompute() throws {
    let args = try parseComputeArgs()
    let (device, queue) = try makeDeviceAndQueue()
    let (pipeline, _) = try makeLibraryAndFunctions(device: device)
    let n = args.matrixSize
    let elementBytes = n * n * MemoryLayout<UInt32>.stride

    guard elementBytes <= device.maxBufferLength else {
        throw NSError(domain: "sky98-metal", code: 6, userInfo: [
            NSLocalizedDescriptionKey: "matrix does not fit in a single Metal buffer"
        ])
    }

    let (a0, b0) = seedToMatrices(seed: args.seed, nonce: args.nonce, n: n)
    var a = a0
    var b = b0
    var roundOutputs = [[UInt32]]()
    var roundCommitments = [UInt64]()
    roundOutputs.reserveCapacity(args.rounds)
    roundCommitments.reserveCapacity(args.rounds)

    let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
    let threadsPerGrid = MTLSize(width: n, height: n, depth: 1)

    for round in 0..<args.rounds {
        guard
            let bufferA = device.makeBuffer(bytes: a, length: elementBytes, options: .storageModeShared),
            let bufferB = device.makeBuffer(bytes: b, length: elementBytes, options: .storageModeShared),
            let bufferOut = device.makeBuffer(length: elementBytes, options: .storageModeShared)
        else {
            throw NSError(domain: "sky98-metal", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "failed to allocate Metal buffers"
            ])
        }

        var n32 = UInt32(n)
        var seed64 = roundNonceSeed(args.seed, round, args.nonce)

        guard
            let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw NSError(domain: "sky98-metal", code: 8, userInfo: [
                NSLocalizedDescriptionKey: "failed to create command buffer or encoder"
            ])
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferOut, offset: 0, index: 2)
        encoder.setBytes(&n32, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&seed64, length: MemoryLayout<UInt64>.stride, index: 4)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let output = Array(
            UnsafeBufferPointer(
                start: bufferOut.contents().bindMemory(to: UInt32.self, capacity: n * n),
                count: n * n
            )
        )

        roundCommitments.append(commitMatrix(output, n: n))
        roundOutputs.append(output)
        a = output
        b = permute(output, n: n)
    }

    try writeTrace(
        outputPath: args.outputPath,
        n: n,
        rounds: args.rounds,
        a0: a0,
        b0: b0,
        roundOutputs: roundOutputs,
        roundCommitments: roundCommitments
    )
}

func runVerify() throws {
    let args = try parseVerifyArgs()
    let (device, queue) = try makeDeviceAndQueue()
    let (_, pipeline) = try makeLibraryAndFunctions(device: device)
    let (n, rounds, a0, b0, outputs) = try readTrace(path: args.tracePath)
    let matrixBytes = n * n * MemoryLayout<UInt32>.stride
    let checks = args.checks
    let checksBytes = checks * MemoryLayout<UInt32>.stride

    var a = a0
    var b = b0

    for round in 0..<rounds {
        let c = outputs[round]
        let roundSeedValue = roundNonceSeed(args.seed, round, args.nonce)
        let commitment = commitMatrix(c, n: n)
        let challenge = deriveChallengeSeed(
            roundSeed: roundSeedValue,
            commitment: commitment,
            verifierSecret: args.verifierSecret
        )
        let (rows, cols) = sampleIndices(n: n, checks: checks, challengeSeed: challenge)

        guard
            let bufferA = device.makeBuffer(bytes: a, length: matrixBytes, options: .storageModeShared),
            let bufferB = device.makeBuffer(bytes: b, length: matrixBytes, options: .storageModeShared),
            let bufferC = device.makeBuffer(bytes: c, length: matrixBytes, options: .storageModeShared),
            let rowBuffer = device.makeBuffer(bytes: rows, length: checksBytes, options: .storageModeShared),
            let colBuffer = device.makeBuffer(bytes: cols, length: checksBytes, options: .storageModeShared),
            let resultsBuffer = device.makeBuffer(length: checksBytes, options: .storageModeShared)
        else {
            throw NSError(domain: "sky98-metal", code: 11, userInfo: [
                NSLocalizedDescriptionKey: "failed to allocate verification buffers"
            ])
        }

        var n32 = UInt32(n)
        var roundSeed64 = roundSeedValue
        var checks32 = UInt32(checks)

        guard
            let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw NSError(domain: "sky98-metal", code: 12, userInfo: [
                NSLocalizedDescriptionKey: "failed to create verification command buffer"
            ])
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        encoder.setBuffer(rowBuffer, offset: 0, index: 3)
        encoder.setBuffer(colBuffer, offset: 0, index: 4)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 5)
        encoder.setBytes(&n32, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&roundSeed64, length: MemoryLayout<UInt64>.stride, index: 7)
        encoder.setBytes(&checks32, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: checks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(32, checks), height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let results = Array(
            UnsafeBufferPointer(
                start: resultsBuffer.contents().bindMemory(to: UInt32.self, capacity: checks),
                count: checks
            )
        )

        guard results.allSatisfy({ $0 == 1 }) else {
            throw NSError(domain: "sky98-metal", code: 13, userInfo: [
                NSLocalizedDescriptionKey: "verification failed"
            ])
        }

        a = c
        b = permute(c, n: n)
    }
}

do {
    switch try parseMode() {
    case .compute:
        try runCompute()
    case .verify:
        try runVerify()
    }
} catch {
    fputs("sky98-metal-helper error: \(error)\n", stderr)
    exit(1)
}
