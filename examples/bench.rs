//! CPU-path speed bench, the same source natively and on wasm32-wasip1 (run by tools/wasi-run.mjs under Node's WASI).
//!   bench <mlp dir>   -> one JSON line: student decide p50/p95 (batch 1 and 16) and a 256x256 matmul, in microseconds.
//! Hard cap: a fixed number of iterations (a few seconds). Progress: one line per measurement on stderr.
use realtime_game_decisions::{cpu, Student};
use std::time::Instant;

fn pct(v: &mut Vec<f64>, q: f64) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * q) as usize] }

fn main() {
    let dir = std::env::args().nth(1).expect("usage: bench <mlp dir>");
    let rd = |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
    let s = Student::load(&String::from_utf8(rd("heads.json")).unwrap(), &String::from_utf8(rd("weights.json")).unwrap(), &rd("weights.bin"), &cpu()).unwrap();
    let n = s.n_features();
    let mut out = vec![format!("\"target\":\"{}\"", std::env::consts::ARCH), format!("\"simd128\":{}", cfg!(target_feature = "simd128"))];
    for batch in [1usize, 16] {
        let x: Vec<f32> = (0..n * batch).map(|i| ((i * 7919) % 1000) as f32 / 1000.0).collect();
        for _ in 0..20 { pollster::block_on(s.decide(&x)).unwrap(); }
        let mut t: Vec<f64> = (0..500).map(|_| { let t0 = Instant::now(); pollster::block_on(s.decide(&x)).unwrap(); t0.elapsed().as_secs_f64() * 1e6 }).collect();
        let (p50, p95) = (pct(&mut t, 0.5), pct(&mut t, 0.95));
        eprintln!("decide batch {batch}: p50 {p50:.1} us, p95 {p95:.1} us");
        out.push(format!("\"decide_b{batch}_p50_us\":{p50:.1},\"decide_b{batch}_p95_us\":{p95:.1}"));
    }
    use burn::tensor::{Tensor, TensorData};
    let d = cpu();
    let a = Tensor::<2>::from_data(TensorData::new((0..65536).map(|i| (i % 13) as f32 * 0.1).collect::<Vec<f32>>(), [256, 256]), &d);
    let mut t: Vec<f64> = (0..50).map(|_| { let t0 = Instant::now(); let _ = a.clone().matmul(a.clone()).into_data(); t0.elapsed().as_secs_f64() * 1e6 }).collect();
    let p50 = pct(&mut t, 0.5);
    eprintln!("matmul 256: p50 {p50:.0} us = {:.2} GFLOP/s", 2.0 * 256f64.powi(3) / p50 / 1e3);
    out.push(format!("\"matmul256_p50_us\":{p50:.0},\"matmul256_gflops\":{:.2}", 2.0 * 256f64.powi(3) / p50 / 1e3));
    // posii's MoveNet conv shapes (pose-native fc58bb92 flexcheck): NCHW f32 batch 1, p50 over 30 runs.
    use burn::tensor::{module::conv2d, ops::ConvOptions, Distribution};
    let cases: [(&str, [usize; 4], usize, usize, usize, usize); 5] = [
        ("stem3x3s2_3to48_256", [1, 3, 256, 256], 48, 3, 2, 1), ("dw3x3_144_64", [1, 144, 64, 64], 144, 3, 1, 144),
        ("pw1x1_144to40_64", [1, 144, 64, 64], 40, 1, 1, 1), ("dw3x3_576_16", [1, 576, 16, 16], 576, 3, 1, 576),
        ("pw1x1_96to576_16", [1, 96, 16, 16], 576, 1, 1, 1)];
    for (name, x, cout, k, st, g) in cases {
        let xi = Tensor::<4>::random(x, Distribution::Normal(0.0, 1.0), &d);
        let w = Tensor::<4>::random([cout, x[1] / g, k, k], Distribution::Normal(0.0, 0.1), &d);
        let b = Tensor::<1>::random([cout], Distribution::Normal(0.0, 0.1), &d);
        let o = ConvOptions::new([st, st], [k / 2, k / 2], [1, 1], g);
        let (ho, wo) = ((x[2] + 2 * (k / 2) - k) / st + 1, (x[3] + 2 * (k / 2) - k) / st + 1);
        let flops = 2.0 * (ho * wo * cout * (x[1] / g) * k * k) as f64;
        let mut t: Vec<f64> = (0..33).map(|_| { let t0 = Instant::now(); let _ = conv2d(xi.clone(), w.clone(), Some(b.clone()), o.clone()).into_data(); t0.elapsed().as_secs_f64() * 1e6 }).skip(3).collect();
        let p50 = pct(&mut t, 0.5);
        eprintln!("conv {name}: p50 {:.3} ms = {:.2} GFLOP/s", p50 / 1e3, flops / p50 / 1e3);
        out.push(format!("\"conv_{name}_p50_ms\":{:.3},\"conv_{name}_gflops\":{:.2}", p50 / 1e3, flops / p50 / 1e3));
    }
    // voicii's Whisper shapes (M x K x N, f32): encoder at P=200 (and base at P=750), decoder M=1, vocab head.
    // p50 over up to 40 runs or 0.4 s each; GB/s counts the weight (K x N f32), what bounds M=1 decode.
    let mm: [(&str, usize, usize, usize); 15] = [
        ("enc_tiny_qkvo_P200", 200, 384, 384), ("enc_tiny_up_P200", 200, 384, 1536), ("enc_tiny_down_P200", 200, 1536, 384),
        ("enc_base_qkvo_P200", 200, 512, 512), ("enc_base_up_P200", 200, 512, 2048), ("enc_base_down_P200", 200, 2048, 512),
        ("enc_base_up_P750", 750, 512, 2048),
        ("dec_tiny_qkvo", 1, 384, 384), ("dec_tiny_up", 1, 384, 1536), ("dec_tiny_down", 1, 1536, 384),
        ("dec_base_qkvo", 1, 512, 512), ("dec_base_up", 1, 512, 2048), ("dec_base_down", 1, 2048, 512),
        ("vocab_tiny", 1, 384, 51864), ("vocab_base", 1, 512, 51864)];
    for (name, m, k, n) in mm {
        let xa = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &d);
        let wb = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 0.1), &d);
        let _ = xa.clone().matmul(wb.clone()).into_data();
        let (mut t, t_all) = (Vec::new(), Instant::now());
        while t.len() < 40 && (t.len() < 5 || t_all.elapsed().as_secs_f64() < 0.4) {
            let t0 = Instant::now(); let _ = xa.clone().matmul(wb.clone()).into_data(); t.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let p50 = pct(&mut t, 0.5); let (gf, gbs) = (2.0 * (m * k * n) as f64 / p50 / 1e3, (k * n * 4) as f64 / p50 / 1e3);
        eprintln!("mm {name} {m}x{k}x{n}: p50 {:.3} ms = {gf:.2} GFLOP/s, weight {gbs:.1} GB/s", p50 / 1e3);
        out.push(format!("\"mm_{name}_p50_ms\":{:.3},\"mm_{name}_gflops\":{gf:.2},\"mm_{name}_weight_gbs\":{gbs:.1}", p50 / 1e3));
    }
    // voicii: base encoder attention at P=750, 8 heads x head_dim 64: scores [8,750,64]x[8,64,750], values [8,750,750]x[8,750,64].
    for (name, a, b) in [("attn_base_scores_P750", [8usize, 750, 64], [8usize, 64, 750]), ("attn_base_values_P750", [8, 750, 750], [8, 750, 64])] {
        let xa = Tensor::<3>::random(a, Distribution::Normal(0.0, 1.0), &d);
        let xb = Tensor::<3>::random(b, Distribution::Normal(0.0, 0.1), &d);
        let _ = xa.clone().matmul(xb.clone()).into_data();
        let mut t: Vec<f64> = (0..10).map(|_| { let t0 = Instant::now(); let _ = xa.clone().matmul(xb.clone()).into_data(); t0.elapsed().as_secs_f64() * 1e6 }).collect();
        let p50 = pct(&mut t, 0.5); let gf = 2.0 * (a[0] * a[1] * a[2] * b[2]) as f64 / p50 / 1e3;
        eprintln!("bmm {name}: p50 {:.3} ms = {gf:.2} GFLOP/s", p50 / 1e3);
        out.push(format!("\"bmm_{name}_p50_ms\":{:.3},\"bmm_{name}_gflops\":{gf:.2}", p50 / 1e3));
    }
    // Whisper's encoder front, as conv2d with height 1: 80 -> d k3 over 2P = 400 frames, then d -> d k3 stride 2.
    use burn::tensor::ops::ConvOptions as CO;
    for (name, cin, cout, st) in [("front1_tiny", 80usize, 384usize, 1usize), ("front2_tiny", 384, 384, 2), ("front1_base", 80, 512, 1), ("front2_base", 512, 512, 2)] {
        let xi = Tensor::<4>::random([1, cin, 1, 400], Distribution::Normal(0.0, 1.0), &d);
        let w = Tensor::<4>::random([cout, cin, 1, 3], Distribution::Normal(0.0, 0.1), &d);
        let o = CO::new([1, st], [0, 1], [1, 1], 1);
        let mut t: Vec<f64> = (0..23).map(|_| { let t0 = Instant::now(); let _ = conv2d(xi.clone(), w.clone(), None, o.clone()).into_data(); t0.elapsed().as_secs_f64() * 1e6 }).skip(3).collect();
        let p50 = pct(&mut t, 0.5); let gf = 2.0 * ((400 / st) * cout * cin * 3) as f64 / p50 / 1e3;
        eprintln!("conv1d {name}: p50 {:.3} ms = {gf:.2} GFLOP/s", p50 / 1e3);
        out.push(format!("\"conv1d_{name}_p50_ms\":{:.3},\"conv1d_{name}_gflops\":{gf:.2}", p50 / 1e3));
    }
    println!("{{{}}}", out.join(","));
}
