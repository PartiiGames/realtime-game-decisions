//! Tests with hard caps: each is a few ms of CPU; the farm job's timeout is the outer cap.
use realtime_game_decisions::{cpu, Student};

fn tiny() -> (String, String, Vec<u8>) {
    // 3 features -> 4 -> 4 -> 5 logits (heads: a 2-way noul, a 3-way choice); weights 0.01 * k, deterministic.
    let heads = r#"{"heads":[{"id":"shoot","type":"noul","keys":["false","true"]},{"id":"move","type":"choice","keys":["n","e","stay"]}],
                   "offsets":[0,2,5],"temperatures":[1.0,2.0]}"#.to_string();
    let shapes: [(&str, Vec<usize>); 6] = [("net.0.weight", vec![4, 3]), ("net.0.bias", vec![4]), ("net.2.weight", vec![4, 4]),
                                           ("net.2.bias", vec![4]), ("net.4.weight", vec![5, 4]), ("net.4.bias", vec![5])];
    let (mut idx, mut bin, mut off) = (serde_json::Map::new(), Vec::new(), 0usize);
    for (name, shape) in shapes.iter() {
        let n: usize = shape.iter().product();
        for k in 0..n { bin.extend_from_slice(&(((k as f32) - (n as f32) / 2.0) * 0.05).to_le_bytes()); }
        idx.insert(name.to_string(), serde_json::json!({"shape": shape, "offset": off}));
        off += n;
    }
    (heads, serde_json::Value::Object(idx).to_string(), bin)
}

#[test]
fn answers_have_the_reply_shape() {
    let (h, w, b) = tiny();
    let s = Student::load(&h, &w, &b, &cpu()).unwrap();
    assert_eq!(s.n_features(), 3);
    let out = pollster::block_on(s.decide(&[0.1, -0.2, 0.3, 1.0, 0.0, -1.0])).unwrap();
    assert_eq!(out.len(), 2);
    for row in &out {
        assert_eq!(row.len(), 2);
        for a in row {
            let sum: f32 = a.probabilities.iter().map(|p| p.1).sum();
            assert!((sum - 1.0).abs() < 1e-5, "{sum}");
            assert!(a.probabilities.iter().any(|p| p.0 == a.choice));
        }
    }
    assert_eq!(s.stats.readbacks.load(std::sync::atomic::Ordering::Relaxed), 1, "one readback for the whole batch");
    assert_eq!(s.stats.dispatches.load(std::sync::atomic::Ordering::Relaxed), 0, "CPU path: no GPU dispatches");
}

#[test]
fn bad_input_is_refused() {
    let (h, w, b) = tiny();
    let s = Student::load(&h, &w, &b, &cpu()).unwrap();
    assert!(pollster::block_on(s.decide(&[0.1, 0.2])).is_err());
    assert!(Student::load(&h, &w, &b[..b.len() - 4], &cpu()).is_err());
}

/// The real Arena student against PyTorch: testdata/arena-mlp (distill.py's models/mlp: 93.5% acc, 27.8k params,
/// fixture = 16 rows of features + PyTorch's calibrated probabilities), or DECIDE_MLP_DIR for a fresh one.
#[test]
fn matches_pytorch_fixture() {
    let dir = mlp_dir();
    let rd = |f: &str| std::fs::read_to_string(format!("{dir}/{f}")).unwrap();
    let bin = std::fs::read(format!("{dir}/weights.bin")).unwrap();
    let s = Student::load(&rd("heads.json"), &rd("weights.json"), &bin, &cpu()).unwrap();
    let fx: serde_json::Value = serde_json::from_str(&rd("fixture.json")).unwrap();
    let feats: Vec<f32> = fx["features"].as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32)).collect();
    let out = pollster::block_on(s.decide(&feats)).unwrap();
    let mut worst = 0f32;
    for (r, row) in out.iter().enumerate() {
        for (i, a) in row.iter().enumerate() {
            for (j, p) in a.probabilities.iter().enumerate() {
                let want = fx["probs"][r][i][j].as_f64().unwrap() as f32;
                worst = worst.max((p.1 - want).abs());
            }
        }
    }
    println!("FIXTURE rows={} max_abs_diff={worst:.2e}", out.len());
    assert!(worst < 1e-4, "max abs diff {worst}");
}

/// Latency of one state per call on the CPU path (native here; the WASM numbers come from the browser tests).
#[test]
fn latency_one_state() {
    let dir = mlp_dir();
    let rd = |f: &str| std::fs::read_to_string(format!("{dir}/{f}")).unwrap();
    let bin = std::fs::read(format!("{dir}/weights.bin")).unwrap();
    let s = Student::load(&rd("heads.json"), &rd("weights.json"), &bin, &cpu()).unwrap();
    let x = vec![0.1f32; s.n_features()];
    for _ in 0..50 { pollster::block_on(s.decide(&x)).unwrap(); }
    let mut t: Vec<f64> = (0..1000).map(|_| { let t0 = std::time::Instant::now(); pollster::block_on(s.decide(&x)).unwrap(); t0.elapsed().as_secs_f64() * 1e6 }).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("LATENCY native cpu 1 state: p50 {:.1} us, p95 {:.1} us", t[500], t[950]);
}

fn mlp_dir() -> String {
    std::env::var("DECIDE_MLP_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/arena-mlp").to_string())
}
