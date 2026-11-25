//! realtime-game-decisions: tiny, fast decision models for games.
//!
//! A game asks questions about its current state (which way to move? fire now? how dangerous is it?) and gets an
//! answer per question with calibrated probabilities. The model here is a per-game **student**: the state as a vector
//! of numbers -> a 2x128 GELU MLP -> one block of logits per question, softmaxed with one temperature per question.
//! It runs on burn's pure-Rust CPU backend (flex), natively or in the browser (wasm32), in microseconds per state.
//!
//! - **async only**: [`Student::decide`] never blocks, so it also works inside a browser's event loop.
//! - **one readback per call**: all heads of all rows come back in one tensor read.
//! - **counters**: [`Stats`] counts calls, readbacks and GPU dispatches (always 0 on the CPU path).
//!
//! The model files are three: `heads.json` (questions, their answer keys, logit offsets and temperatures),
//! `weights.json` (an index of tensors into the blob) and `weights.bin` (little-endian f32). See
//! `examples/decide.rs` and `testdata/arena-mlp/`.

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod web;

use burn::tensor::{activation, Device, Tensor, TensorData};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// The CPU device every target has (burn 0.22's flex: a pure-Rust CPU backend).
pub fn cpu() -> Device { Device::flex() }

#[derive(Debug, Clone, Deserialize)]
pub struct Head {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub keys: Vec<String>,
}

/// heads.json: the questions (`heads`), the logit range of each (`offsets`, one longer than `heads`), and one
/// calibration temperature per question.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadsFile {
    pub heads: Vec<Head>,
    pub offsets: Vec<usize>,
    pub temperatures: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct WeightEntry {
    shape: Vec<usize>,
    offset: usize,
}

/// One question's answer: the chosen key, its probability, and every key's probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub id: String,
    pub kind: String,
    pub choice: String,
    pub confidence: f32,
    pub probabilities: Vec<(String, f32)>,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub calls: AtomicU64,
    pub readbacks: AtomicU64,
    pub dispatches: AtomicU64,
}

#[derive(Debug)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for Error {}
fn err(m: impl Into<String>) -> Error { Error(m.into()) }

struct Layer {
    w_t: Tensor<2>, // [in, out] (torch stores [out, in]; transposed once at load)
    b: Tensor<2>,   // [1, out]
}

pub struct Student {
    layers: Vec<Layer>,
    n_in: usize,
    heads: HeadsFile,
    device: Device,
    pub stats: Stats,
}

impl Student {
    /// Loads a student from its three files: heads.json, weights.json (index) and weights.bin (little-endian f32).
    pub fn load(heads_json: &str, weights_json: &str, weights_bin: &[u8], device: &Device) -> Result<Self, Error> {
        let heads: HeadsFile = serde_json::from_str(heads_json).map_err(|e| err(format!("heads.json: {e}")))?;
        if heads.offsets.len() != heads.heads.len() + 1 || heads.temperatures.len() != heads.heads.len() {
            return Err(err("heads.json: offsets/temperatures don't match heads"));
        }
        let index: HashMap<String, WeightEntry> = serde_json::from_str(weights_json).map_err(|e| err(format!("weights.json: {e}")))?;
        if weights_bin.len() % 4 != 0 { return Err(err("weights.bin: not whole f32s")); }
        let all: Vec<f32> = weights_bin.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let take = |name: &str| -> Result<(Vec<usize>, Vec<f32>), Error> {
            let e = index.get(name).ok_or_else(|| err(format!("weights: {name} missing")))?;
            let n: usize = e.shape.iter().product();
            let v = all.get(e.offset..e.offset + n).ok_or_else(|| err(format!("weights: {name} out of range")))?;
            Ok((e.shape.clone(), v.to_vec()))
        };
        // nn.Sequential(Linear, GELU, Linear, GELU, Linear): Linear layers at net.0, net.2, net.4.
        let mut layers = Vec::new();
        let mut n_in = 0;
        for (li, idx) in [0usize, 2, 4].iter().enumerate() {
            let (ws, wv) = take(&format!("net.{idx}.weight"))?;
            let (bs, bv) = take(&format!("net.{idx}.bias"))?;
            if ws.len() != 2 || bs != vec![ws[0]] { return Err(err(format!("net.{idx}: bad shapes {ws:?} {bs:?}"))); }
            if li == 0 { n_in = ws[1]; }
            let w = Tensor::<2>::from_data(TensorData::new(wv, [ws[0], ws[1]]), device);
            let b = Tensor::<2>::from_data(TensorData::new(bv, [1, bs[0]]), device);
            layers.push(Layer { w_t: w.transpose(), b });
        }
        let out = layers.last().map(|l| l.w_t.dims()[1]).unwrap_or(0);
        if out != *heads.offsets.last().unwrap() { return Err(err(format!("logits {out} != offsets end {}", heads.offsets.last().unwrap()))); }
        Ok(Student { layers, n_in, heads, device: device.clone(), stats: Stats::default() })
    }

    pub fn n_features(&self) -> usize { self.n_in }
    pub fn heads(&self) -> &[Head] { &self.heads.heads }

    /// Answers for a batch of states (rows of `n_features()` numbers): one forward, one readback.
    pub async fn decide(&self, features: &[f32]) -> Result<Vec<Vec<Answer>>, Error> {
        if features.is_empty() || features.len() % self.n_in != 0 {
            return Err(err(format!("features: {} is not a multiple of {}", features.len(), self.n_in)));
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        let rows = features.len() / self.n_in;
        let mut x = Tensor::<2>::from_data(TensorData::new(features.to_vec(), [rows, self.n_in]), &self.device);
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            x = x.matmul(l.w_t.clone()) + l.b.clone();
            if i != last { x = activation::gelu(x); }
        }
        let z = read(x).await?;
        self.stats.readbacks.fetch_add(1, Ordering::Relaxed);
        let width = *self.heads.offsets.last().unwrap();
        Ok((0..rows).map(|r| self.answers(&z[r * width..(r + 1) * width])).collect())
    }

    /// Calibrated softmax per head (logits / T).
    fn answers(&self, z: &[f32]) -> Vec<Answer> {
        self.heads.heads.iter().enumerate().map(|(i, h)| {
            let (a, b) = (self.heads.offsets[i], self.heads.offsets[i + 1]);
            let t = self.heads.temperatures[i] as f32;
            let m = z[a..b].iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v / t));
            let e: Vec<f32> = z[a..b].iter().map(|v| (v / t - m).exp()).collect();
            let s: f32 = e.iter().sum();
            let p: Vec<f32> = e.iter().map(|v| v / s).collect();
            let (k, c) = p.iter().enumerate().fold((0, f32::MIN), |acc, (j, v)| if *v > acc.1 { (j, *v) } else { acc });
            Answer { id: h.id.clone(), kind: h.kind.clone(), choice: h.keys[k].clone(), confidence: c,
                     probabilities: h.keys.iter().cloned().zip(p).collect() }
        }).collect()
    }
}

async fn read(x: Tensor<2>) -> Result<Vec<f32>, Error> {
    let data = x.into_data_async().await;
    let data = data.map_err(|e| err(format!("readback: {e:?}")))?;
    data.try_to_vec::<f32>().map_err(|e| err(format!("readback: {e:?}")))
}
