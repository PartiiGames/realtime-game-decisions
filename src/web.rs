//! The browser API (wasm32-unknown-unknown, via wasm-bindgen). Async only, one readback per call. Answers come back as JSON in the System One reply shape, one array of answers per state row.
use crate::{cpu, Student};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct DecideStudent(Student);

#[wasm_bindgen]
impl DecideStudent {
    /// heads.json and weights.json as text, weights.bin as bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(heads_json: &str, weights_json: &str, weights_bin: &[u8]) -> Result<DecideStudent, JsError> {
        Student::load(heads_json, weights_json, weights_bin, &cpu()).map(DecideStudent).map_err(|e| JsError::new(&e.0))
    }
    #[wasm_bindgen(js_name = nFeatures)]
    pub fn n_features(&self) -> usize { self.0.n_features() }
    /// Rows of n_features() numbers in, JSON out: [[{id, type, choice, confidence, probabilities:{key: p}}...]...].
    pub async fn decide(&self, features: Vec<f32>) -> Result<String, JsError> {
        let rows = self.0.decide(&features).await.map_err(|e| JsError::new(&e.0))?;
        let v: Vec<Vec<serde_json::Value>> = rows.iter().map(|r| r.iter().map(|a| serde_json::json!({
            "id": a.id, "type": a.kind, "choice": a.choice, "confidence": a.confidence,
            "probabilities": a.probabilities.iter().map(|(k, p)| (k.clone(), serde_json::json!(p))).collect::<serde_json::Map<_, _>>(),
        })).collect()).collect();
        Ok(serde_json::to_string(&v).unwrap())
    }
    /// [calls, readbacks, dispatches] (the counters every addon keeps).
    pub fn stats(&self) -> Vec<u64> {
        use std::sync::atomic::Ordering::Relaxed;
        vec![self.0.stats.calls.load(Relaxed), self.0.stats.readbacks.load(Relaxed), self.0.stats.dispatches.load(Relaxed)]
    }
}
