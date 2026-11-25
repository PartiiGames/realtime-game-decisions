//! Run the bundled Arena student on one state: `cargo run --example decide`
use realtime_game_decisions::{cpu, Student};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/arena-mlp");
    let heads = std::fs::read_to_string(format!("{dir}/heads.json"))?;
    let index = std::fs::read_to_string(format!("{dir}/weights.json"))?;
    let blob = std::fs::read(format!("{dir}/weights.bin"))?;
    let student = Student::load(&heads, &index, &blob, &cpu())?;

    // One state: n_features() numbers, here the first row of the fixture (real games compute them from their state).
    let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/fixture.json"))?)?;
    let row: Vec<f32> = fixture["features"][0].as_array().ok_or("fixture: features")?.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
    assert_eq!(row.len(), student.n_features());

    let answers = pollster::block_on(student.decide(&row))?;
    for a in &answers[0] {
        println!("{:>10}: {:<10} ({:.0} %)", a.id, a.choice, a.confidence * 100.0);
    }
    Ok(())
}
