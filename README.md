# realtime-game-decisions

**On-device game AI for NPCs, bots and difficulty: microsecond decisions with calibrated probabilities.**
Like Jev, but open and on-device. A game asks questions about its current state ("which way should I move?",
"fire now?", "how dangerous is it?") and gets one answer per question, with calibrated probabilities, in
microseconds, on the CPU, natively or in the browser (wasm32). Inside the Partii app, games reach it as
`partii.ai.decide`.

The model is a per-game **student**: the game's state as a vector of numbers, a 2x128 GELU MLP, and one block of
logits per question (choice, yes/no or a score level), softmaxed with one calibration temperature per question. One
forward pass answers every question for a whole batch of states.

Built on [burn](https://burn.dev) (pure-Rust CPU backend, `flex`). Made by the Partii team.

The question-and-answer shape (choice, yes/no and score questions about a game state, with probabilities and
confidence) follows the Jev / [Kev](https://github.com/jaredpalmer/kev) style of game decision model, but this crate
runs small per-game models locally instead of calling an API. It is not affiliated with Jev or its makers, and it does
not use Kev's weights.

## Example

```rust
use realtime_game_decisions::{cpu, Student};

let student = Student::load(&heads_json, &weights_json, &weights_bin, &cpu())?;
let answers = pollster::block_on(student.decide(&features))?;   // features: rows of student.n_features() numbers
for a in &answers[0] {
    println!("{}: {} ({:.0} %)", a.id, a.choice, a.confidence * 100.0);
}
```

Runnable, with the bundled Arena student (`testdata/arena-mlp/`):

```sh
cargo run --example decide
```

## Model files

| File | What |
|---|---|
| `heads.json` | `heads` (each `{id, type, keys}`), `offsets` (logit range per head, one longer than `heads`), `temperatures` |
| `weights.json` | tensor index: `net.{0,2,4}.{weight,bias}` -> `{shape, offset}` (PyTorch `nn.Linear` layout) |
| `weights.bin` | the tensors, little-endian f32 |

## Browser

On `wasm32-unknown-unknown` the crate exports `DecideStudent` through wasm-bindgen: `new DecideStudent(heads, index,
bytes)`, `await s.decide(Float32Array)` returning JSON, and `s.stats()`. Build with
`RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --release`.

## Features

- default: the CPU path only.
- `gpu`: pulls in burn's wgpu backend for the coming GPU path (not implemented yet).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. The
bundled Arena student weights were trained by us on our own synthetic data and carry the same licence. See
[NOTICE](NOTICE) for third-party details.
