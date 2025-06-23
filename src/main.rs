use std::io::{self, Read};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct F16State {
    pub vt: f64,
    pub alpha: f64,
    pub beta: f64,
    pub phi: f64,
    pub theta: f64,
    pub psi: f64,
    pub p: f64,
    pub q: f64,
    pub r: f64,
    pub pn: f64,
    pub pe: f64,
    pub h: f64,
    pub pow: f64,
    pub time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Position {
    alt: f64,
    lat: f64,
    lon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Target {
    id: u32,
    init_state: F16State,
    waypoints: Vec<Position>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Simulation {
    targets: Vec<Target>,
    time_step: f64,
    max_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationResult {
    target_id: u32,
    time: Vec<f64>,
    waypoints: Vec<Position>,
    run_time: f64,
    final_state: Vec<F16State>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationResultList {
    results: Vec<SimulationResult>,
}

fn main() {
    // Read JSON input from stdin
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).expect("Failed to read input");
    if buffer.trim().is_empty() {
        eprintln!("Error: No input received from stdin.");
        std::process::exit(1);
    }
    // Print the first 500 characters of the input for debugging
    let preview_len = buffer.len().min(500);
    eprintln!("[DEBUG] First 500 chars of input:\n{}", &buffer[..preview_len]);
    let sim: Result<SimulationResultList, _> = serde_json::from_str(&buffer);
    match sim {
        Ok(sim_results) => {
            println!("Received {} simulation result(s).", sim_results.results.len());
            for (i, result) in sim_results.results.iter().enumerate() {
                println!("Result {}: target_id={}, time steps={}, waypoints={}, run_time={:.4}",
                    i + 1,
                    result.target_id,
                    result.time.len(),
                    result.waypoints.len(),
                    result.run_time
                );
            }
        }
        Err(e) => {
            eprintln!("Error: Failed to parse input JSON: {}", e);
            eprintln!("[DEBUG] Input length: {}", buffer.len());
            std::process::exit(1);
        }
    }
}
