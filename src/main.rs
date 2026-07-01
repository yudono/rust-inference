// ============================================================================
// main.rs — GGUF Inference CLI (Pure Rust)
// ============================================================================
//
// Command-line interface for the GGUF inference engine.
//
// Usage:
//   cargo run --release -- --model <model.gguf> --prompt "<text>" [options]
//
// Options:
//   --model     Path to GGUF model file (required)
//   --prompt    Input text prompt (required)
//   --max-tokens   Maximum tokens to generate (default: 512)
//   --temperature  Sampling temperature (default: 0.8)
//   --top-k        Top-K sampling (default: 40)
//   --top-p        Top-P nucleus sampling (default: 0.95)
//   --seed         Random seed (default: 12345)
//   --max-seq-len  Maximum sequence length (default: from model)

#![allow(dead_code, non_camel_case_types)]

mod attention;
mod gguf;
mod gpu_backend;
mod kv_cache;
mod math;
mod mlp;
mod model;
mod quant;
mod rmsnorm;
mod rope;
mod sampler;
mod tensor;
mod tokenizer;
mod transformer;

use std::env;
use std::io::{self, Write, BufRead};
use std::path::PathBuf;

use gpu_backend::GpuContext;
use model::Model;
use sampler::SamplerConfig;

// ============================================================================
// Argument Parsing
// ============================================================================

struct Args {
    model_path: PathBuf,
    prompt: Option<String>,
    max_tokens: usize,
    sampler_config: SamplerConfig,
    max_seq_len: Option<usize>,
    interactive: bool,
    use_gpu: Option<bool>,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut model_path: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut max_tokens = 512usize;
    let mut temperature = 0.8f32;
    let mut top_k = 40usize;
    let mut top_p = 0.95f32;
    let mut seed = 12345u64;
    let mut max_seq_len: Option<usize> = None;
    let mut interactive = false;
    let mut use_gpu: Option<bool> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("--model requires a value")?
                        .clone(),
                ));
            }
            "--prompt" | "-p" => {
                i += 1;
                prompt = Some(
                    args.get(i)
                        .ok_or("--prompt requires a value")?
                        .clone(),
                );
            }
            "--max-tokens" | "-n" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or("--max-tokens requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid --max-tokens: {}", e))?;
            }
            "--temperature" | "-t" => {
                i += 1;
                temperature = args
                    .get(i)
                    .ok_or("--temperature requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid --temperature: {}", e))?;
            }
            "--top-k" | "-k" => {
                i += 1;
                top_k = args
                    .get(i)
                    .ok_or("--top-k requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid --top-k: {}", e))?;
            }
            "--top-p" | "-p_arg" => {
                i += 1;
                top_p = args
                    .get(i)
                    .ok_or("--top-p requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid --top-p: {}", e))?;
            }
            "--seed" | "-s" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("--seed requires a value")?
                    .parse()
                    .map_err(|e| format!("Invalid --seed: {}", e))?;
            }
            "--max-seq-len" => {
                i += 1;
                max_seq_len = Some(
                    args.get(i)
                        .ok_or("--max-seq-len requires a value")?
                        .parse()
                        .map_err(|e| format!("Invalid --max-seq-len: {}", e))?,
                );
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            "--cpu" => {
                use_gpu = Some(false);
            }
            "--gpu" => {
                use_gpu = Some(true);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = model_path.ok_or("--model is required")?;

    if !model_path.exists() {
        return Err(format!("Model file not found: {}", model_path.display()));
    }

    Ok(Args {
        model_path,
        prompt,
        max_tokens,
        sampler_config: SamplerConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.1,
            repetition_window: 64,
            seed,
        },
        max_seq_len,
        interactive,
        use_gpu,
    })
}

fn print_usage() {
    eprintln!("GGUF Rust Inference Engine");
    eprintln!();
    eprintln!("Usage: gguf-infer --model <model.gguf> --prompt \"<text>\" [options]");
    eprintln!("       gguf-infer --model <model.gguf> --interactive");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model, -m        Path to GGUF model file (required)");
    eprintln!("  --prompt, -p       Input text prompt (for single inference)");
    eprintln!("  --interactive, -i  Interactive mode (like Ollama)");
    eprintln!("  --max-tokens, -n   Maximum tokens to generate (default: 512)");
    eprintln!("  --temperature, -t  Sampling temperature (default: 0.8)");
    eprintln!("  --top-k, -k        Top-K sampling (default: 40)");
    eprintln!("  --top-p            Top-P nucleus sampling (default: 0.95)");
    eprintln!("  --seed, -s         Random seed (default: 12345)");
    eprintln!("  --max-seq-len      Override max sequence length");
    eprintln!("  --cpu              Force CPU backend (default: auto)");
    eprintln!("  --gpu              Force GPU backend (default: auto)");
    eprintln!("  --help, -h         Show this help message");
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    // Parse command-line arguments
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            print_usage();
            std::process::exit(1);
        }
    };

    // Initialize GPU context (if requested or auto-detect)
    let gpu = match args.use_gpu {
        Some(true) => {
            eprintln!("Initializing GPU backend...");
            GpuContext::try_init()
        }
        Some(false) => {
            eprintln!("CPU backend (--cpu flag)");
            None
        }
        None => {
            // Auto-detect: try GPU, fall back to CPU
            GpuContext::try_init()
        }
    };
    if gpu.is_some() {
        eprintln!("GPU backend initialized");
    } else if args.use_gpu == Some(true) {
        eprintln!("WARNING: --gpu requested but no GPU adapter found, falling back to CPU");
    }

    // Load model
    let mut model = match Model::load(&args.model_path, args.max_seq_len) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    model.gpu = gpu;

    if args.interactive {
        // Interactive mode
        eprintln!("Interactive mode (type 'exit' to quit)\n");
        
        let stdin = io::stdin();
        loop {
            eprint!(">>> ");
            io::stderr().flush().ok();
            
            let mut input = String::new();
            match stdin.lock().read_line(&mut input) {
                Ok(_) => {
                    let input = input.trim();
                    if input == "exit" || input == "quit" {
                        break;
                    }
                    if input.is_empty() {
                        continue;
                    }
                    
                    // Generate response
                    eprintln!();
                    let _ = model.generate(input, args.max_tokens, args.sampler_config.clone());
                    eprintln!();
                }
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    break;
                }
            }
        }
    } else {
        // Single prompt mode
        let prompt = match args.prompt {
            Some(p) => p,
            None => {
                eprintln!("Error: --prompt is required (or use --interactive)");
                std::process::exit(1);
            }
        };
        
        let _ = model.generate(&prompt, args.max_tokens, args.sampler_config);
    }
}
