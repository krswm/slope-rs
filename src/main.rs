// GPT-2 Inference with tenferro
// Copyright (C) 2026  Kurosawa Mutsumi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use serde_json::Value;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::TypedTensor;

pub mod loader;
pub mod tokenizer;
pub mod transformer;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        println!("GPT-2 Inference with tenferro");
        println!(
            "Usage: \x1b[1mcargo run --release <path to model repository> <your prompt>\x1b[22m"
        );
        println!("You may have to enclose 'your prompt' with quotes.");
        return Ok(());
    }

    // ==== Loading Files ====

    let config = {
        let path = &format!("{}/config.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let conf: HashMap<String, Value> = serde_json::from_reader(reader)?;
        transformer::Config {
            layer_norm_epsilon: conf["layer_norm_epsilon"].as_f64().unwrap() as f32,
            n_ctx: conf["n_ctx"].as_u64().unwrap() as usize,
            n_embd: conf["n_embd"].as_u64().unwrap() as usize,
            n_head: conf["n_head"].as_u64().unwrap() as usize,
            n_layer: conf["n_layer"].as_u64().unwrap() as usize,
            vocab_size: conf["vocab_size"].as_u64().unwrap() as usize,
        }
    };

    let (token_to_id, id_to_token) = {
        let path = &format!("{}/vocab.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let token_to_id: HashMap<String, usize> = serde_json::from_reader(reader)?;
        let id_to_token: HashMap<usize, String> = token_to_id
            .iter()
            .map(|(key, value)| (*value, key.clone()))
            .collect();
        (token_to_id, id_to_token)
    };

    let ranks = {
        let path = &format!("{}/merges.txt", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut ranks = HashMap::new();
        let mut rank = 0u32;
        for line in reader.lines().map_while(Result::ok) {
            // Skip a comment line.
            if line.starts_with("#") {
                continue;
            }

            let mut split = line.split(" ");
            let token0 = split.next().unwrap().to_string();
            let token1 = split.next().unwrap().to_string();

            ranks.insert((token0, token1), rank);
            rank += 1;
        }
        ranks
    };

    // ==== Tokenization ====

    if args[2].is_empty() {
        println!("Your prompt should not be empty.");
        return Ok(());
    }

    let mut ids = tokenizer::tokenize(&token_to_id, &ranks, &args[2])?;

    if ids.len() >= config.n_ctx {
        println!("Your prompt exceeds the context length. Try shorter prompt.");
        return Ok(());
    }

    // ==== Loading Tensors ====

    let tensors = {
        let path = &format!("{}/model.safetensors", &args[1]);
        loader::load_safetensors(path)?
    };

    // ==== Inference ====

    print!("\x1b[1;90m{}\x1b[22;39m", &args[2]);
    std::io::stdout().flush()?;

    let mut backend = CpuBackend::new();
    let mut utf8_buffer = Vec::new();
    loop {
        match generate_next_id(&tensors, &config, &ids, &mut backend) {
            Ok(next_id) => {
                if ids.len() == config.n_ctx - 1 {
                    ids.remove(0);
                }
                ids.push(next_id);

                let decoded =
                    tokenizer::decode_unique_encoding(&id_to_token[&next_id], &mut utf8_buffer);
                print!("\x1b[1m{decoded}\x1b[22m");
                std::io::stdout().flush()?;
            }
            Err(err) => {
                println!();
                return Err(err);
            }
        };
    }
}

fn generate_next_id(
    tensors: &HashMap<String, TypedTensor<f32>>,
    config: &transformer::Config,
    ids: &Vec<usize>,
    backend: &mut CpuBackend,
) -> Result<usize, Box<dyn Error>> {
    let output = transformer::transform(tensors, config, ids, backend)?;

    // Greedy sampling: Choose the token with the highest probability.
    let next_id = output
        .host_data()?
        .iter()
        .enumerate()
        .max_by(|(_, prob0), (_, prob1)| prob0.total_cmp(prob1))
        .map(|(id, _)| id)
        .unwrap();

    Ok(next_id)
}
