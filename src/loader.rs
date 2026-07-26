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
use std::io::Read;
use std::iter::zip;

use serde::Deserialize;
use serde_json::Value;
use tenferro_runtime::TypedTensor;

#[derive(Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Load a Safetensors file and convert the tensors into `tenferro_runtime::TypedTensor<f32>`.
///
/// [Specification of the Safetensors file format](https://github.com/safetensors/safetensors#format)
pub fn load_safetensors(
    path_to_model: &str,
) -> Result<HashMap<String, TypedTensor<f32>>, Box<dyn Error>> {
    let mut file = File::open(path_to_model)?;

    let size_of_header = {
        let mut buffer = [0; 8];
        file.read_exact(&mut buffer)?;
        usize::from_le_bytes(buffer)
    };

    let header: HashMap<String, Value> = {
        let mut buffer = vec![0; size_of_header];
        file.read_exact(&mut buffer)?;
        serde_json::from_slice(&buffer)?
    };

    let byte_buffer = {
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        buffer
    };

    let mut tensors = HashMap::new();

    for (key, value) in header.into_iter() {
        // I do not use metadata in my inferenece engine.
        if key == "__metadata__" {
            continue;
        }

        let name = key;
        let info: Info = serde_json::from_value(value)?;

        // My inference engine uses only f32 tensors.
        if info.dtype != "F32" {
            continue;
        }

        // f32 is 4 bytes long.
        let size = 4 * info.shape.iter().product::<usize>();
        if size < info.data_offsets.1 - info.data_offsets.0 {
            return Err("tensor data smaller than tensor shape suggests".into());
        }
        let begin = info.data_offsets.0;
        let end = begin + size;

        // ⎛a₁₁ a₁₂ a₁₃⎞
        // ⎝a₂₁ a₂₂ a₂₃⎠
        //
        // Safetensors uses row-major: a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃.
        // https://github.com/safetensors/safetensors#format
        //
        // tenferro uses column-major: a₁₁ a₂₁ a₁₂ a₂₂ a₁₃ a₂₃.
        // https://tensor4all.org/tenferro-rs/getting-started/pytorch-jax-mapping.html#column-major-storage

        let rowmaj: Vec<f32> = byte_buffer[begin..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(*chunk.as_array::<4>().unwrap()))
            .collect();

        let (rowmaj_factors, colmaj_factors) = {
            let mut rowmaj_factors: Vec<usize> = Vec::with_capacity(info.shape.len());
            let mut colmaj_factors: Vec<usize> = Vec::with_capacity(info.shape.len());

            for i in 0..info.shape.len() {
                let rowmaj_factor = info.shape[i + 1..info.shape.len()].iter().product();
                rowmaj_factors.push(rowmaj_factor);

                let colmaj_factor = info.shape[0..i].iter().product();
                colmaj_factors.push(colmaj_factor);
            }

            (rowmaj_factors, colmaj_factors)
        };

        let colmaj = {
            let capacity: usize = info.shape.iter().product();
            let mut colmaj = Vec::with_capacity(capacity);
            for index_for_colmaj in 0..capacity {
                let mut index_for_rowmaj = 0usize;
                for (size, (colmaj_factor, rowmaj_factor)) in
                    zip(&info.shape, zip(&colmaj_factors, &rowmaj_factors))
                {
                    let index = index_for_colmaj / colmaj_factor % size;
                    index_for_rowmaj += index * rowmaj_factor;
                }
                colmaj.push(rowmaj[index_for_rowmaj]);
            }
            colmaj
        };

        let tensor = TypedTensor::<f32>::from_vec_col_major(info.shape, colmaj)?;
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}
