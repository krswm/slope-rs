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

use serde::Deserialize;
use serde_json::Value;
use tenferro_runtime::TypedTensor;

#[derive(Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
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
        let begin = info.data_offsets[0];
        let end = info.data_offsets[0];
        if 4 * size < end - begin {
            return Err("tensor data smaller than tensor shape suggests".into());
        }

        // ⎛a₁₁ a₁₂ a₁₃⎞
        // ⎝a₂₁ a₂₂ a₂₃⎠
        //
        // Safetensors uses row-major: a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃.
        // https://github.com/safetensors/safetensors#format
        //
        // tenferro uses column-major: a₁₁ a₂₁ a₁₂ a₂₂ a₁₃ a₂₃.
        // https://tensor4all.org/tenferro-rs/getting-started/pytorch-jax-mapping.html#column-major-storage

        let rowmaj: Vec<f32> = byte_buffer[begin..begin + size]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(*chunk.as_array::<4>().unwrap()))
            .collect();

        /*
        // My inference engine uses only 1D and 2D tensors.
        match info.shape.len() {
            1 => {
                // Row-major and column-major are identical in 1D.
                let colmaj = rowmaj;

                let tensor = TypedTensor::<f32>::from_vec_col_major(info.shape, colmaj)?;
                tensors.insert(name, tensor);
            }
            2 => {
                let mut colmaj = Vec::with_capacity(info.shape[0] * info.shape[1]);

                for col in 0..info.shape[1] {
                    for row in 0..info.shape[0] {
                        colmaj.push(rowmaj[row * info.shape[1] + col]);
                    }
                }

                let tensor = TypedTensor::<f32>::from_vec_col_major(info.shape, colmaj)?;
                tensors.insert(name, tensor);
            }
            _ => (),
        };
        */

        let mut divisors = Vec::with_capacity(info.shape.len());
        for j in 0..info.shape.len() {
            let mut divisor = 1usize;
            let mut k = j + 1;
            while k < info.shape.len() {
                divisor *= &info.shape[k];
                k += 1;
            }
            divisors.push(divisor);
        }

        let mut units = Vec::with_capacity(info.shape.len());
        for j in 0..info.shape.len() {
            let mut unit = 1usize;
            let mut k = 0;
            while k < j {
                unit *= &info.shape[k];
                k += 1;
            }
            units.push(unit);
        }

        let capacity: usize = info.shape.iter().product();
        let mut colmaj = Vec::with_capacity(capacity);
        for i in 0..capacity {
            let mut indices = Vec::with_capacity(info.shape.len());
            for (unit, size) in std::iter::zip(&units, &info.shape) {
                let index = i / unit % size;
                indices.push(index);
            }

            let mut l = 0usize;
            for j in 0..info.shape.len() {
                l += indices[j] * divisors[j];
            }

            let element = &rowmaj[l];

            colmaj.push(*element);
        }

        let tensor = TypedTensor::<f32>::from_vec_col_major(info.shape, colmaj)?;
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}
