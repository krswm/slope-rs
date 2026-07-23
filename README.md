# GPT-2 Inference with tenferro

I built a GPT-2 inference engine from scratch in Rust.

It is built on [tenferro](https://github.com/tensor4all/tenferro-rs), a Rust-native tensor library.

![Demo](asset/demo.gif)

I also built [a Julia counterpart](https://github.com/krswm/slope-jl).

## Quickstart

I made this project just for educational purpose. Use at your own risk.

It is assumed that you have Git, cURL, and Cargo installed on your machine.

**Download a pretrained GPT-2 model from Hugging Face.**

```
curl --progress-bar --location --remote-name --output-dir model --create-dirs 'https://huggingface.co/openai-community/gpt2/resolve/main/{config.json,vocab.json,merges.txt,model.safetensors}'
```

**Clone this repository.**

```
git clone https://github.com/krswm/slope-rs.git
cd slope-rs
```

**Compile the program.**

```
cargo build --release
```

**Start generating text.**
The GPT-2 model is not for chat conversation, but for text continuation.
Watch the model continues your prompt.

```
./target/release/slope-rs ../model 'Natural language processing is a branch of computer science. We study'
```

Hit `Control+C` to stop generating text.

## Supported Models

This program only supports models that are build on the GPT-2 architecture.

This program only supports models that have the following files in the model repository.

- `config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors`

I have verified that this program works with the following models.

- [GPT-2](https://huggingface.co/openai-community/gpt2)
- [GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium)
- [GPT-2 Large](https://huggingface.co/openai-community/gpt2-large)
- [GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl)

## Source Files

- [`src/loader.rs`](src/loader.rs) loads [a Safetensors file](https://github.com/safetensors/safetensors) and convert the tensors into tenferro’s `TypedTensor`.
- [`src/tokenizer.rs`](src/tokenizer.rs) tokenizes your prompt with the BPE algorithm.
- [`src/transformer.rs`](src/transformer.rs) is the heart of the GPT-2 inferenece. It receives tokens (your prompt + already generated text) and predicts the next token.
- [`src/main.rs`](src/main.rs) loads files from the GPT-2 repository and generates text.

## My Future Plans

- Make the text generation not deterministic. Apply stochasticity for sampling.
- Optimize the inference. Use techniques such as KV-cache.
- Use GPU backend. tenferro provides CUDA and WebGPU backend. (I’m not sure whether I currently have an access for such hardware though…)
- Support variety of LLM architectures beyond GPT-2 such as Llama 2.
- Ultimately, implement LLM training from scratch. (I’m interested on [TinyStories](https://arxiv.org/pdf/2305.07759).)

## Credits

- [GPT-2](https://huggingface.co/openai-community/gpt2) for devising an influental LLM architecture.
- [*GPT in 60 Lines of NumPy*](https://jaykmody.com/blog/gpt-from-scratch/) (a blog post) for teaching me how to implement a GPT-2 inference engine from scratch.
- [*Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch*](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) (a blog post) for teaching me how to implement a BPE tokenizer from scratch.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me an amazing tensor library for Rust.

## Development

This is a hobby project of mine I started from scratch.

I started this project on 2026-07-03 and finished my first implementation on 2026-07-14.

I used open source LLM inference engines (Ollama, etc.) and open source LLM models (TinyLlama, GPT-2, etc.) only for the purpose to observe their behavior as LLM architecture.
Except for that, I did **not** use generative AI for this project at all.
