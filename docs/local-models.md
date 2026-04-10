# Local Models

This document provides detailed information about running and configuring local models with Kon.

## Tested Models

| Model | Quantization | Context Length | TPS | System Specs |
| ----- | -------------- | -------------- | --- | ------------ |
| `zai-org/glm-4.7-flash` | Q4_K_M | 65,536 | N/A | i7-14700F × 28, 64GB RAM, 24GB VRAM (RTX 3090) |
| `unsloth/Qwen3.5-27B-GGUF` | Q4_K_M | 32,768 | ~30 | i7-14700F × 28, 64GB RAM, 24GB VRAM (RTX 3090) |
| `unsloth/gemma-4-26B-A4B-it-GGUF` | UD-Q4_K_M | 8,192 | N/A | i7-14700F × 28, 64GB RAM, 24GB VRAM (RTX 3090) |

Run Qwen3.5 27B on an RTX 3090 with a 32k context window using llama-server:

```bash
/home/torrent/llama-b8740/llama-server \
  --model /home/torrent/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf \
  --port 5000 \
  --ctx-size 32768 \
  --gpu-layers all \
  --threads 8 \
  --threads-batch 8 \
  --batch-size 1024 \
  --ubatch-size 512 \
  --flash-attn on
```

On this machine, that setup generates at roughly 30 tokens per second.

Then start Kon for a one-off local session:

```bash
kon --model unsloth/Qwen3.5-27B-GGUF --provider openai \
  --base-url http://localhost:5000/v1 \
  --openai-compat-auth none
```

Run Gemma 4 26B A4B on the same machine using llama-server:

```bash
/home/torrent/llama-b8740/llama-server \
  --model /home/torrent/models/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --port 5002 \
  --ctx-size 8192 \
  --gpu-layers all \
  --threads 8 \
  --threads-batch 8 \
  --batch-size 1024 \
  --ubatch-size 512 \
  --flash-attn on
```

Then start Kon against that local server:

```bash
kon --model unsloth/gemma-4-26B-A4B-it-GGUF --provider openai \
  --base-url http://localhost:5002/v1 \
  --openai-compat-auth none
```

Notes for Gemma on llama.cpp / llama-server:

- Use the exact model id above in Kon so it matches the local endpoint setup.
- Reasoning visibility is model/server-template dependent. On this setup, the server can emit `reasoning_content`, but it is not guaranteed on every prompt.
- If thinking does not appear, verify the server template and thinking controls supported by your llama.cpp build.

If this is your default setup, put it in `~/.kon/config.toml` instead:

```toml
[llm]
default_provider = "openai"
default_model = "unsloth/Qwen3.5-27B-GGUF"
default_base_url = "http://localhost:5000/v1"

[llm.auth]
openai_compat = "auto" # or "none" to always inject a placeholder key
```

