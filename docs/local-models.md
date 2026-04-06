# Local Models

This document provides detailed information about running and configuring local models with Kon.

## Tested Models

| Model | Quantization | Context Length | TPS | System Specs |
| ----- | -------------- | -------------- | --- | ------------ |
| `qwen/qwen3-coder-next` | Q4_K_M | 65,536 | N/A | i7-14700F × 28, 64GB RAM, 24GB VRAM (RTX 3090) |
| `zai-org/glm-4.7-flash` | Q4_K_M | 65,536 | N/A | i7-14700F × 28, 64GB RAM, 24GB VRAM (RTX 3090) |
| `unsloth/Qwen3.5-9B-GGUF` | Q4_K_M | 65,536 | N/A | i7-14700F × 28, 64GB RAM, 24GB VRAM (RTX 3090) |

Run a local model using llama-server with the following command:

```bash
./llama-server \
  --hf-repo unsloth/Qwen3.5-9B-GGUF \
  --hf-file Qwen3.5-9B-Q4_K_M.gguf \
  --port 5000 \
  -c 65536
```

Then start Kon for a one-off local session:

```bash
kon --model unsloth/Qwen3.5-9B-GGUF --provider openai \
  --base-url http://localhost:5000/v1 \
  --openai-compat-auth none
```

If this is your default setup, put it in `~/.kon/config.toml` instead:

```toml
[llm]
default_provider = "openai"
default_model = "unsloth/Qwen3.5-9B-GGUF"
default_base_url = "http://localhost:5000/v1"

[llm.auth]
openai_compat = "auto" # or "none" to always inject a placeholder key
```

