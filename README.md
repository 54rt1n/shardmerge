<pre>
  ____    _    _    ____    ____    ____    __  __   _____   ____    _____   _____ 
 / ___|  | |  | |  / _  \  |  _ \  |  _ \  |  \/  | |  ___| |  _ \  / ____| |  ___|
 \___ \  | |__| | | |_| |  | |_| | | | | | | |\/| | | |__   | |_| | | |  _  | |__ 
  ___) | |  __  | |  _  |  |    /  | |_| | | |  | | |  __|  |    /  | |_| | |  __|
 |____/  |_|  |_| |_| |_|  |_|\_\  |____/  |_|  |_| |_____| |_|\_\  \_____| |_____|
</pre>

# ShardMerge

A powerful tool for merging multiple finetuned LLM models by computing and combining their delta weights.

## Features

- Merge multiple finetuned models with their base model
- Efficient shard-based processing with minimal memory and disk footprint
- Concurrent downloads with progress tracking
- Support for HuggingFace model hub integration
- SafeTensors format support

## Installation

```bash
poetry install --no-root
```

## Usage

Create a YAML configuration file:

```yaml
output_base_model: "unsloth/Meta-Llama-3.1-70B-Instruct"
output_dtype: "bfloat16"
finetune_merge:
  - { "model": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", "base": "unsloth/Meta-Llama-3.1-70B-Instruct", "alpha": 0.3, "is_input": true }
  - { "model": "another/finetuned-model", "base": "unsloth/Meta-Llama-3.1-70B-Instruct", "alpha": 0.5, "is_output": true }
output_dir: "output_model"
device: "cpu"
clean_cache: false
cache_dir: "cache"
storage_dir: "storage"
```

Run the merge:

```bash
python -m shard merge config.yaml
```

## Execute

Optional arguments:
- `--verbose`: Enable detailed logging

## How It Works

1. Downloads model shards concurrently
2. Computes delta weights between base and finetuned models
3. Combines deltas efficiently using tensor operations
4. Writes output in compatible SafeTensors format

## License

LGPL-3.0 - See LICENSE.txt file for details.

## Contributing

Contributions welcome! Please feel free to submit pull requests.