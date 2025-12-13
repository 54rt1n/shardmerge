# shard/__main__.py
# Copyright (C) 2024 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import click
import logging
import asyncio
from pathlib import Path
from typing import List, Optional
from .merge.fast_fourier import FourierMerge
from .download import DownloadManager, DownloadStats
from .index import HFMultiModelIndex
from .writer import ModelWriter
from .config import MergeConfig

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool):
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def progress_callback(stats: DownloadStats):
    """Display download progress"""
    print(f"\rDownload Progress: {stats.progress_pct:.1f}% | "
        f"Active: {stats.active_workers} | "
        f"Complete: {stats.completed_jobs} | "
        f"Failed: {stats.failed_jobs} | "
        f"Downloaded: {stats.total_downloaded/1024/1024:.1f}MB", 
        end="")

async def run_merge(
    config: MergeConfig,
    device: str,
    clean_cache: bool,
    **kwargs,
):
    """Run the merge operation with the given configuration"""
    # Setup components
    download_manager = DownloadManager(
        storage_path=config.storage_path,
        clean_cache=clean_cache,
    )
    download_manager.progress_callbacks.append(progress_callback)
    
    index_manager = HFMultiModelIndex(
        download_manager=download_manager,
        cache_path=config.cache_path,
    )
    
    # Create and run merger
    merger = FourierMerge(
        config=config,
        index_manager=index_manager,
        **kwargs,
    )
    
    await merger.merge(device=device)
    
    if clean_cache:
        await download_manager.cleanup()

@click.group()
def cli():
    """Shard merge utility for merging and managing model shards."""
    pass

@cli.command('merge')
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--cache-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Directory for caching downloaded files'
)
@click.option(
    '--clean_cache',
    is_flag=True,
    help='Delete cached files after merging'
)
@click.option(
    '--device',
    type=str,
    default=None,
    help='Device to perform tensor operations on (cuda/cpu)'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose logging'
)
def merge_command(
    config_file: Path,
    cache_dir: Optional[Path],
    verbose: bool,
    **kwargs,
):
    """
    Merge multiple finetuned models by computing and combining their deltas.
    
    CONFIG_FILE should be a YAML file with the following structure:
    
    \b
    base_model:
      model: "unsloth/Meta-Llama-3.1-70B-Instruct"
      alpha: 1.0
      is_input: false
      is_output: false
      start_layer: 0
      end_layer: -1
    finetune_merge:
      - model: "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
        alpha: 0.8
      - model: "another/finetuned-model"
        alpha: 0.5
        start_layer: 2
        end_layer: -2
    output_path: "merged_model"
    """
    # Setup logging
    setup_logging(verbose)
    
    try:
        # Load configuration
        config = MergeConfig.from_yaml(config_file)
        logger.info(f"Loaded configuration: {config}")
        
        if cache_dir:
            config.cache_dir = cache_dir

        config.update({k: v for k, v in kwargs.items() if v is not None})
        
        # Run merge operation
        asyncio.run(run_merge(
            config=config,
            **config.to_dict(),
        ))
        
    except Exception as e:
        logging.error(f"Error during merge: {e}", exc_info=verbose)
        import traceback
        traceback.print_exc()
        raise click.Abort()

@cli.command('copy-model')
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--revision',
    type=str,
    default="main",
    help='Model revision/tag to use'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose logging'
)
def copy_model_command(config_file: Path, revision: str, verbose: bool):
    """
    Copy model configuration files from Hugging Face to the output directory.
    
    CONFIG_FILE should be a YAML file containing 'input_model' and 'output_path' fields.
    """
    setup_logging(verbose)
    
    try:
        # Load configuration
        config = MergeConfig.from_yaml(config_file)
        input_model = config.input_model.model if config.input_model else config.output_base_model
        output_path = config.output_path
        
        logger.info(f"Copying model configuration from {input_model} to {output_path}")
        
        # Get layer order from the model configuration
        writer = ModelWriter.from_huggingface(
            model_id=input_model,
            output_path=output_path,
            layer_order=[],  # Empty list since we're only copying configs
            revision=revision
        )
        
        logger.info(f"Successfully copied model configuration files to {output_path}")
        
    except Exception as e:
        logging.error(f"Error copying model configuration: {e}", exc_info=verbose)
        raise click.Abort()
    
from .inference import InferenceEngine

@cli.command('generate')
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.argument('prompt', type=str)
@click.option(
    '--max-tokens',
    type=int,
    default=512,
    help='Maximum number of tokens to generate'
)
@click.option(
    '--temperature',
    type=float,
    default=0.7,
    help='Sampling temperature (higher = more random)'
)
@click.option(
    '--top-p',
    type=float,
    default=0.95,
    help='Nucleus sampling parameter'
)
@click.option(
    '--top-k',
    type=int,
    default=40,
    help='Top-k sampling parameter'
)
@click.option(
    '--repetition-penalty',
    type=float,
    default=1.1,
    help='Penalty for repeating tokens'
)
@click.option(
    '-4', '--load-in-4bit',
    is_flag=True,
    help='Load model in 4-bit precision'
)
@click.option(
    '-8', '--load-in-8bit',
    is_flag=True,
    help='Load model in 8-bit precision'
)
@click.option(
    '--device',
    type=str,
    default=None,
    help='Device to perform tensor operations on (cuda/cpu)'
)
def generate_command(
    model_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    load_in_4bit: bool,
    load_in_8bit: bool,
    device: Optional[str]=None
):
    """
    Generate text from a model using the provided prompt.
    
    MODEL_PATH should be the path to a model directory containing model files.
    PROMPT is the input text to generate from.
    """
    try:
        # Initialize inference engine with quantization options
        engine = InferenceEngine.from_pretrained(
            str(model_path),
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device=device
        )
        
        # Stream generate text
        for text_chunk in engine.stream_generation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        ):
            print(text_chunk, end='', flush=True)
        print()  # Final newline
            
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        import traceback
        traceback.print_exc()
        raise click.Abort()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    cli()