# shard/config.py
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
from dataclasses import dataclass
from pathlib import Path
import torch
from typing import List
import yaml

@dataclass
class MergeModel:
    model: str
    base: str
    alpha: float = 1.0
    is_input: bool = False
    is_output: bool = False
    is_norm: bool = False
    start_layer: int = 0
    end_layer: int = -1

    def use_layer_index(self, layer_index: int) -> bool:
        if self.start_layer > layer_index:
            return False
        if self.end_layer != -1 and self.end_layer < layer_index:
            return False
        return True

@dataclass
class MergeConfig:
    finetune_merge: List[MergeModel]
    output_base_model: str
    output_dir: str
    output_dtype: str = "bfloat16"
    device: str = "cpu"
    clean_cache: bool = False
    cache_dir: str = "cache"
    storage_dir: str = "storage"

    @property
    def input_model(self) -> MergeModel:
        for model in self.finetune_merge:
            if model.is_input:
                return model
        return None
    
    @property
    def output_model(self) -> MergeModel:
        for model in self.finetune_merge:
            if model.is_output:
                return model
        return None

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
    
    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)
    
    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir)
    
    @property
    def output_astype(self) -> torch.dtype:
        return getattr(torch, self.output_dtype)

    def update(self, config: dict = {}, **kwargs):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict:
        return {
            "output_base_model": self.output_base_model,
            "finetune_merge": [model.model for model in self.finetune_merge],
            "output_dir": self.output_dir,
            "device": self.device,
            "clean_cache": self.clean_cache,
            "cache_dir": self.cache_dir,
            "storage_dir": self.storage_dir,
        }

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'MergeConfig':
        """Load and validate YAML configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required = ['output_base_model', 'finetune_merge', 'output_dir']
        missing = [field for field in required if field not in config]
        if missing:
            raise click.BadParameter(
                f"Missing required configuration fields: {', '.join(missing)}"
            )
            
        # Validate finetune_merge is a list
        if not isinstance(config['finetune_merge'], list):
            raise click.BadParameter(
                "finetune_merge must be a list of model URIs"
            )

        # Convert model configurations to MergeModel instances
        config['finetune_merge'] = [MergeModel(**model) for model in config['finetune_merge']]
            
        return MergeConfig(**config)

