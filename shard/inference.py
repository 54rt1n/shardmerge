# shard/inference.py
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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Optional


class InferenceEngine:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Set padding side to left for better streaming
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        load_in_4bit = False,
        load_in_8bit = False,
        device: Optional[str] = None
    ):
        """
        Factory method to create an InferenceEngine instance with optional quantization.
        
        Args:
            model_path (str): Path to the model
            load_in_4bit (bool): Whether to load the model in 4-bit precision
            load_in_8bit (bool): Whether to load the model in 8-bit precision
            
        Returns:
            InferenceEngine: Initialized inference engine
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure quantization if requested
        quantization_config = None
        device_map = "auto"
        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot load model in both 4-bit and 8-bit precision")
        elif load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                llm_int8_enable_fp32_cpu_offload=True
            )
            device_map = {
                "": "cpu",
            }
            
        # Load model with appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device_map
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return cls(model, tokenizer, device)

    def stream_generation(
        self,
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    ):
        """
        Stream generate text from the model token by token
        
        Args:
            prompt (str): Input prompt text
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature 
            top_p (float): Nucleus sampling parameter
            top_k (int): Top-k sampling parameter
            repetition_penalty (float): Penalty for repeating tokens
            
        Yields:
            str: Generated text chunks
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Create empty past key values
        past_key_values = None
        input_ids = inputs["input_ids"]
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    use_cache=True
                )
                
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Decode the last generated token
                current_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                if current_text.strip():  # Only yield non-empty strings
                    yield current_text
                
                # Stop if we hit the EOS token
                if next_token[0] == self.tokenizer.eos_token_id:
                    break