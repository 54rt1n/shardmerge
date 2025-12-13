# tests/test_inference.py
import pytest
import torch
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from shard.inference import ChatMessage, InferenceEngine


class TestChatMessage:
    """Test suite for ChatMessage"""

    def test_chat_message_creation(self):
        """Test ChatMessage creation"""
        msg = ChatMessage(role="user", content="Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_to_dict(self):
        """Test to_dict method"""
        msg = ChatMessage(role="assistant", content="Hi there!")

        result = msg.to_dict()

        assert result == {"role": "assistant", "content": "Hi there!"}
        assert isinstance(result, dict)

    def test_str_representation(self):
        """Test __str__ returns JSON string"""
        msg = ChatMessage(role="system", content="You are helpful")

        result = str(msg)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["role"] == "system"
        assert parsed["content"] == "You are helpful"


class TestInferenceEngine:
    """Test suite for InferenceEngine"""

    def test_inference_engine_creation(self, mock_transformers):
        """Test InferenceEngine creation"""
        model = mock_transformers["model"]
        tokenizer = mock_transformers["tokenizer"]

        engine = InferenceEngine(model, tokenizer, device="cpu")

        assert engine.model == model
        assert engine.tokenizer == tokenizer
        assert engine.device == "cpu"

    def test_tokenizer_padding_configuration(self, mock_transformers):
        """Test tokenizer is configured with left padding"""
        tokenizer = mock_transformers["tokenizer"]
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"

        engine = InferenceEngine(mock_transformers["model"], tokenizer, device="cpu")

        assert tokenizer.padding_side == "left"
        assert tokenizer.pad_token == "<eos>"

    def test_tokenizer_keeps_existing_pad_token(self, mock_transformers):
        """Test tokenizer keeps existing pad_token"""
        tokenizer = mock_transformers["tokenizer"]
        tokenizer.pad_token = "<pad>"

        engine = InferenceEngine(mock_transformers["model"], tokenizer, device="cpu")

        assert tokenizer.pad_token == "<pad>"

    def test_context_manager_enter(self, mock_transformers):
        """Test __enter__ returns engine"""
        engine = InferenceEngine(
            mock_transformers["model"],
            mock_transformers["tokenizer"],
            device="cpu"
        )

        with engine as eng:
            assert eng == engine

    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.memory_allocated")
    def test_context_manager_exit(self, mock_memory, mock_empty_cache, mock_transformers):
        """Test __exit__ cleans up resources"""
        mock_memory.return_value = 1024 * 1024  # 1 MB

        engine = InferenceEngine(
            mock_transformers["model"],
            mock_transformers["tokenizer"],
            device="cpu"
        )

        with engine:
            pass

        # Should have called cleanup functions
        mock_empty_cache.assert_called_once()

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_from_pretrained_cpu(self, mock_tokenizer_class, mock_model_class):
        """Test from_pretrained creates engine with CPU"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model_class.return_value = mock_model
        mock_tokenizer_class.return_value = mock_tokenizer

        engine = InferenceEngine.from_pretrained(
            "test/model",
            load_in_4bit=False,
            load_in_8bit=False,
            device="cpu"
        )

        assert engine.model == mock_model
        assert engine.tokenizer == mock_tokenizer
        assert engine.device == "cpu"

        # Check model was loaded with correct dtype
        call_kwargs = mock_model_class.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float32

    @patch("torch.cuda.is_available")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_from_pretrained_4bit(self, mock_tokenizer_class, mock_model_class, mock_cuda):
        """Test from_pretrained with 4-bit quantization"""
        mock_cuda.return_value = True
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model_class.return_value = mock_model
        mock_tokenizer_class.return_value = mock_tokenizer

        engine = InferenceEngine.from_pretrained(
            "test/model",
            load_in_4bit=True,
            load_in_8bit=False,
            device="cuda"
        )

        # Check quantization config was passed
        call_kwargs = mock_model_class.call_args[1]
        assert call_kwargs["quantization_config"] is not None
        assert call_kwargs["quantization_config"].load_in_4bit is True

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_from_pretrained_8bit(self, mock_tokenizer_class, mock_model_class):
        """Test from_pretrained with 8-bit quantization"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model_class.return_value = mock_model
        mock_tokenizer_class.return_value = mock_tokenizer

        engine = InferenceEngine.from_pretrained(
            "test/model",
            load_in_4bit=False,
            load_in_8bit=True
        )

        # Check quantization config was passed
        call_kwargs = mock_model_class.call_args[1]
        assert call_kwargs["quantization_config"] is not None
        assert call_kwargs["quantization_config"].load_in_8bit is True

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_from_pretrained_both_quant_raises_error(self, mock_tokenizer_class, mock_model_class):
        """Test from_pretrained raises error for both 4-bit and 8-bit"""
        with pytest.raises(ValueError, match="Cannot load model in both 4-bit and 8-bit"):
            InferenceEngine.from_pretrained(
                "test/model",
                load_in_4bit=True,
                load_in_8bit=True
            )

    def test_stream_generation_with_template(self, mock_transformers):
        """Test stream_generation applies chat template"""
        model = mock_transformers["model"]
        tokenizer = mock_transformers["tokenizer"]

        # Setup mocks
        tokenizer.apply_chat_template = Mock(return_value=torch.tensor([[1, 2, 3]]))
        tokenizer.eos_token_id = 2

        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 100)
        mock_output.past_key_values = None
        model.return_value = mock_output

        tokenizer.decode = Mock(return_value="test")

        engine = InferenceEngine(model, tokenizer, device="cpu")

        result = list(engine.stream_generation(
            prompt="Hello",
            max_new_tokens=2,
            use_template=True
        ))

        # Verify chat template was applied
        tokenizer.apply_chat_template.assert_called_once()
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert call_args[-1]["role"] == "user"
        assert call_args[-1]["content"] == "Hello"

    def test_stream_generation_without_template(self, mock_transformers):
        """Test stream_generation without chat template"""
        model = mock_transformers["model"]
        tokenizer = mock_transformers["tokenizer"]

        # Setup mocks - tokenizer returns BatchEncoding-like dict
        mock_encoding = MagicMock()
        mock_encoding.to = Mock(return_value=torch.tensor([[1, 2, 3]]))
        tokenizer.return_value = mock_encoding
        tokenizer.eos_token_id = 2

        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 100)
        mock_output.past_key_values = None
        model.return_value = mock_output

        tokenizer.decode = Mock(return_value="test")

        engine = InferenceEngine(model, tokenizer, device="cpu")

        result = list(engine.stream_generation(
            prompt="Hello",
            max_new_tokens=2,
            use_template=False
        ))

        # Verify tokenizer was called directly
        tokenizer.assert_called_once()

    def test_stream_generation_with_system_prompt(self, mock_transformers):
        """Test stream_generation with system prompt"""
        model = mock_transformers["model"]
        tokenizer = mock_transformers["tokenizer"]

        # Setup mocks
        tokenizer.apply_chat_template = Mock(return_value=torch.tensor([[1, 2, 3]]))
        tokenizer.eos_token_id = 2

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 100)
        mock_output.past_key_values = None
        model.return_value = mock_output

        tokenizer.decode = Mock(return_value="test")

        engine = InferenceEngine(model, tokenizer, device="cpu")

        result = list(engine.stream_generation(
            prompt="Hello",
            max_new_tokens=2,
            use_template=True,
            system_prompt="You are helpful"
        ))

        # Verify system prompt was included
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == "You are helpful"

    def test_stream_generation_with_previous_messages(self, mock_transformers):
        """Test stream_generation with previous conversation history"""
        model = mock_transformers["model"]
        tokenizer = mock_transformers["tokenizer"]

        # Setup mocks
        tokenizer.apply_chat_template = Mock(return_value=torch.tensor([[1, 2, 3]]))
        tokenizer.eos_token_id = 2

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 100)
        mock_output.past_key_values = None
        model.return_value = mock_output

        tokenizer.decode = Mock(return_value="test")

        engine = InferenceEngine(model, tokenizer, device="cpu")

        previous_msgs = [
            ChatMessage(role="user", content="First message"),
            ChatMessage(role="assistant", content="First response")
        ]

        result = list(engine.stream_generation(
            prompt="Second message",
            max_new_tokens=2,
            use_template=True,
            previous_messages=previous_msgs
        ))

        # Verify previous messages were included
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert len(call_args) >= 3
        assert call_args[0]["content"] == "First message"
        assert call_args[1]["content"] == "First response"
