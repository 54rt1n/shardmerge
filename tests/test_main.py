# tests/test_main.py
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from click.testing import CliRunner

from shard.__main__ import (
    cli,
    merge_command,
    copy_model_command,
    generate_command,
    setup_logging,
    progress_callback
)
from shard.download import DownloadStats


class TestSetupLogging:
    """Test suite for setup_logging function"""

    @patch("logging.basicConfig")
    def test_setup_logging_verbose(self, mock_basicConfig):
        """Test setup_logging with verbose=True"""
        setup_logging(verbose=True)

        mock_basicConfig.assert_called_once()
        call_kwargs = mock_basicConfig.call_args[1]
        import logging
        assert call_kwargs["level"] == logging.DEBUG

    @patch("logging.basicConfig")
    def test_setup_logging_not_verbose(self, mock_basicConfig):
        """Test setup_logging with verbose=False"""
        setup_logging(verbose=False)

        mock_basicConfig.assert_called_once()
        call_kwargs = mock_basicConfig.call_args[1]
        import logging
        assert call_kwargs["level"] == logging.INFO


class TestProgressCallback:
    """Test suite for progress_callback function"""

    async def test_progress_callback_output(self, capsys):
        """Test progress_callback prints correct format"""
        stats = DownloadStats(
            active_workers=2,
            completed_jobs=5,
            failed_jobs=1,
            total_downloaded=1024 * 1024 * 50,  # 50 MB
            total_size=1024 * 1024 * 100  # 100 MB
        )

        await progress_callback(stats)

        captured = capsys.readouterr()
        assert "50.0%" in captured.out
        assert "Active: 2" in captured.out
        assert "Complete: 5" in captured.out
        assert "Failed: 1" in captured.out


class TestCLI:
    """Test suite for CLI commands"""

    def test_cli_group(self):
        """Test CLI group is callable"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Shard merge utility" in result.output


class TestMergeCommand:
    """Test suite for merge command"""

    def test_merge_command_help(self):
        """Test merge command help text"""
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--help"])

        assert result.exit_code == 0
        assert "Merge multiple finetuned models" in result.output
        assert "CONFIG_FILE" in result.output

    @patch("shard.__main__.asyncio.run")
    @patch("shard.__main__.MergeConfig.from_yaml")
    def test_merge_command_valid_config(self, mock_from_yaml, mock_asyncio_run, sample_yaml_config, mock_merge_config):
        """Test merge command with valid config"""
        mock_from_yaml.return_value = mock_merge_config

        runner = CliRunner()
        result = runner.invoke(cli, [
            "merge",
            str(sample_yaml_config),
            "--device", "cpu",
            "--verbose"
        ])

        assert mock_from_yaml.called
        assert mock_asyncio_run.called

    def test_merge_command_missing_config(self):
        """Test merge command with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "merge",
            "/nonexistent/config.yaml"
        ])

        assert result.exit_code != 0

    @patch("shard.__main__.MergeConfig.from_yaml")
    def test_merge_command_with_cache_dir(self, mock_from_yaml, sample_yaml_config, mock_merge_config):
        """Test merge command with custom cache directory"""
        mock_from_yaml.return_value = mock_merge_config

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                "merge",
                str(sample_yaml_config),
                "--cache-dir", "/custom/cache",
                "--clean_cache"
            ])

            # Check config was updated (cache_dir is converted to Path)
            assert str(mock_from_yaml.return_value.cache_dir) == "/custom/cache"

    @patch("shard.__main__.MergeConfig.from_yaml")
    def test_merge_command_error_handling(self, mock_from_yaml, sample_yaml_config):
        """Test merge command handles errors gracefully"""
        mock_from_yaml.side_effect = Exception("Test error")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "merge",
            str(sample_yaml_config)
        ])

        assert result.exit_code != 0


class TestCopyModelCommand:
    """Test suite for copy-model command"""

    def test_copy_model_command_help(self):
        """Test copy-model command help text"""
        runner = CliRunner()
        result = runner.invoke(cli, ["copy-model", "--help"])

        assert result.exit_code == 0
        assert "Copy model configuration files" in result.output

    def test_copy_model_missing_config(self):
        """Test copy-model with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "copy-model",
            "/nonexistent/config.yaml"
        ])

        assert result.exit_code != 0

    @patch("shard.__main__.ModelWriter.from_huggingface")
    @patch("shard.__main__.MergeConfig.from_yaml")
    def test_copy_model_with_input_model(self, mock_from_yaml, mock_writer, sample_yaml_config, tmp_path):
        """Test copy-model command with input_model set"""
        from shard.config import MergeModel, MergeConfig

        # Create config with input model marked
        models = [
            MergeModel(model="test/input-model", base="test/base", alpha=1.0, is_input=True),
            MergeModel(model="test/model2", base="test/base", alpha=0.3),
        ]
        config = MergeConfig(
            finetune_merge=models,
            output_base_model="test/base",
            output_dir=str(tmp_path / "output")
        )
        mock_from_yaml.return_value = config

        runner = CliRunner()
        result = runner.invoke(cli, [
            "copy-model",
            str(sample_yaml_config),
            "--revision", "v1.0"
        ])

        assert mock_writer.called
        call_kwargs = mock_writer.call_args[1]
        assert call_kwargs["model_id"] == "test/input-model"
        assert call_kwargs["revision"] == "v1.0"

    @patch("shard.__main__.ModelWriter.from_huggingface")
    @patch("shard.__main__.MergeConfig.from_yaml")
    def test_copy_model_with_output_base_model(self, mock_from_yaml, mock_writer, sample_yaml_config, tmp_path):
        """Test copy-model command fallback to output_base_model"""
        from shard.config import MergeModel, MergeConfig

        # Create config without input model
        models = [
            MergeModel(model="test/model1", base="test/base", alpha=0.5),
            MergeModel(model="test/model2", base="test/base", alpha=0.3),
        ]
        config = MergeConfig(
            finetune_merge=models,
            output_base_model="test/base",
            output_dir=str(tmp_path / "output")
        )
        mock_from_yaml.return_value = config

        runner = CliRunner()
        result = runner.invoke(cli, [
            "copy-model",
            str(sample_yaml_config)
        ])

        assert mock_writer.called
        call_kwargs = mock_writer.call_args[1]
        assert call_kwargs["model_id"] == "test/base"

    @patch("shard.__main__.MergeConfig.from_yaml")
    def test_copy_model_error_handling(self, mock_from_yaml, sample_yaml_config):
        """Test copy-model command handles errors gracefully"""
        mock_from_yaml.side_effect = Exception("Test error")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "copy-model",
            str(sample_yaml_config)
        ])

        assert result.exit_code != 0


class TestGenerateCommand:
    """Test suite for generate command"""

    def test_generate_command_help(self):
        """Test generate command help text"""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate text from a model" in result.output
        assert "MODEL_PATH" in result.output
        assert "PROMPT" in result.output

    def test_generate_command_missing_model(self):
        """Test generate with missing model path"""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "/nonexistent/model",
            "Test prompt"
        ])

        assert result.exit_code != 0

    @patch("shard.__main__.InferenceEngine.from_pretrained")
    def test_generate_command_with_model(self, mock_engine_class, tmp_path):
        """Test generate command with valid model"""
        # Create a fake model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Mock the engine
        mock_engine = Mock()
        mock_engine.stream_generation.return_value = ["Hello", " ", "world"]
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            str(model_dir),
            "Test prompt",
            "--max-tokens", "10",
            "--temperature", "0.7"
        ])

        assert mock_engine_class.called
        assert mock_engine.stream_generation.called

        # Check parameters were passed correctly
        call_kwargs = mock_engine.stream_generation.call_args[1]
        assert call_kwargs["max_new_tokens"] == 10
        assert call_kwargs["temperature"] == 0.7

    @patch("shard.__main__.InferenceEngine.from_pretrained")
    def test_generate_command_with_4bit(self, mock_engine_class, tmp_path):
        """Test generate command with 4-bit quantization"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_engine = Mock()
        mock_engine.stream_generation.return_value = ["test"]
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            str(model_dir),
            "Test prompt",
            "--load-in-4bit"
        ])

        # Check 4-bit flag was passed
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["load_in_8bit"] is False

    @patch("shard.__main__.InferenceEngine.from_pretrained")
    def test_generate_command_with_8bit(self, mock_engine_class, tmp_path):
        """Test generate command with 8-bit quantization"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_engine = Mock()
        mock_engine.stream_generation.return_value = ["test"]
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            str(model_dir),
            "Test prompt",
            "--load-in-8bit"
        ])

        # Check 8-bit flag was passed
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs["load_in_4bit"] is False
        assert call_kwargs["load_in_8bit"] is True

    @patch("shard.__main__.InferenceEngine.from_pretrained")
    def test_generate_command_with_device(self, mock_engine_class, tmp_path):
        """Test generate command with custom device"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_engine = Mock()
        mock_engine.stream_generation.return_value = ["test"]
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            str(model_dir),
            "Test prompt",
            "--device", "cuda"
        ])

        # Check device was passed
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs["device"] == "cuda"

    @patch("shard.__main__.InferenceEngine.from_pretrained")
    def test_generate_command_custom_parameters(self, mock_engine_class, tmp_path):
        """Test generate command with custom generation parameters"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_engine = Mock()
        mock_engine.stream_generation.return_value = ["test"]
        mock_engine_class.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            str(model_dir),
            "Test prompt",
            "--max-tokens", "256",
            "--temperature", "0.9",
            "--top-p", "0.8",
            "--top-k", "50",
            "--repetition-penalty", "1.2"
        ])

        # Check all parameters were passed
        call_kwargs = mock_engine.stream_generation.call_args[1]
        assert call_kwargs["max_new_tokens"] == 256
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["repetition_penalty"] == 1.2

    @patch("shard.__main__.InferenceEngine.from_pretrained")
    def test_generate_command_error_handling(self, mock_engine_class, tmp_path):
        """Test generate command handles errors gracefully"""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_engine_class.side_effect = Exception("Model loading error")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            str(model_dir),
            "Test prompt"
        ])

        assert result.exit_code != 0


class TestRunMerge:
    """Test suite for run_merge function"""

    @patch("shard.__main__.FourierMerge")
    @patch("shard.__main__.HFMultiModelIndex")
    @patch("shard.__main__.DownloadManager")
    async def test_run_merge_basic(self, mock_dm_class, mock_index_class, mock_merger_class, mock_merge_config):
        """Test run_merge executes merge operation"""
        from shard.__main__ import run_merge

        mock_dm = AsyncMock()
        mock_dm_class.return_value = mock_dm
        mock_dm.progress_callbacks = []

        mock_index = Mock()
        mock_index_class.return_value = mock_index

        mock_merger = AsyncMock()
        mock_merger_class.return_value = mock_merger

        await run_merge(
            config=mock_merge_config,
            device="cpu",
            clean_cache=False
        )

        assert mock_merger.merge.called
        assert not mock_dm.cleanup.called

    @patch("shard.__main__.FourierMerge")
    @patch("shard.__main__.HFMultiModelIndex")
    @patch("shard.__main__.DownloadManager")
    async def test_run_merge_with_cleanup(self, mock_dm_class, mock_index_class, mock_merger_class, mock_merge_config):
        """Test run_merge with clean_cache=True calls cleanup"""
        from shard.__main__ import run_merge

        mock_dm = AsyncMock()
        mock_dm_class.return_value = mock_dm
        mock_dm.progress_callbacks = []

        mock_index = Mock()
        mock_index_class.return_value = mock_index

        mock_merger = AsyncMock()
        mock_merger_class.return_value = mock_merger

        await run_merge(
            config=mock_merge_config,
            device="cpu",
            clean_cache=True
        )

        assert mock_merger.merge.called
        assert mock_dm.cleanup.called
