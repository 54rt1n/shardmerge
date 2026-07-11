import json

from click.testing import CliRunner
import torch
import yaml

from shard.exlora.nvfp4_residual import NVFP4ResidualWriter, build_nvfp4_residual_entry
from shard.exlora.peft import AdapterAccumulator
from shard.loramerge.config import LoraMergeConfig
from shard.loramerge.scan import cli, report_to_markdown, scan_config


LAYER = "model.layers.0.self_attn.q_proj.weight"


def write_yaml(tmp_path, data):
    path = tmp_path / "loramerge.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def write_index_model(path, tensor_names):
    path.mkdir(parents=True)
    weight_map = {name: "model-00001.safetensors" for name in tensor_names}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return path


def write_peft(path):
    acc = AdapterAccumulator(base_model="base", target_modules=["q_proj"])
    acc.add(LAYER, torch.ones(2, 4), torch.ones(3, 2), rank=2)
    acc.write(path)
    return path


def write_nvfp4(path):
    torch.manual_seed(4)
    delta = torch.randn(4, 8)
    entry = build_nvfp4_residual_entry(delta, extra_bpe=1.0, index_bits=32)
    writer = NVFP4ResidualWriter(
        path,
        base_model="base",
        finetune_model="dense",
        target_modules=["q_proj"],
        extra_bpe=1.0,
        residual_dtype=torch.bfloat16,
        index_bits=32,
    )
    writer.add(LAYER, entry)
    writer.write_manifest()
    return path


def test_scan_config_reports_sources_and_io_routes(tmp_path):
    storage = tmp_path / "models"
    base = write_index_model(storage / "base", [
        "model.embed_tokens.weight",
        LAYER,
        "lm_head.weight",
    ])
    qat = write_index_model(storage / "qat", ["model.embed_tokens.weight"])
    head = write_index_model(storage / "head", ["lm_head.weight"])
    peft = write_peft(storage / "adapter")
    nvfp4 = write_nvfp4(storage / "dense")
    cfg_file = write_yaml(tmp_path, {
        "base_model": str(base),
        "output_dir": str(tmp_path / "out"),
        "storage_dir": str(storage),
        "sources": [
            {"model": str(qat), "kind": "model", "is_input": True, "name": "qat-input"},
            {"model": str(head), "kind": "model", "is_output": True, "name": "head-output"},
            {"path": str(peft), "kind": "peft", "weight": 1.5, "name": "adapter"},
            {"path": str(nvfp4), "kind": "nvfp4_residual", "name": "dense"},
        ],
    })

    report = scan_config(LoraMergeConfig.from_yaml(cfg_file), config_path=cfg_file)

    assert report.embedding_model == str(qat)
    assert report.lm_head_model == str(head)
    assert report.base_checkpoint.details["tensor_count"] == 3
    by_name = {source.name: source for source in report.sources}
    assert by_name["qat-input"].role == "embedding"
    assert by_name["head-output"].role == "lm_head"
    assert by_name["adapter"].representation == "peft_lora_adapter"
    assert by_name["adapter"].enabled is True
    assert by_name["adapter"].details["rank_histogram"] == {"2": 1}
    assert by_name["adapter"].details["module_counts"] == {"q_proj": 1}
    assert by_name["dense"].representation == "dense_nvfp4_residual"
    assert by_name["dense"].details["tensor_count"] == 1


def test_report_to_markdown_includes_summary(tmp_path):
    storage = tmp_path / "models"
    base = write_index_model(storage / "base", ["model.embed_tokens.weight", "lm_head.weight"])
    peft = write_peft(storage / "adapter")
    cfg_file = write_yaml(tmp_path, {
        "base_model": str(base),
        "output_dir": str(tmp_path / "out"),
        "storage_dir": str(storage),
        "sources": [{"path": str(peft), "kind": "peft", "name": "adapter"}],
    })
    report = scan_config(LoraMergeConfig.from_yaml(cfg_file), config_path=cfg_file)

    rendered = report_to_markdown(report)

    assert "# LoRA Merge Scan Report" in rendered
    assert "| adapter | true | delta | peft_lora_adapter | 1 | ok |" in rendered
    assert "`embed_tokens`" in rendered


def test_scan_cli_emits_json(tmp_path):
    storage = tmp_path / "models"
    base = write_index_model(storage / "base", ["model.embed_tokens.weight", "lm_head.weight"])
    peft = write_peft(storage / "adapter")
    cfg_file = write_yaml(tmp_path, {
        "base_model": str(base),
        "output_dir": str(tmp_path / "out"),
        "storage_dir": str(storage),
        "sources": [{"path": str(peft), "kind": "peft", "name": "adapter"}],
    })

    result = CliRunner().invoke(cli, ["--config", str(cfg_file), "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["sources"][0]["name"] == "adapter"
    assert payload["sources"][0]["representation"] == "peft_lora_adapter"
