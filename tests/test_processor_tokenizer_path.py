import json

import pytest

from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_common import _read_tokenizer_path


def test_read_tokenizer_path_uses_sft_preprocessor_config(tmp_path):
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"registry_name": "to_batch_processor", "config": {}},
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": "local/tokenizer"},
                    },
                ]
            }
        )
    )

    assert _read_tokenizer_path(str(tmp_path)) == "local/tokenizer"


def test_read_tokenizer_path_requires_tokenizer_step(tmp_path):
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps({"steps": []}))

    with pytest.raises(ValueError, match="tokenizer_processor"):
        _read_tokenizer_path(str(tmp_path))
