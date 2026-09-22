import shutil

import pytest

from qavs.__main__ import checkpoint_identity, main


def test_checkpoint_identity_detects_copies_and_changed_weights(tmp_path):
    first = tmp_path / "generator"
    first.mkdir()
    (first / "config.json").write_text('{"model_type":"internvl_chat"}')
    (first / "model.safetensors").write_bytes(b"frozen weights")
    second = tmp_path / "verifier"
    shutil.copytree(first, second)
    assert checkpoint_identity(first) == checkpoint_identity(second)
    (second / "model.safetensors").write_bytes(b"different frozen weights")
    assert checkpoint_identity(first) != checkpoint_identity(second)


def test_cli_help_needs_no_model_checkpoints(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "--question" in capsys.readouterr().out
