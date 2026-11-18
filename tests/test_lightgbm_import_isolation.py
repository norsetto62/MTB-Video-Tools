"""Regression tests ensuring the LightGBM runtime has no PyTorch dependency."""

from __future__ import annotations

import subprocess
import sys


def test_lightgbm_modules_import_without_torch():
    script = r"""
import builtins

real_import = builtins.__import__

def reject_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"Unexpected PyTorch import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_torch

import autocut.ml.lightgbm_model
import autocut.ml.lightgbm_checkpoint
import autocut.ml.lightgbm_fit
import autocut.ml.lightgbm_inference
import autocut.ml.lightgbm_training
import autocut.pipeline.lightgbm_training
import autocut.pipeline.lightgbm_hpo
import autocut.pipeline.run
"""
    subprocess.run([sys.executable, "-c", script], check=True)
