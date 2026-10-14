from pathlib import Path

import pytest

from autocut.config_loader import load_run_config, load_training_config


def test_load_run_config_maps_existing_dataclasses(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        """
paths:
  data_dir: datafiles
  models_dir: trained
  output_dir: rendered
  cache_dir: cache
video:
  sampling_fps: 2
  processing_width: 640
features:
  window_duration: 4
  window_stride: 2
model:
  checkpoint: trained/model.pt
  inference_batch_size: 64
selection:
  min_clip: 3
  max_clip_duration: 12
  junk_threshold: 0.6
  action_threshold: 0.4
  score_threshold: 0.7
  min_candidate_duration: 5
  max_gap_to_merge: 2
  padding_seconds: 1
audio:
  sync_mode: beat
  max_clip_duration: 12
  fade_duration: 1.5
logging:
  level: DEBUG
""",
        encoding="utf-8",
    )

    cfg = load_run_config(path)
    assert cfg.common.data_dir == Path("datafiles")
    assert cfg.common.models_dir == Path("trained")
    assert cfg.common.output_dir == Path("rendered")
    assert cfg.common.window_duration == 4
    assert cfg.sample_fps == 2
    assert cfg.processing_width == 640
    assert cfg.checkpoint == Path("trained/model.pt")
    assert cfg.inference_batch_size == 64
    assert cfg.max_clip_duration == 12
    assert cfg.candidates.junk_threshold == 0.6
    # min_clip is the only final candidate-duration filter; the legacy
    # min_candidate_duration key is accepted but does not override it.
    assert cfg.candidates.min_candidate_duration == 3
    assert cfg.candidates.flow_threshold == 0.4
    assert cfg.audio.sync_mode.value == "beat"
    assert cfg.audio.fade_duration == 1.5
    assert cfg.common.log_level == "DEBUG"


def test_load_run_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("selection:\n  typo: 1\n")
    with pytest.raises(ValueError, match="Unknown field"):
        load_run_config(path)



def test_load_run_config_rejects_removed_target_duration(tmp_path):
    path = tmp_path / "obsolete.yaml"
    path.write_text("selection:\n  target_duration: 120\n")
    with pytest.raises(ValueError, match="Unknown field"):
        load_run_config(path)

def test_load_run_config_rejects_nonpositive_sampling_rate(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("video:\n  sampling_fps: 0\nselection:\n  min_clip: 3\n")
    with pytest.raises(ValueError, match="sampling_fps"):
        load_run_config(path)


def test_load_training_config_maps_training_and_disjoint_groups(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        """
data:
  manifest: data/manifest.yaml
  train_groups: [ascoli, castelli]
  validation_groups: [mentorella]
  sampling_fps: 2
  processing_width: 480
  window_duration: 4
  window_stride: 2
training:
  epochs: 8
  lr: 0.002
  batch_size: 16
  hidden_dim: 96
  classifier_mid_dim: 32
  dropout: 0.2
  num_classes: 3
  use_scaling: true
  patience: 4
  min_delta: 0.0002
  weight_decay: 0.003
  seed: 7
  class_weights: [0.3, 1.0, 5.0]
experiment:
  name: trial-01
  checkpoint_dir: models/trials
  results_dir: output/trials
""",
        encoding="utf-8",
    )
    cfg = load_training_config(path)
    assert cfg.manifest == Path("data/manifest.yaml")
    assert cfg.train_groups == ("ascoli", "castelli")
    assert cfg.validation_groups == ("mentorella",)
    assert cfg.training.epochs == 8
    assert cfg.training.class_weights == (0.3, 1.0, 5.0)
    assert cfg.experiment_name == "trial-01"
    assert cfg.checkpoint_dir == Path("models/trials")


def test_training_config_rejects_overlapping_groups(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        "data:\n  train_groups: [ascoli]\n  validation_groups: [ascoli]\n"
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        load_training_config(path)


def test_training_config_requires_explicit_groups(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("data:\n  train_groups: []\n  validation_groups: [ascoli]\n")
    with pytest.raises(ValueError, match="non-empty YAML list"):
        load_training_config(path)
