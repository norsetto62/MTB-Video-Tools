from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    output_dir: Path = Path("output")
    log_level: str = "INFO"