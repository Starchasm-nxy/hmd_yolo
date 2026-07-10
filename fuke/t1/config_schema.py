"""
Configuration schema and loader for the detection pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import yaml


@dataclass
class CameraConfig:
    width: int = 848
    height: int = 480
    fps: int = 30


@dataclass
class ModelConfig:
    path: str = "/home/fu/weights/tong_blue_v0.pt"
    device: str = "cpu"
    iou: float = 0.45
    imgsz_step: int = 32


@dataclass
class InferenceConfig:
    locked_conf: float = 0.5
    unlocked_1m_imgsz: int = 640
    unlocked_1m_conf: float = 0.5
    unlocked_2m_imgsz: int = 640
    unlocked_2m_conf: float = 0.5
    max_area: int = 100000


@dataclass
class LockTrackerConfig:
    max_hit: int = 15
    max_miss: int = 7
    search_ratio: float = 2.5
    min_search_radius: int = 130
    max_search_radius: int = 270


@dataclass
class FrameSkipConfig:
    enabled: bool = False
    n: int = 2


@dataclass
class HistoryConfig:
    clear_enabled: bool = True
    clear_timeout: float = 10.0


@dataclass
class DisplayConfig:
    window_name: str = "d435"


@dataclass
class FilesConfig:
    data: str = "data.txt"
    output: str = "gaozhi.txt"
    clear_on_start: List[str] = field(default_factory=lambda: ["data.txt", "gaozhi.txt"])


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    lock_tracker: LockTrackerConfig = field(default_factory=LockTrackerConfig)
    frame_skip: FrameSkipConfig = field(default_factory=FrameSkipConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str) -> Config:
    """Load configuration from a YAML file, returning a Config dataclass."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if 'camera' in raw:
        cfg.camera = CameraConfig(**raw['camera'])

    if 'model' in raw:
        cfg.model = ModelConfig(**raw['model'])

    if 'inference' in raw:
        inf = raw['inference']
        cfg.inference = InferenceConfig(
            locked_conf=inf.get('locked', {}).get('conf', 0.5),
            unlocked_1m_imgsz=inf.get('unlocked', {}).get('1m', {}).get('imgsz', 640),
            unlocked_1m_conf=inf.get('unlocked', {}).get('1m', {}).get('conf', 0.5),
            unlocked_2m_imgsz=inf.get('unlocked', {}).get('2m', {}).get('imgsz', 640),
            unlocked_2m_conf=inf.get('unlocked', {}).get('2m', {}).get('conf', 0.5),
            max_area=inf.get('max_area', 100000),
        )

    if 'lock_tracker' in raw:
        cfg.lock_tracker = LockTrackerConfig(**raw['lock_tracker'])

    if 'frame_skip' in raw:
        cfg.frame_skip = FrameSkipConfig(**raw['frame_skip'])

    if 'history' in raw:
        cfg.history = HistoryConfig(**raw['history'])

    if 'display' in raw:
        cfg.display = DisplayConfig(**raw['display'])

    if 'files' in raw:
        cfg.files = FilesConfig(**raw['files'])

    if 'logging' in raw:
        cfg.logging = LoggingConfig(**raw['logging'])

    return cfg
