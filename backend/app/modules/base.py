"""Detection module protocol and event dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


@dataclass
class DetectionEvent:
    label: str
    module_id: str
    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    frame: Optional[np.ndarray] = None


@dataclass
class CameraContext:
    key: str
    name: str
    camera_id: str
    device: str
    modules: list[str]
    state: dict[str, Any] = field(default_factory=dict)


class DetectionModule(Protocol):
    id: str
    labels: list[str]

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None: ...

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]: ...

    def unload(self) -> None: ...
