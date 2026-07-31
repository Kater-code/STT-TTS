"""ru2tl — пайплайн «русская речь → тагальская речь» на облачных API."""

from .config import Config
from .models import PipelineResult, Segment
from .pipeline import SpeechPipeline

__all__ = ["Config", "Segment", "PipelineResult", "SpeechPipeline"]
__version__ = "0.1.0"
