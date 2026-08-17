"""Best-of-N proxy for OpenAI-compatible backends."""

from bon_proxy.app import create_app
from bon_proxy.config import AppConfig, load_config

__all__ = ["AppConfig", "create_app", "load_config"]
__version__ = "2.0.0"
