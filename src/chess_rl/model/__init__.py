"""Optional PyTorch model operations; importing the adapter does not import these."""

from .gab import CompactGAB, SharedTemplateBank

__all__ = ["CompactGAB", "SharedTemplateBank"]
