"""Kernel adapter interfaces and MVP implementations."""

from .base import AdapterResult, KernelAdapter
from .mock import MockKernelAdapter

__all__ = ["AdapterResult", "KernelAdapter", "MockKernelAdapter"]
