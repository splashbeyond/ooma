"""LLM analyst: reads strategy setups + module findings + positions + notebook, writes a Thesis.

Also writes post-mortems (via learning.cycle.Reflector). Pluggable provider. Milestone 6.
"""
from __future__ import annotations

from .schema import Thesis


class Analyst:
    def __init__(self, provider, prompt_path: str = "prompt.md"):
        self.provider, self.prompt_path = provider, prompt_path

    def write_thesis(self, setups, findings, positions, notes) -> Thesis:
        raise NotImplementedError
