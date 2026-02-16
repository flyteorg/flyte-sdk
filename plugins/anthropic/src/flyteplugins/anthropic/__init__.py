"""Anthropic Claude plugin for Flyte.

This plugin provides integration between Flyte tasks and Anthropic's Claude API,
enabling you to use Flyte tasks as tools for Claude agents.
"""

from .agents import Agent, function_tool, run_agent

__all__ = ["Agent", "function_tool", "run_agent"]
