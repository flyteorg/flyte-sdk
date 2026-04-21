"""Shared datatypes for autoresearch agent task outputs and reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetProfile:
    n_parquet_files: int
    parquet_files: list[str]
    has_tokenizer_pkl: bool
    has_token_bytes: bool
    data_bytes: int
    tokenizer_bytes: int


@dataclass
class AutoresearchMetrics:
    val_metric: float
    model_name: str
    notes: str


@dataclass
class HistoryEntry:
    round: int
    code: str
    title: str | None = None
    metrics: AutoresearchMetrics | None = None
    resources: str | None = None
    oom: bool | None = None
    error: str | None = None


@dataclass
class AutoresearchOutput:
    research_topic: str | None
    literature_excerpt: str | None
    dataset_profile: DatasetProfile
    best: HistoryEntry | None
    history: list[HistoryEntry]
