"""Dollar pricing for every token the benchmark spends.

Prices are the **publicly documented list rates** for each vendor, so what the
report shows is a list-price equivalent: what these tokens would cost at public
rates. A negotiated contract, a self-hosted deployment, or a gateway that marks
up or absorbs cost will bill something different — the point is a like-for-like
comparison between the arms, not an invoice.

Sources (fetched 2026-09-18):

* **System 1 — TypeSafe (Jev)**: https://typesafe.ai quotes
  *"$42 Per Billion input tokens"* — i.e. **$0.042 per million input tokens** —
  alongside *"238x Lower input price than Claude Fable 5.1"*, which checks out
  against Fable 5.1's $10/MTok input. TypeSafe publishes **no separate output
  price**, so this module prices Jev's output at the same rate and flags the
  assumption. Output is ~10% of Jev's tokens here, so the assumption moves the
  Jev figure by a few percent at most — see ``JEV.output_is_assumed``.

* **System 2 — Anthropic**: https://platform.claude.com/docs/en/about-claude/pricing
  Claude Sonnet 4.5 (``claude-sonnet-4-5-20250929``) **$3 / $15** per MTok
  input/output; Claude Opus 4.5 (``claude-opus-4-5-20251101``) **$5 / $25**.
  (Cache reads are 0.1x input and the Batch API is 50% off; this pipeline uses
  neither, so every System 2 token here is billed at the base rate.)

* **System 2 — Qwen 3.8 27B**: self-hosted on a Union-hosted
  ``g6e.2xlarge``. A model you host has no per-token list price — it has an
  hourly one — so its $/MTok is *derived* from the instance rate and the
  deployment's throughput. :class:`SelfHostedPrice` does that derivation and
  every input is an overridable assumption; see its docstring for the full
  arithmetic and the utilization caveat, which dominates everything else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """List price in USD per million tokens."""

    label: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    source: str
    output_is_assumed: bool = False  # vendor publishes no separate output price

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok) / 1_000_000


# --------------------------------------------------------------------------- #
# System 1 — TypeSafe (Jev)                                                   #
# --------------------------------------------------------------------------- #
JEV = Price(
    label="TypeSafe (Jev)",
    input_usd_per_mtok=0.042,  # "$42 Per Billion input tokens"
    output_usd_per_mtok=0.042,  # no published output price — assumed equal to input
    source="https://typesafe.ai",
    output_is_assumed=True,
)


# --------------------------------------------------------------------------- #
# Self-hosted models: an hourly bill, converted to a per-token one            #
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    return float(os.environ[name]) if os.environ.get(name) else default


@dataclass(frozen=True)
class SelfHostedPrice:
    """Derive $/MTok for a model you host from what the box costs per hour.

    A hosted API bills per token; an instance bills per second whether or not it
    is doing anything. So the conversion is::

        $/token = ($/hour ÷ 3600) ÷ (tokens/second) ÷ utilization

    and it needs three assumptions, each independently overridable:

    ``usd_per_hour``
        The instance list price. ``g6e.2xlarge`` is $2.242/hr on-demand
        (1x L40S 48 GB, 8 vCPU, 64 GiB). Committed use cuts it hard — $1.413
        1-year, $0.969 3-year, $2.177 spot — and a Union-hosted deployment may
        bill differently again, so this is the number most worth replacing with
        your real one.

    ``prefill_tokens_per_second`` (input)
        Prefill is compute-bound at ~2 FLOPs per parameter per token. A 27B
        model is 54 GFLOP/token; the L40S does 733 TFLOPS dense FP8, and at a
        realistic 30% MFU that is ~220 TFLOPS -> **~4,000 tok/s**.

    ``decode_tokens_per_second`` (output)
        Decode is memory-bound: every token reads the whole weight set. At FP8
        the 27B weights are 27 GB (bf16 would be 54 GB and would not fit in 48
        GB at all), and the L40S has 864 GB/s, so a *single* stream gets
        864/27 = ~32 tok/s. Continuous batching amortises that weight read
        across the batch, so aggregate throughput scales with batch size —
        ~256 tok/s at the memory-bound ideal for a batch of 8. **200 tok/s** is
        the assumption here: batched, with headroom for KV-cache traffic.

    ``utilization``
        The one that dominates. The instance bills wall-clock, so the per-token
        cost is the saturated figure divided by how busy you actually keep it.
        The default of 1.0 is a **lower bound** — a dedicated endpoint serving
        bursty traffic at 10% utilization costs 10x the number below. This is
        the structural difference from a hosted API, where idle time is free.
    """

    label: str
    usd_per_hour: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    utilization: float = 1.0
    source: str = ""

    @property
    def usd_per_second(self) -> float:
        return self.usd_per_hour / 3600

    def as_price(self) -> Price:
        """The equivalent per-token price, for apples-to-apples with a hosted API."""
        per_mtok = 1_000_000 / (self.utilization or 1.0)
        return Price(
            label=self.label,
            input_usd_per_mtok=self.usd_per_second / self.prefill_tokens_per_second * per_mtok,
            output_usd_per_mtok=self.usd_per_second / self.decode_tokens_per_second * per_mtok,
            source=self.source,
        )

    def derivation(self) -> str:
        """One line showing where the numbers came from, for the report."""
        price = self.as_price()
        return (
            f"${self.usd_per_hour:g}/hr ÷ 3600 ÷ "
            f"{self.prefill_tokens_per_second:,.0f} tok/s prefill (input) and "
            f"{self.decode_tokens_per_second:,.0f} tok/s batched decode (output)"
            + (f", ÷ {self.utilization:.0%} utilization" if self.utilization != 1.0 else "")
            + f" = ${price.input_usd_per_mtok:.3f} in / ${price.output_usd_per_mtok:.2f} out per MTok"
        )


# Qwen 3.8 27B on a Union-hosted g6e.2xlarge (1x NVIDIA L40S 48 GB).
QWEN_DEPLOYMENT = SelfHostedPrice(
    label="Qwen 3.8 27B (self-hosted, g6e.2xlarge)",
    usd_per_hour=_env_float("QWEN_USD_PER_HOUR", 2.242),
    prefill_tokens_per_second=_env_float("QWEN_PREFILL_TOK_S", 4_000),
    decode_tokens_per_second=_env_float("QWEN_DECODE_TOK_S", 200),
    utilization=_env_float("QWEN_UTILIZATION", 1.0),
    source="g6e.2xlarge on-demand $2.242/hr; L40S 864 GB/s, 733 TFLOPS dense FP8",
)

# --------------------------------------------------------------------------- #
# System 2 — one entry per provider key in _config.SYSTEM2_PROVIDERS          #
# --------------------------------------------------------------------------- #

SYSTEM2_PRICES: dict[str, Price | None] = {
    "sonnet": Price(
        label="Claude Sonnet 4.5",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        source="https://platform.claude.com/docs/en/about-claude/pricing",
    ),
    "opus": Price(
        label="Claude Opus 4.5",
        input_usd_per_mtok=5.0,
        output_usd_per_mtok=25.0,
        source="https://platform.claude.com/docs/en/about-claude/pricing",
    ),
    "qwen": QWEN_DEPLOYMENT.as_price(),
}


# Providers whose price is derived from an hourly instance rate rather than
# published per-token — the report labels these so the assumption is visible.
SELF_HOSTED: dict[str, SelfHostedPrice] = {"qwen": QWEN_DEPLOYMENT}


def system2_price(provider: str) -> Price | None:
    """Price for a System 2 provider, or None when it is unpriced."""
    return SYSTEM2_PRICES.get(provider)


def jev_cost(input_tokens: int, output_tokens: int) -> float:
    return JEV.cost(input_tokens, output_tokens)


def system2_cost(provider: str, input_tokens: int, output_tokens: int) -> float | None:
    """None when the provider has no published or configured price."""
    price = system2_price(provider)
    return None if price is None else price.cost(input_tokens, output_tokens)


def unit_costs(result) -> dict:
    """Cost breakdown for one benchmark unit, in USD.

    ``system2`` (and therefore ``total``) is ``None`` for an unpriced provider —
    the report renders that as "n/a", never as zero, so a self-hosted arm is
    never made to look free.
    """
    jev = jev_cost(result.jev_input_tokens, result.jev_output_tokens)
    judge = jev_cost(result.judge_input_tokens, result.judge_output_tokens)
    s2 = system2_cost(result.provider, result.s2_input_tokens, result.s2_output_tokens)
    return {
        "system1": jev,
        "system2": s2,
        "judge": judge,
        # What the pipeline itself costs to run, excluding the benchmark's own judge.
        "pipeline": None if s2 is None else jev + s2,
        "total": None if s2 is None else jev + s2 + judge,
    }
