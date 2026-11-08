"""Resolve the configuration of a run: dataset -> profile -> parameters -> memory check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wvbench import dataset as ds
from wvbench import sizing
from wvbench.settings import ConfigError, ProfileConfig, Settings, manual_profile, merge, parse_params


class OversizeError(ConfigError):
    """The configuration does not fit the memory budget and the policy rejects it."""


@dataclass
class Resolved:
    info: ds.DatasetInfo
    settings: Settings
    estimate: sizing.Estimate
    notes: list[str]

    @property
    def index(self) -> dict[str, Any]:
        return self.settings.index

    def section(self, name: str) -> dict[str, Any]:
        return self.settings.section(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.info.as_dict(),
            **self.settings.as_dict(),
            "estimate": self.estimate.as_dict(),
            "notes": self.notes,
        }


def resolve(
    dataset_name: str,
    profile: str | None = None,
    params: list[str] | None = None,
    allow_oversize: bool = False,
    require_file: bool = True,
    pc: ProfileConfig | None = None,
    info: ds.DatasetInfo | None = None,
) -> Resolved:
    pc = pc or ProfileConfig()
    info = info or ds.info(dataset_name, require_file=require_file)
    cli = parse_params(params)
    notes: list[str] = []

    chosen = manual_profile(pc, dataset_name, profile)
    manual = chosen is not None
    if manual:
        name, reason = chosen
    else:
        name, reason = sizing.select_profile(pc.rules, info.rows, info.dim)

    def build(profile_name: str, why: str) -> tuple[Settings, sizing.Estimate]:
        st = merge(pc, dataset_name, profile_name, why, cli)
        return st, sizing.estimate(info.rows, info.dim, st.index, st.section("estimator"))

    if name is not None:
        st, est = build(name, reason)
        if est.fits:
            return Resolved(info, st, est, notes)
        notes.append(f"profile {name!r}: {est.summary()}")
    else:
        notes.append(reason)
        st, est = None, None

    # Oversize handling. Policy is read from the merged sections (can be set via --param oversize.policy=...).
    sections = (st or merge(pc, dataset_name, next(iter(pc.profiles)), "", cli)).sections
    policy = sections["oversize"].get("policy", "reject")
    if manual:
        # A manual choice is never replaced; it only needs --allow-oversize.
        if allow_oversize:
            notes.append("--allow-oversize: using the manual profile despite the estimate")
            return Resolved(info, st, est, notes)
        raise OversizeError(f"{dataset_name}: {notes[-1]}. Use --allow-oversize to run it anyway.")

    if policy == "fallback":
        fb = sections["oversize"].get("fallback")
        st, est = build(fb, f"fallback (oversize.policy=fallback) after: {reason}")
        notes.append(f"fallback profile {fb!r}: {est.summary()}")
        if est.fits or allow_oversize:
            return Resolved(info, st, est, notes)
        raise OversizeError(f"{dataset_name}: fallback {fb!r} does not fit either. Use --allow-oversize to run it anyway.")
    if policy != "reject":
        raise ConfigError(f"oversize.policy must be 'reject' or 'fallback', got {policy!r}")
    if allow_oversize and st is not None:
        notes.append("--allow-oversize: rejected by the estimate but running anyway")
        return Resolved(info, st, est, notes)
    raise OversizeError(
        f"{dataset_name}: rejected (oversize.policy=reject). {notes[-1]}. "
        "Set oversize.policy=fallback, choose a --profile, or pass --allow-oversize."
    )
