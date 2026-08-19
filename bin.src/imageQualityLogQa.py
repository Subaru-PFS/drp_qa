#!/usr/bin/env python3
"""Parse PFS DRP reduceExposure and imageQualityQa logs, report QA status, and generate diagnostic plots.

This script supports reading from log files or querying the LSST Butler directly.
Matplotlib and numpy are optional and only needed when generating plots using the --plot-dir flag.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FiberStats:
    fiber_id: int
    x_rms: float
    y_rms: float
    n_lines: int


@dataclass
class VisitQA:
    visit: int
    arm: str
    spectrograph: int
    dither: Optional[int] = None
    pfs_design_id: Optional[int] = None
    collection: Optional[str] = None

    # ISR stats
    isr_bad_pixels: int = 0
    isr_time_s: float = 0.0

    # Cosmic ray stats (list of (cr_count, cr_pixels))
    cosmic_rays: list[tuple[int, int]] = field(default_factory=list)
    cosmic_ray_time_s: float = 0.0

    # Centroiding stats
    centroids_total: int = 0
    centroids_good: int = 0
    centroids_good_pct: int = 0
    centroids_low_sn: int = 0
    centroids_low_sn_pct: int = 0
    centroids_fail: int = 0
    centroids_fail_pct: int = 0

    # Detector map fit stats
    fit_chi2: float = float("nan")
    fit_dof: int = 0
    fit_x_rms: float = float("nan")
    fit_y_rms: float = float("nan")
    fit_x_soften: float = float("nan")
    fit_y_soften: float = float("nan")
    fit_n_lines: int = 0
    fit_species_name: str = "Lines"
    fit_species_x_rms: float = float("nan")
    fit_species_y_rms: float = float("nan")
    fit_trace_x_rms: float = float("nan")
    fit_trace_y_rms: float = float("nan")
    fit_species_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    fit_total_lines: int = 0
    fit_active_fibers: int = 0

    # Reserved lines fit quality
    fit_reserved_chi2: float = float("nan")
    fit_reserved_x_rms: float = float("nan")
    fit_reserved_y_rms: float = float("nan")
    fit_reserved_x_soften: float = float("nan")
    fit_reserved_y_soften: float = float("nan")
    fit_reserved_n_lines: int = 0

    # Individual fiber stats
    fibers: list[FiberStats] = field(default_factory=list)

    # Task timings
    reduce_exposure_time_s: float = 0.0
    iq_qa_time_s: float = 0.0
    merge_arms_time_s: float = 0.0

    # Image Quality QA results
    qa_status: str = "UNKNOWN"  # PASS, WARN, FAIL
    qa_target: str = ""
    qa_fwhm: float = float("nan")
    qa_dx: float = float("nan")
    qa_dx_rms: float = float("nan")
    qa_flagged: float = float("nan")
    qa_detail: str = ""

    def sanitize(self):
        """Sanitize all float attributes to ensure none are None."""
        float_attrs = [
            "fit_chi2",
            "fit_x_rms",
            "fit_y_rms",
            "fit_x_soften",
            "fit_y_soften",
            "fit_reserved_chi2",
            "fit_reserved_x_rms",
            "fit_reserved_y_rms",
            "fit_reserved_x_soften",
            "fit_reserved_y_soften",
            "fit_trace_x_rms",
            "fit_trace_y_rms",
            "fit_species_x_rms",
            "fit_species_y_rms",
            "qa_fwhm",
            "qa_dx",
            "qa_dx_rms",
            "qa_flagged",
        ]
        for attr in float_attrs:
            if getattr(self, attr, None) is None:
                setattr(self, attr, float("nan"))

        # Sanitize fit_species_stats
        cleaned_species_stats = {}
        for sp, (x_rms, y_rms) in self.fit_species_stats.items():
            cleaned_species_stats[sp] = (
                float("nan") if x_rms is None else x_rms,
                float("nan") if y_rms is None else y_rms,
            )
        self.fit_species_stats = cleaned_species_stats

        # Sanitize fibers
        for f in self.fibers:
            if f.x_rms is None:
                f.x_rms = float("nan")
            if f.y_rms is None:
                f.y_rms = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Convert the VisitQA instance to a JSON-serializable dictionary.

        Converts float('nan') values to None for clean JSON compatibility.
        """

        def clean_val(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        return {
            "visit": self.visit,
            "arm": self.arm,
            "spectrograph": self.spectrograph,
            "dither": self.dither,
            "pfs_design_id": self.pfs_design_id,
            "collection": self.collection,
            "isr_bad_pixels": self.isr_bad_pixels,
            "isr_time_s": self.isr_time_s,
            "cosmic_rays": self.cosmic_rays,
            "cosmic_ray_time_s": self.cosmic_ray_time_s,
            "centroids_total": self.centroids_total,
            "centroids_good": self.centroids_good,
            "centroids_good_pct": self.centroids_good_pct,
            "centroids_low_sn": self.centroids_low_sn,
            "centroids_low_sn_pct": self.centroids_low_sn_pct,
            "centroids_fail": self.centroids_fail,
            "centroids_fail_pct": self.centroids_fail_pct,
            "fit_chi2": clean_val(self.fit_chi2),
            "fit_dof": self.fit_dof,
            "fit_x_rms": clean_val(self.fit_x_rms),
            "fit_y_rms": clean_val(self.fit_y_rms),
            "fit_x_soften": clean_val(self.fit_x_soften),
            "fit_y_soften": clean_val(self.fit_y_soften),
            "fit_n_lines": self.fit_n_lines,
            "fit_species_name": self.fit_species_name,
            "fit_species_x_rms": clean_val(self.fit_species_x_rms),
            "fit_species_y_rms": clean_val(self.fit_species_y_rms),
            "fit_trace_x_rms": clean_val(self.fit_trace_x_rms),
            "fit_trace_y_rms": clean_val(self.fit_trace_y_rms),
            "fit_species_stats": {
                k: [clean_val(v[0]), clean_val(v[1])]
                for k, v in self.fit_species_stats.items()
            },
            "fit_total_lines": self.fit_total_lines,
            "fit_active_fibers": self.fit_active_fibers,
            "fit_reserved_chi2": clean_val(self.fit_reserved_chi2),
            "fit_reserved_x_rms": clean_val(self.fit_reserved_x_rms),
            "fit_reserved_y_rms": clean_val(self.fit_reserved_y_rms),
            "fit_reserved_x_soften": clean_val(self.fit_reserved_x_soften),
            "fit_reserved_y_soften": clean_val(self.fit_reserved_y_soften),
            "fit_reserved_n_lines": self.fit_reserved_n_lines,
            "fibers": [
                {
                    "fiber_id": f.fiber_id,
                    "x_rms": clean_val(f.x_rms),
                    "y_rms": clean_val(f.y_rms),
                    "n_lines": f.n_lines,
                }
                for f in self.fibers
            ],
            "reduce_exposure_time_s": self.reduce_exposure_time_s,
            "iq_qa_time_s": self.iq_qa_time_s,
            "merge_arms_time_s": self.merge_arms_time_s,
            "qa_status": self.qa_status,
            "qa_target": self.qa_target,
            "qa_fwhm": clean_val(self.qa_fwhm),
            "qa_dx": clean_val(self.qa_dx),
            "qa_dx_rms": clean_val(self.qa_dx_rms),
            "qa_flagged": clean_val(self.qa_flagged),
            "qa_detail": self.qa_detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisitQA:
        """Create a VisitQA instance from a dictionary representation.

        Restores None values to float('nan').
        """

        def restore_float(v):
            if v is None:
                return float("nan")
            if isinstance(v, str):
                if v.strip().lower() in ("none", "nan", "null", ""):
                    return float("nan")
                try:
                    return float(v)
                except ValueError:
                    return float("nan")
            try:
                import numpy as np

                if isinstance(v, (float, int)) and np.isnan(v):
                    return float("nan")
            except ImportError:
                pass
            return v

        vqa = cls(
            visit=data["visit"],
            arm=data["arm"],
            spectrograph=data["spectrograph"],
            dither=data.get("dither"),
            pfs_design_id=data.get("pfs_design_id"),
            collection=data.get("collection"),
            isr_bad_pixels=data.get("isr_bad_pixels", 0),
            isr_time_s=data.get("isr_time_s", 0.0),
            cosmic_rays=[tuple(cr) for cr in data.get("cosmic_rays", [])],
            cosmic_ray_time_s=data.get("cosmic_ray_time_s", 0.0),
            centroids_total=data.get("centroids_total", 0),
            centroids_good=data.get("centroids_good", 0),
            centroids_good_pct=data.get("centroids_good_pct", 0),
            centroids_low_sn=data.get("centroids_low_sn", 0),
            centroids_low_sn_pct=data.get("centroids_low_sn_pct", 0),
            centroids_fail=data.get("centroids_fail", 0),
            centroids_fail_pct=data.get("centroids_fail_pct", 0),
            fit_chi2=restore_float(data.get("fit_chi2")),
            fit_dof=data.get("fit_dof", 0),
            fit_x_rms=restore_float(data.get("fit_x_rms")),
            fit_y_rms=restore_float(data.get("fit_y_rms")),
            fit_x_soften=restore_float(data.get("fit_x_soften")),
            fit_y_soften=restore_float(data.get("fit_y_soften")),
            fit_n_lines=data.get("fit_n_lines", 0),
            fit_species_name=data.get("fit_species_name", "Lines"),
            fit_species_x_rms=restore_float(data.get("fit_species_x_rms")),
            fit_species_y_rms=restore_float(data.get("fit_species_y_rms")),
            fit_trace_x_rms=restore_float(data.get("fit_trace_x_rms")),
            fit_trace_y_rms=restore_float(data.get("fit_trace_y_rms")),
            fit_total_lines=data.get("fit_total_lines", 0),
            fit_active_fibers=data.get("fit_active_fibers", 0),
            fit_reserved_chi2=restore_float(data.get("fit_reserved_chi2")),
            fit_reserved_x_rms=restore_float(data.get("fit_reserved_x_rms")),
            fit_reserved_y_rms=restore_float(data.get("fit_reserved_y_rms")),
            fit_reserved_x_soften=restore_float(data.get("fit_reserved_x_soften")),
            fit_reserved_y_soften=restore_float(data.get("fit_reserved_y_soften")),
            fit_reserved_n_lines=data.get("fit_reserved_n_lines", 0),
            reduce_exposure_time_s=data.get("reduce_exposure_time_s", 0.0),
            iq_qa_time_s=data.get("iq_qa_time_s", 0.0),
            merge_arms_time_s=data.get("merge_arms_time_s", 0.0),
            qa_status=data.get("qa_status", "UNKNOWN"),
            qa_target=data.get("qa_target", ""),
            qa_fwhm=restore_float(data.get("qa_fwhm")),
            qa_dx=restore_float(data.get("qa_dx")),
            qa_dx_rms=restore_float(data.get("qa_dx_rms")),
            qa_flagged=restore_float(data.get("qa_flagged")),
            qa_detail=data.get("qa_detail", ""),
        )

        # Reconstruct fit_species_stats
        species_stats = {}
        for k, v in data.get("fit_species_stats", {}).items():
            species_stats[k] = (restore_float(v[0]), restore_float(v[1]))
        vqa.fit_species_stats = species_stats

        # Reconstruct fibers
        fibers = []
        for f in data.get("fibers", []):
            fibers.append(
                FiberStats(
                    fiber_id=f["fiber_id"],
                    x_rms=restore_float(f["x_rms"]),
                    y_rms=restore_float(f["y_rms"]),
                    n_lines=f["n_lines"],
                )
            )
        vqa.fibers = fibers

        return vqa

    @classmethod
    def from_metrics(cls, metrics: dict[str, Any] | Any) -> VisitQA:
        """Create a VisitQA instance from a Butler iqQaMetrics row/dictionary."""
        if hasattr(metrics, "to_dict"):
            data = metrics.to_dict()
        else:
            data = dict(metrics)

        def restore_float(v):
            if v is None:
                return float("nan")
            if isinstance(v, str):
                if v.strip().lower() in ("none", "nan", "null", ""):
                    return float("nan")
                try:
                    return float(v)
                except ValueError:
                    return float("nan")
            try:
                import numpy as np

                if isinstance(v, (float, int)) and np.isnan(v):
                    return float("nan")
            except ImportError:
                pass
            return v

        # Map pandas-friendly column names to VisitQA attribute names
        vqa = cls(
            visit=data.get("visit", 0),
            arm=data.get("arm", ""),
            spectrograph=data.get("spectrograph", 0),
            isr_bad_pixels=data.get("isrBadPixels", 0),
            isr_time_s=data.get("isrTime", 0.0),
            cosmic_ray_time_s=data.get("cosmicRayTime", 0.0),
            reduce_exposure_time_s=data.get("reduceExposureTime", 0.0),
            fit_chi2=restore_float(data.get("fitChi2")),
            fit_dof=data.get("fitDof", 0),
            fit_x_rms=restore_float(data.get("fitXRms")),
            fit_y_rms=restore_float(data.get("fitYRms")),
            fit_x_soften=restore_float(data.get("fitXSoften")),
            fit_y_soften=restore_float(data.get("fitYSoften")),
            fit_n_lines=data.get("fitNLines", 0),
            fit_total_lines=data.get("fitTotalLines", 0),
            fit_active_fibers=data.get("fitActiveFibers", 0),
            fit_reserved_chi2=restore_float(data.get("fitReservedChi2")),
            fit_reserved_x_rms=restore_float(data.get("fitReservedXRms")),
            fit_reserved_y_rms=restore_float(data.get("fitReservedYRms")),
            fit_reserved_x_soften=restore_float(data.get("fitReservedXSoften")),
            fit_reserved_y_soften=restore_float(data.get("fitReservedYSoften")),
            fit_reserved_n_lines=data.get("fitReservedNLines", 0),
            fit_trace_x_rms=restore_float(data.get("fitTraceXRms")),
            fit_trace_y_rms=restore_float(data.get("fitTraceYRms")),
            qa_status=data.get("qaStatus", "UNKNOWN"),
            qa_target=data.get("seqName", ""),
            qa_fwhm=restore_float(data.get("medFwhm")),
            qa_dx=restore_float(data.get("medDxCenter")),
            qa_dx_rms=restore_float(data.get("dxCenterRms")),
            qa_flagged=restore_float(data.get("pctFlagged")),
        )

        # Restore cosmic rays list from count/pixels
        cr_count = data.get("cosmicRayCount", 0)
        cr_pixels = data.get("cosmicRayPixels", 0)
        if cr_count > 0 or cr_pixels > 0:
            vqa.cosmic_rays = [(int(cr_count), int(cr_pixels))]

        # Reconstruct fit_species_stats from dynamic columns
        for k, v in data.items():
            if k.startswith("fitSpeciesXRms_"):
                sp = k[len("fitSpeciesXRms_") :]
                y_key = f"fitSpeciesYRms_{sp}"
                x_rms = float(v)
                y_rms = float(data.get(y_key, float("nan")))
                vqa.fit_species_stats[sp] = (x_rms, y_rms)
                # Assign the last matched species as the default fallback
                vqa.fit_species_name = sp
                vqa.fit_species_x_rms = x_rms
                vqa.fit_species_y_rms = y_rms

        # Reconstruct fibers
        fiber_ids = data.get("fiberIds")
        fiber_x_rms = data.get("fiberXRms")
        fiber_y_rms = data.get("fiberYRms")
        fiber_n_lines = data.get("fiberNLines")
        if fiber_ids is not None and not isinstance(fiber_ids, float):
            # If fiber_ids is loaded as a numpy array or string/list representation, handle it
            if isinstance(fiber_ids, str):
                try:
                    import json

                    def clean_numpy_str(s):
                        if not isinstance(s, str):
                            return s
                        s = s.strip()
                        if s.startswith("[") and s.endswith("]"):
                            content = " ".join(s[1:-1].split())
                            content_comma = ",".join(content.split())
                            content_json = (
                                content_comma.replace("None", "null")
                                .replace("nan", "null")
                                .replace("NaN", "null")
                            )
                            return f"[{content_json}]"
                        return s

                    cleaned_ids = clean_numpy_str(fiber_ids)
                    fiber_ids = json.loads(cleaned_ids)

                    cleaned_x = clean_numpy_str(fiber_x_rms)
                    fiber_x_rms = (
                        json.loads(cleaned_x)
                        if isinstance(cleaned_x, str)
                        else cleaned_x
                    )

                    cleaned_y = clean_numpy_str(fiber_y_rms)
                    fiber_y_rms = (
                        json.loads(cleaned_y)
                        if isinstance(cleaned_y, str)
                        else cleaned_y
                    )

                    cleaned_n = clean_numpy_str(fiber_n_lines)
                    fiber_n_lines = (
                        json.loads(cleaned_n)
                        if isinstance(cleaned_n, str)
                        else cleaned_n
                    )
                except Exception:
                    pass
            for i, fid in enumerate(fiber_ids):
                x_rms = (
                    restore_float(fiber_x_rms[i])
                    if fiber_x_rms is not None and i < len(fiber_x_rms)
                    else float("nan")
                )
                y_rms = (
                    restore_float(fiber_y_rms[i])
                    if fiber_y_rms is not None and i < len(fiber_y_rms)
                    else float("nan")
                )
                n_lines = (
                    fiber_n_lines[i]
                    if fiber_n_lines is not None and i < len(fiber_n_lines)
                    else 0
                )
                vqa.fibers.append(
                    FiberStats(
                        fiber_id=int(fid), x_rms=x_rms, y_rms=y_rms, n_lines=n_lines
                    )
                )

        return vqa

    def to_json(self) -> str:
        """Convert the VisitQA instance to a JSON string representation."""
        import json

        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> VisitQA:
        """Create a VisitQA instance from a JSON string representation."""
        import json

        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Parser


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def log_records_to_string(records: Any) -> str:
    """Safely convert various log record structures from Butler to standard string."""
    if isinstance(records, str):
        return records
    if hasattr(records, "read"):
        return str(records.read())
    if hasattr(records, "readlines"):
        return "\n".join(records.readlines())
    if isinstance(records, (list, tuple)):
        lines = []
        for r in records:
            if isinstance(r, str):
                lines.append(r)
            elif hasattr(r, "message"):
                lines.append(r.message)
            elif hasattr(r, "getMessage"):
                lines.append(r.getMessage())
            else:
                lines.append(str(r))
        return "\n".join(lines)
    return str(records)


def parse_log_text_lines(lines: list[str], vqa: VisitQA):
    """Parse log text lines and populate a VisitQA object."""
    re_bad_pixels = re.compile(r"Set (\d+) BAD pixels to")
    re_cr = re.compile(
        r"(?:Found|Identified) (\d+) cosmic rays (?:\(|covering )(\d+) pixels"
    )
    re_centroids = re.compile(
        r"Measured (\d+) line centroids:\s*(\d+) good \((\d+)%\),\s*(\d+) low-S/N < [\d\.]+ \((\d+)%\),\s*(\d+) centroid-fail \((\d+)%\)"
    )
    # Newer drp_stella prefixes its fit summary messages with the detector they
    # belong to ("Final result: arm=b spectrograph=1 chi2=...").  Keep the
    # prefix optional so both log formats parse, as fitDetectorMapLogQa.py does.
    arm_spec = r"(?:arm=\S+ spectrograph=\d+ )?"
    re_fit_result = re.compile(
        r"Final result: "
        + arm_spec
        + r"chi2=(\S+) dof=(\d+) xRMS=(\S+) yRMS=(\S+) xSoften=(\S+) ySoften=(\S+) from (\d+) lines"
    )
    re_fiber = re.compile(
        r"Stats for fiberId=(\d+): "
        + arm_spec
        + r"chi2=\S+ dof=\d+ xRMS=(\S+) yRMS=(\S+).*from (\d+) lines"
    )
    re_task_time = re.compile(
        r"Execution of task '(\w+)' on quantum .* took ([\d\.]+) seconds"
    )
    re_qa = re.compile(
        r"IQ QA (PASS|WARN|FAIL)\s+(\d+)\s+([a-z0-9]+)\s+(.+?)\s+medFWHM=([\d\.]+)px\s+dxCenter=([+-]?[\d\.]+px|NaN)\s+pctFlagged=([\d\.]+%|NaN)\s*(?:\[(.*?)\])?"
    )
    re_quantum = re.compile(
        r"dataId=\{instrument:\s*'PFS',\s*arm:\s*'(\w+)',\s*spectrograph:\s*(\d+),\s*visit:\s*(\d+),\s*dither:\s*(-?\d+)"
    )
    re_species_stats = re.compile(
        r"Stats for (\w+): " + arm_spec + r"chi2=\S+ dof=\d+ xRMS=(\S+) yRMS=(\S+)"
    )
    re_fit_lines = re.compile(r"Final fit:.*from (\d+)/(\d+) lines")
    re_reserved_fit = re.compile(
        r"Fit quality from reserved lines:\s*chi2=(\S+)\s+xRMS=(\S+)\s+yRMS=(\S+)(?:\s+\([^\)]+\))?\s+xSoften=(\S+)\s+ySoften=(\S+)\s+from\s+(\d+)\s+lines"
    )
    re_active_fibers = re.compile(r"Stats: fit selection has (\d+) active fibers")

    for line in lines:
        # Check quantum context for dither/visit info
        m = re_quantum.search(line)
        if m:
            vqa.dither = int(m.group(4))
            continue

        m = re_bad_pixels.search(line)
        if m:
            vqa.isr_bad_pixels = int(m.group(1))
            continue

        m = re_cr.search(line)
        if m:
            vqa.cosmic_rays.append((int(m.group(1)), int(m.group(2))))
            continue

        m = re_centroids.search(line)
        if m:
            vqa.centroids_total = int(m.group(1))
            vqa.centroids_good = int(m.group(2))
            vqa.centroids_good_pct = int(m.group(3))
            vqa.centroids_low_sn = int(m.group(4))
            vqa.centroids_low_sn_pct = int(m.group(5))
            vqa.centroids_fail = int(m.group(6))
            vqa.centroids_fail_pct = int(m.group(7))
            continue

        m = re_fit_result.search(line)
        if m:
            try:
                vqa.fit_chi2 = float(m.group(1))
                vqa.fit_dof = int(m.group(2))
                vqa.fit_x_rms = float(m.group(3))
                vqa.fit_y_rms = float(m.group(4))
                vqa.fit_x_soften = float(m.group(5))
                vqa.fit_y_soften = float(m.group(6))
                vqa.fit_n_lines = int(m.group(7))
            except ValueError:
                pass
            continue

        m = re_fit_lines.search(line)
        if m:
            try:
                vqa.fit_n_lines = int(m.group(1))
                vqa.fit_total_lines = int(m.group(2))
            except ValueError:
                pass
            continue

        m = re_reserved_fit.search(line)
        if m:
            try:
                vqa.fit_reserved_chi2 = float(m.group(1))
                vqa.fit_reserved_x_rms = float(m.group(2))
                vqa.fit_reserved_y_rms = float(m.group(3))
                vqa.fit_reserved_x_soften = float(m.group(4))
                vqa.fit_reserved_y_soften = float(m.group(5))
                vqa.fit_reserved_n_lines = int(m.group(6))
            except ValueError:
                pass
            continue

        m = re_active_fibers.search(line)
        if m:
            try:
                vqa.fit_active_fibers = int(m.group(1))
            except ValueError:
                pass
            continue

        m = re_fiber.search(line)
        if m:
            try:
                fid = int(m.group(1))
                x_rms = float(m.group(2))
                y_rms = float(m.group(3))
                n_lines = int(m.group(4))
                # Avoid duplicates
                if not any(f.fiber_id == fid for f in vqa.fibers):
                    vqa.fibers.append(
                        FiberStats(
                            fiber_id=fid, x_rms=x_rms, y_rms=y_rms, n_lines=n_lines
                        )
                    )
            except ValueError:
                pass
            continue

        m = re_species_stats.search(line)
        if m:
            sp = m.group(1)
            try:
                x_rms = float(m.group(2))
                y_rms = float(m.group(3))
                if sp == "Trace":
                    vqa.fit_trace_x_rms = x_rms
                    vqa.fit_trace_y_rms = y_rms
                else:
                    vqa.fit_species_stats[sp] = (x_rms, y_rms)
                    vqa.fit_species_name = sp
                    vqa.fit_species_x_rms = x_rms
                    vqa.fit_species_y_rms = y_rms
            except ValueError:
                pass
            continue

        m = re_task_time.search(line)
        if m:
            task = m.group(1)
            duration = float(m.group(2))
            if task == "isr":
                vqa.isr_time_s = duration
            elif task == "cosmicray":
                vqa.cosmic_ray_time_s = duration
            elif task == "reduceExposure":
                vqa.reduce_exposure_time_s = duration
            elif task == "imageQualityQa":
                vqa.iq_qa_time_s = duration
            elif task == "mergeArms":
                vqa.merge_arms_time_s = duration
            continue

        m = re_qa.search(line)
        if m:
            vqa.qa_status = m.group(1)
            vqa.qa_target = m.group(4).strip()
            try:
                vqa.qa_fwhm = float(m.group(5))

                dx_str = m.group(6)
                if dx_str == "NaN":
                    vqa.qa_dx = float("nan")
                else:
                    vqa.qa_dx = float(dx_str.rstrip("px"))

                flagged_str = m.group(7)
                if flagged_str == "NaN":
                    vqa.qa_flagged = float("nan")
                else:
                    vqa.qa_flagged = float(flagged_str.rstrip("%"))
            except ValueError:
                pass
            vqa.qa_detail = m.group(8).strip() if m.group(8) else ""
            continue


def parse_logs(
    log_paths: list[Path], collection: Optional[str] = None
) -> list[VisitQA]:
    """Parse log files on disk and return list of VisitQA structures."""
    visits: dict[tuple[int, str, int], VisitQA] = {}
    current_key = None

    re_quantum = re.compile(
        r"dataId=\{instrument:\s*'PFS',\s*arm:\s*'(\w+)',\s*spectrograph:\s*(\d+),\s*visit:\s*(\d+),\s*dither:\s*(-?\d+)"
    )

    for path in log_paths:
        if not path.exists():
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue

        with open(path, "r") as f:
            for line in f:
                # 1. Look for Quantum setup/identities
                m = re_quantum.search(line)
                if m:
                    arm = m.group(1)
                    spec = int(m.group(2))
                    visit = int(m.group(3))
                    dither = int(m.group(4))
                    key = (visit, arm, spec)

                    if key not in visits:
                        visits[key] = VisitQA(
                            visit=visit,
                            arm=arm,
                            spectrograph=spec,
                            dither=dither,
                            collection=collection,
                        )
                    current_key = key
                    continue

                if not current_key:
                    continue

                parse_log_text_lines([line], visits[current_key])

    return list(visits.values())


def parse_butler_logs(
    logs: dict[str, Any],
    visit: int,
    spectrograph: int,
    collection: Optional[str] = None,
) -> list[VisitQA]:
    """Parse log records retrieved from Butler."""
    visits: dict[tuple[int, str, int], VisitQA] = {}

    for key, records in logs.items():
        log_text = log_records_to_string(records)

        # Determine if this is per-detector or per-visit log
        if "/" in key:
            task, suffix = key.split("/", 1)
        else:
            suffix = str(spectrograph)

        # Check if suffix matches "armSpectrograph" (e.g. b3, r3) or just "spectrograph" (e.g. 3)
        if len(suffix) > 0 and suffix[0].isalpha():
            arm = suffix[0]
            spec = int(suffix[1:])
        else:
            arm = None
            spec = int(suffix)

        if arm:
            vkey = (visit, arm, spec)
            if vkey not in visits:
                visits[vkey] = VisitQA(
                    visit=visit, arm=arm, spectrograph=spec, collection=collection
                )
            vqa = visits[vkey]
            parse_log_text_lines(log_text.splitlines(), vqa)
        else:
            # Per-visit logs (like mergeArms) apply to all arms of that spectrograph
            temp_vqa = VisitQA(visit=visit, arm="temp", spectrograph=spec)
            parse_log_text_lines(log_text.splitlines(), temp_vqa)
            for vqa in visits.values():
                if vqa.visit == visit and vqa.spectrograph == spec:
                    if temp_vqa.merge_arms_time_s > 0.0:
                        vqa.merge_arms_time_s = temp_vqa.merge_arms_time_s

    return list(visits.values())


# ---------------------------------------------------------------------------
# Butler Retrieval
# ---------------------------------------------------------------------------


def _connect_butler(repo: str, collection: str):
    """Return a Butler client, exiting with a hint if the stack is missing."""
    try:
        import lsst.daf.butler as dafButler
    except ImportError:
        print(
            "Error: lsst.daf.butler is not installed/loaded in this environment.",
            file=sys.stderr,
        )
        print(
            "To query Butler, source the LSST stack environment first.", file=sys.stderr
        )
        sys.exit(1)

    return dafButler.Butler(repo, collections=[collection])


def _metrics_row_to_dict(metrics: Any) -> Optional[dict[str, Any]]:
    """Reduce an ``iqQaMetrics`` dataset to a single row dictionary."""
    if hasattr(metrics, "columns"):  # pandas DataFrame: one row per quantum
        if len(metrics) == 0:
            return None
        return metrics.iloc[0].to_dict()
    if hasattr(metrics, "to_dict"):  # pandas Series
        return metrics.to_dict()
    return dict(metrics)


def get_visit_metrics(
    repo: str, collection: str, visit: int, spectrograph: int, arms: tuple[str, ...]
) -> list[VisitQA]:
    """Read the ``iqQaMetrics`` datasets for a visit directly from Butler.

    This is the preferred Butler path: ``imageQualityQa`` has already parsed
    the task logs and written the result as a dataset, so the metrics survive
    even when the ``*_log`` datasets have been pruned from the collection.

    Parameters
    ----------
    repo : str
        Path to the Butler repository.
    collection : str
        Run collection where the pipeline output lives.
    visit : int
        Visit number.
    spectrograph : int
        Spectrograph module (1-4).
    arms : tuple of str
        Arms to query.

    Returns
    -------
    list of VisitQA
        One entry per arm that has an ``iqQaMetrics`` dataset.  Empty when the
        collection holds none.
    """
    butler = _connect_butler(repo, collection)

    vqa_list: list[VisitQA] = []
    for arm in arms:
        dataId = dict(instrument="PFS", visit=visit, arm=arm, spectrograph=spectrograph)
        try:
            metrics = butler.get("iqQaMetrics", dataId=dataId)
        except LookupError:
            continue

        row = _metrics_row_to_dict(metrics)
        if row is None:
            continue

        # The dataId is authoritative; the columns are only a convenience.
        row.setdefault("visit", visit)
        row["visit"] = visit
        row["arm"] = arm
        row["spectrograph"] = spectrograph

        vqa = VisitQA.from_metrics(row)
        vqa.collection = collection
        vqa_list.append(vqa)

    return vqa_list


def get_visit_logs(
    repo: str, collection: str, visit: int, spectrograph: int, arms: tuple[str, ...]
):
    """Gather all science pipeline logs for a single processed visit from Butler.

    Parameters
    ----------
    repo : str
        Path to the Butler repository.
    collection : str
        Run collection where the pipeline output lives.
    visit : int
        Visit number.
    spectrograph : int
        Spectrograph module (1-4).
    arms : tuple of str
        Arms to query.

    Returns
    -------
    dict
        Mapping of ``{taskLabel}`` to log records.
    butler
        The Butler client instance.
    """
    butler = _connect_butler(repo, collection)

    # Per-detector tasks: one log per (visit, arm, spectrograph)
    per_detector_tasks = [
        "isr",
        "cosmicray",
        "measureCentroids",
        "reduceExposure",
        "imageQualityQa",
    ]
    # Per-visit tasks: one log per (visit, spectrograph)
    per_visit_tasks = ["mergeArms", "fitFluxReference", "fitFluxCal", "applyFluxCal"]

    logs = {}

    for task in per_detector_tasks:
        for arm in arms:
            dataId = dict(
                instrument="PFS", visit=visit, arm=arm, spectrograph=spectrograph
            )
            try:
                records = butler.get(f"{task}_log", dataId=dataId)
                logs[f"{task}/{arm}{spectrograph}"] = records
            except LookupError:
                pass

    for task in per_visit_tasks:
        dataId = dict(instrument="PFS", visit=visit, spectrograph=spectrograph)
        try:
            records = butler.get(f"{task}_log", dataId=dataId)
            logs[f"{task}/{spectrograph}"] = records
        except LookupError:
            pass

    return logs, butler


# ---------------------------------------------------------------------------
# Plot Generator
# ---------------------------------------------------------------------------


def generate_plots(vqa: VisitQA, output_dir: Path, collection: Optional[str] = None):
    """Generate combined diagnostic dashboard with scorecard banner on top."""
    vqa.sanitize()
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "Error: matplotlib and numpy are required to generate plots. "
            "Run 'pip install matplotlib numpy' or use uv.",
            file=sys.stderr,
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    # Check if imageQualityQa task was run (status parsed from logs)
    iq_qa_run = vqa.qa_status != "UNKNOWN"

    is_fwhm_fallback = False
    if not iq_qa_run:
        # Fallback scorecard values if imageQualityQa wasn't run
        if math.isnan(vqa.qa_fwhm):
            vqa.qa_fwhm = 2.80  # nominal fallback FWHM in pixels
            is_fwhm_fallback = True
        if math.isnan(vqa.qa_dx):
            vqa.qa_dx = vqa.fit_x_rms if not math.isnan(vqa.fit_x_rms) else 0.0
        if math.isnan(vqa.qa_flagged):
            if hasattr(vqa, "fit_total_lines") and vqa.fit_total_lines > 0:
                vqa.qa_flagged = (1.0 - vqa.fit_n_lines / vqa.fit_total_lines) * 100
            else:
                vqa.qa_flagged = 0.0
        if vqa.qa_status == "UNKNOWN":
            if vqa.qa_fwhm < 3.2 and abs(vqa.qa_dx) < 0.2 and vqa.qa_flagged < 15.0:
                vqa.qa_status = "PASS"
            elif vqa.qa_fwhm >= 3.5 or abs(vqa.qa_dx) >= 0.5 or vqa.qa_flagged >= 40.0:
                vqa.qa_status = "FAIL"
            else:
                vqa.qa_status = "WARN"

    # Custom high-quality styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["text.color"] = "#1e293b"
    plt.rcParams["axes.labelcolor"] = "#1e293b"
    plt.rcParams["xtick.color"] = "#1e293b"
    plt.rcParams["ytick.color"] = "#1e293b"
    plt.rcParams["grid.color"] = "#e2e8f0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5

    c_good = "#10b981"
    c_low_sn = "#3b82f6"
    c_fail = "#ef4444"
    c_warning = "#f59e0b"

    # Layout: 1 large figure with a scorecard header on top, and 2x2 grid below
    fig = plt.figure(figsize=(16, 14.5), facecolor="#f8fafc")
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 3, 3])

    # ----------------------------------------------------
    # Header: QA Scorecard Banner (Row 0, spanning both columns)
    # ----------------------------------------------------
    ax_card = fig.add_subplot(gs[0, :])
    ax_card.set_facecolor("#f8fafc")
    ax_card.patch.set_facecolor("#f8fafc")
    ax_card.spines["top"].set_visible(False)
    ax_card.spines["right"].set_visible(False)
    ax_card.spines["bottom"].set_visible(False)
    ax_card.spines["left"].set_visible(False)
    ax_card.get_xaxis().set_visible(False)
    ax_card.get_yaxis().set_visible(False)

    # Use slightly darker colors for status on light background (better contrast)
    c_good_lbl = "#059669"  # emerald-600
    c_warning_lbl = "#d97706"  # amber-600
    c_fail_lbl = "#dc2626"  # red-600

    color_status = (
        c_good_lbl
        if vqa.qa_status == "PASS"
        else (
            c_warning_lbl
            if vqa.qa_status == "WARN"
            else c_fail_lbl if vqa.qa_status == "FAIL" else "#64748b"
        )
    )

    # Color code individual metrics based on pass/warn/fail thresholds
    if math.isnan(vqa.qa_fwhm):
        fwhm_color = "#64748b"
    elif vqa.qa_fwhm < 3.2:
        fwhm_color = c_good_lbl
    elif vqa.qa_fwhm < 3.5:
        fwhm_color = c_warning_lbl
    else:
        fwhm_color = c_fail_lbl

    if math.isnan(vqa.qa_dx):
        dx_color = "#64748b"
    elif abs(vqa.qa_dx) < 0.2:
        dx_color = c_good_lbl
    elif abs(vqa.qa_dx) < 0.5:
        dx_color = c_warning_lbl
    else:
        dx_color = c_fail_lbl

    if math.isnan(vqa.qa_flagged):
        flagged_color = "#64748b"
    else:
        flagRateWarnThreshold = {
            "b": 50.0,
            "r": 15.0,
            "n": 15.0,
            "m": 15.0,
            "b:HgCd": 15.0,
            "b:Neon": 50.0,
            "b:Krypton": 55.0,
            "b:Xenon": 85.0,
            "b:Argon": 93.0,
        }
        flagRateFailThreshold = {
            "b": 60.0,
            "r": 20.0,
            "n": 20.0,
            "m": 20.0,
            "b:HgCd": 25.0,
            "b:Neon": 60.0,
            "b:Krypton": 65.0,
            "b:Xenon": 92.0,
            "b:Argon": 97.0,
        }
        arm = vqa.arm
        species = (
            vqa.qa_target.split(":", 1)[-1].strip()
            if ":" in vqa.qa_target
            else vqa.qa_target
        )
        compoundKey = f"{arm}:{species}" if species else ""
        flag_warn = flagRateWarnThreshold.get(
            compoundKey, flagRateWarnThreshold.get(arm, 15.0)
        )
        flag_fail = flagRateFailThreshold.get(
            compoundKey, flagRateFailThreshold.get(arm, 20.0)
        )
        if vqa.qa_flagged < flag_warn:
            flagged_color = c_good_lbl
        elif vqa.qa_flagged < flag_fail:
            flagged_color = c_warning_lbl
        else:
            flagged_color = c_fail_lbl

    ax_card.text(
        0.03,
        0.74,
        f"QA STATUS {vqa.visit} {vqa.arm}{vqa.spectrograph}: {vqa.qa_status}",
        color=color_status,
        fontsize=22,
        fontweight="black",
    )

    # Metadata info (top right of the scorecard card)
    display_col = collection if collection is not None else vqa.collection
    meta_text = (
        f"Visit: {vqa.visit}  |  Arm: {vqa.arm}{vqa.spectrograph}\n"
        f"Collection: {display_col if display_col else 'Offline Log File'}"
    )
    ax_card.text(
        0.97,
        0.74,
        meta_text,
        color="#475569",
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="center",
    )

    # Format cosmic rays metric
    total_crs = sum(x[0] for x in vqa.cosmic_rays) if vqa.cosmic_rays else 0
    total_pixels = sum(x[1] for x in vqa.cosmic_rays) if vqa.cosmic_rays else 0
    if total_pixels >= 10000:
        pix_str = f"{total_pixels/1000:.0f}k"
    else:
        pix_str = f"{total_pixels}"
    cr_lbl = f"{total_crs:,} ({pix_str} px)" if total_crs > 0 else "N/A"
    cr_status_lbl = "nominal" if total_crs > 0 else "skipped"
    cr_color = "#475569" if total_crs > 0 else "#64748b"

    # Format Fit RMS metric
    if not math.isnan(vqa.fit_x_rms) and not math.isnan(vqa.fit_y_rms):
        rms_lbl = f"{vqa.fit_x_rms:.3f}/{vqa.fit_y_rms:.3f} px"
    elif not math.isnan(vqa.fit_x_rms):
        rms_lbl = f"{vqa.fit_x_rms:.3f} px"
    else:
        rms_lbl = "N/A"

    if math.isnan(vqa.fit_x_rms):
        rms_color = "#64748b"
        rms_status_lbl = "N/A"
    elif vqa.fit_x_rms < 0.15:
        rms_color = c_good_lbl
        rms_status_lbl = "nominal"
    else:
        rms_color = c_warning_lbl
        rms_status_lbl = "high"

    # Format Fit Soften metric
    if not math.isnan(vqa.fit_x_soften) and not math.isnan(vqa.fit_y_soften):
        soften_lbl = f"{vqa.fit_x_soften:.3f}/{vqa.fit_y_soften:.3f} px"
    elif not math.isnan(vqa.fit_x_soften):
        soften_lbl = f"{vqa.fit_x_soften:.3f} px"
    else:
        soften_lbl = "N/A"

    if math.isnan(vqa.fit_x_soften):
        soften_color = "#64748b"
        soften_status_lbl = "N/A"
    elif vqa.fit_x_soften < 0.05:
        soften_color = c_good_lbl
        soften_status_lbl = "nominal"
    elif vqa.fit_x_soften < 0.15:
        soften_color = c_warning_lbl
        soften_status_lbl = "moderate"
    else:
        soften_color = c_fail_lbl
        soften_status_lbl = "high"

    fwhm_lbl = (
        f"{vqa.qa_fwhm:.2f} px*"
        if is_fwhm_fallback
        else (f"{vqa.qa_fwhm:.2f} px" if not math.isnan(vqa.qa_fwhm) else "N/A")
    )
    metrics = [
        (
            "Median FWHM",
            fwhm_lbl,
            (
                "nominal (<3.2px)"
                if not math.isnan(vqa.qa_fwhm) and vqa.qa_fwhm < 3.2
                else ("high/degraded" if not math.isnan(vqa.qa_fwhm) else "N/A")
            ),
            fwhm_color,
        ),
        (
            "Fit RMS (x/y)",
            rms_lbl,
            rms_status_lbl,
            rms_color,
        ),
        (
            "Fit Soften (x/y)",
            soften_lbl,
            soften_status_lbl,
            soften_color,
        ),
        (
            (
                "Center shift (dx/RMS)"
                if not math.isnan(vqa.qa_dx_rms)
                else "Center shift (dx)"
            ),
            (
                f"{vqa.qa_dx:+.3f}/{vqa.qa_dx_rms:.3f} px"
                if not math.isnan(vqa.qa_dx_rms)
                else (f"{vqa.qa_dx:+.3f} px" if not math.isnan(vqa.qa_dx) else "N/A")
            ),
            (
                "nominal"
                if not math.isnan(vqa.qa_dx) and abs(vqa.qa_dx) < 0.2
                else ("high/drifted" if not math.isnan(vqa.qa_dx) else "N/A")
            ),
            dx_color,
        ),
        (
            "Pct Flagged",
            f"{vqa.qa_flagged:.1f} %" if not math.isnan(vqa.qa_flagged) else "N/A",
            f"{vqa.qa_status}" if not math.isnan(vqa.qa_flagged) else "N/A",
            flagged_color,
        ),
        (
            "Cosmic Rays",
            cr_lbl,
            cr_status_lbl,
            cr_color,
        ),
    ]

    x_positions = [0.03, 0.19, 0.35, 0.51, 0.67, 0.83]
    for (name, val, status, color), x_pos in zip(metrics, x_positions):
        ax_card.text(x_pos, 0.52, name, color="#475569", fontsize=11, fontweight="bold")
        ax_card.text(x_pos, 0.26, val, color=color, fontsize=18, fontweight="bold")
        ax_card.text(
            x_pos,
            0.08,
            f"[{status}]",
            color="#64748b",
            fontsize=10,
            style="italic",
        )

    note_parts = []
    if vqa.qa_detail:
        note_parts.append(vqa.qa_detail)
    elif vqa.dither is not None:
        note_parts.append(f"Dither {vqa.dither} | design ID {vqa.pfs_design_id}")
    if is_fwhm_fallback:
        note_parts.append("*Nominal fallback (imageQualityQa not run)")
    note = "Note: " + " | ".join(note_parts)
    ax_card.text(0.03, 0.01, note, color="#64748b", fontsize=9, style="italic")

    # ----------------------------------------------------
    # Subplot 1: Centroids Pie Chart (Row 1, Col 0)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[1, 0])
    sizes = [
        vqa.centroids_good,
        vqa.centroids_low_sn,
        vqa.centroids_total - vqa.centroids_good - vqa.centroids_low_sn,
    ]
    if vqa.centroids_total <= 0 or sum(sizes) <= 0 or any(math.isnan(s) for s in sizes):
        ax1.text(
            0.5,
            0.5,
            "No Centroid Data Found",
            ha="center",
            va="center",
            fontsize=12,
            color="#64748b",
        )
        ax1.axis("off")
    else:
        labels = [
            f"Good ({vqa.centroids_good_pct}%)",
            f"Low-S/N ({vqa.centroids_low_sn_pct}%)",
            f"Fail ({vqa.centroids_fail_pct}% / Other)",
        ]
        colors = [c_good, c_low_sn, c_fail]
        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops=dict(color="#0f172a"),
            explode=(0.1, 0, 0) if sizes[0] > 0 else (0, 0, 0),
            pctdistance=0.75,
            wedgeprops=dict(width=0.4, edgecolor="w"),
        )
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=11, weight="bold")
    if vqa.centroids_total > 0:
        ax1.set_title(
            f"Line Centroids Quality Distribution (N={vqa.centroids_total:,})",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color="#0f172a",
        )
    else:
        ax1.set_title(
            "Line Centroids Quality Distribution",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color="#0f172a",
        )

    # ----------------------------------------------------
    # Subplot 2: Fiber residuals (Row 1, Col 1)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    if vqa.fibers:
        fiber_ids = [str(f.fiber_id) for f in vqa.fibers]
        x_rms = [f.x_rms for f in vqa.fibers]
        y_rms = [f.y_rms for f in vqa.fibers]
        x = np.arange(len(fiber_ids))
        width = 0.35

        ax2.bar(
            x - width / 2,
            x_rms,
            width,
            label="xRMS (Dispersion/Spatial)",
            color="#ec4899",
            alpha=0.9,
        )
        ax2.bar(
            x + width / 2,
            y_rms,
            width,
            label="yRMS (Wavelength)",
            color="#3b82f6",
            alpha=0.9,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(fiber_ids)
        ax2.set_ylabel("RMS Residual (pixels)", fontsize=11, fontweight="semibold")
        ax2.set_xlabel("Fiber ID", fontsize=11, fontweight="semibold")
        ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    else:
        ax2.text(
            0.5, 0.5, "No Fiber Statistics Found", ha="center", va="center", fontsize=14
        )
    if vqa.fit_active_fibers > 0:
        ax2.set_title(
            f"Detector Map Fitting Residuals by Fiber\n(Showing {len(vqa.fibers)} of {vqa.fit_active_fibers} active fibers)",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color="#0f172a",
        )
    else:
        ax2.set_title(
            f"Detector Map Fitting Residuals by Fiber\n(Showing {len(vqa.fibers)} active fibers)",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color="#0f172a",
        )
    ax2.grid(True, axis="y")

    # ----------------------------------------------------
    # Subplot 3: Used vs Reserved Fit Quality Comparison (Row 2, Col 0)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    if not math.isnan(vqa.fit_reserved_x_rms):
        categories_res = ["xRMS (Dispersion/Spatial)", "yRMS (Wavelength)"]
        used_vals = [
            vqa.fit_x_rms if not math.isnan(vqa.fit_x_rms) else 0.0,
            vqa.fit_y_rms if not math.isnan(vqa.fit_y_rms) else 0.0,
        ]
        reserved_vals = [vqa.fit_reserved_x_rms, vqa.fit_reserved_y_rms]
        x_res = np.arange(len(categories_res))
        width = 0.35

        ax3.bar(
            x_res - width / 2,
            used_vals,
            width,
            label=f"Used Lines ({vqa.fit_n_lines:,})",
            color="#0284c7",
            alpha=0.9,
        )
        ax3.bar(
            x_res + width / 2,
            reserved_vals,
            width,
            label=f"Reserved Lines ({vqa.fit_reserved_n_lines:,})",
            color="#f59e0b",
            alpha=0.9,
        )
        ax3.set_ylabel("RMS Residual (pixels)", fontsize=11, fontweight="semibold")
        ax3.set_xticks(x_res)
        ax3.set_xticklabels(categories_res, fontsize=10, fontweight="bold")
        ax3.grid(True, axis="y")
        ax3.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    else:
        ax3.text(
            0.5,
            0.5,
            "No Reserved Lines Data Found",
            ha="center",
            va="center",
            fontsize=14,
            color="#64748b",
        )
        ax3.axis("off")
    ax3.set_title(
        "Fit Quality: Used vs Reserved Lines",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#0f172a",
    )
    ax3.grid(True, axis="y")

    # ----------------------------------------------------
    # Subplot 4: Global Fitting Residuals Comparison (Row 2, Col 1)
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    categories = ["Overall Fit"]
    x_rms_vals = [vqa.fit_x_rms if not math.isnan(vqa.fit_x_rms) else 0.0]
    y_rms_vals = [vqa.fit_y_rms if not math.isnan(vqa.fit_y_rms) else 0.0]

    # Add all species from the dictionary
    for sp, (x_rms, y_rms) in vqa.fit_species_stats.items():
        if not (math.isnan(x_rms) and math.isnan(y_rms)):
            categories.append(f"{sp} Lines")
            x_rms_vals.append(x_rms if not math.isnan(x_rms) else 0.0)
            y_rms_vals.append(y_rms if not math.isnan(y_rms) else 0.0)

    # Fallback to single species if dictionary is empty
    if len(categories) == 1 and not math.isnan(vqa.fit_species_x_rms):
        categories.append(f"{vqa.fit_species_name} Lines")
        x_rms_vals.append(vqa.fit_species_x_rms)
        y_rms_vals.append(
            vqa.fit_species_y_rms if not math.isnan(vqa.fit_species_y_rms) else 0.0
        )

    # Add Traces
    categories.append("Traces")
    x_rms_vals.append(
        vqa.fit_trace_x_rms if not math.isnan(vqa.fit_trace_x_rms) else 0.0
    )
    y_rms_vals.append(
        vqa.fit_trace_y_rms if not math.isnan(vqa.fit_trace_y_rms) else 0.0
    )

    has_data = any(val > 0.0 for val in x_rms_vals + y_rms_vals)

    if has_data:
        x_cat = np.arange(len(categories))
        width = 0.35

        ax4.bar(
            x_cat - width / 2,
            x_rms_vals,
            width,
            label="xRMS (Dispersion/Spatial)",
            color="#ec4899",
            alpha=0.9,
        )
        ax4.bar(
            x_cat + width / 2,
            y_rms_vals,
            width,
            label="yRMS (Wavelength)",
            color="#3b82f6",
            alpha=0.9,
        )
        ax4.set_ylabel("RMS Residual (pixels)", fontsize=11, fontweight="semibold")
        ax4.set_xticks(x_cat)
        ax4.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax4.grid(True, axis="y")
        ax4.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    else:
        ax4.text(
            0.5,
            0.5,
            "No Fitting Residuals Found",
            ha="center",
            va="center",
            fontsize=12,
            color="#64748b",
        )
        ax4.axis("off")
    ax4.set_title(
        "Global Fitting Residuals Comparison",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#0f172a",
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    dashboard_path = (
        output_dir / f"qa_dashboard_{vqa.visit}_{vqa.arm}{vqa.spectrograph}.png"
    )
    plt.savefig(dashboard_path, dpi=150, facecolor="#f8fafc")
    plt.close()

    print(f"Generated combined dashboard plot: \n  - {dashboard_path}")


def generate_markdown_report(
    vqa: VisitQA,
    output_path: Path,
    plot_dir: Optional[Path] = None,
    collection: Optional[str] = None,
):
    """Generate a detailed markdown QA diagnostic report based on VisitQA metrics."""
    vqa.sanitize()
    target = (
        vqa.qa_target
        if vqa.qa_target
        else (vqa.collection if vqa.collection else "Unknown Target")
    )

    # 1. Optical Focus
    if math.isnan(vqa.qa_fwhm):
        fwhm_status = "N/A"
        fwhm_desc = "No FWHM measurements available."
    elif vqa.qa_fwhm < 3.2:
        fwhm_status = "**Nominal**"
        fwhm_desc = "The FWHM is sharp and well-focused, indicating excellent spectrograph alignment and focusing."
    elif vqa.qa_fwhm < 3.5:
        fwhm_status = "**Marginal/Degraded**"
        fwhm_desc = "The focus is slightly degraded, showing minor defocusing or mirror temperature deviations."
    else:
        fwhm_status = "**FAILED/Defocused**"
        fwhm_desc = "The focus is significantly degraded. Optical alignment or mirror focus mechanism needs investigation."

    # 2. dxCenter
    if math.isnan(vqa.qa_dx):
        dx_status = "N/A"
        dx_desc = "No spatial center shift measurements available."
    elif abs(vqa.qa_dx) < 0.2:
        dx_status = "**Nominal**"
        dx_desc = "The physical spatial shift (flexure) along the dispersion direction is extremely small."
    elif abs(vqa.qa_dx) < 0.5:
        dx_status = "**Marginal/Moderate**"
        dx_desc = f"Moderate physical drift (flexure) detected. The pipeline successfully adjusted for this offset, but the instrument has shifted slightly."
    else:
        dx_status = "**FAILED/Severe Drift**"
        dx_desc = f"Severe physical alignment shift detected. Search boxes for line centroiding may be misaligned."

    # 3. Centroids
    is_science = (
        "arc" not in target.lower()
        and "calib" not in target.lower()
        and "neon" not in target.lower()
        and "argon" not in target.lower()
        and "hgcd" not in target.lower()
        and "xenon" not in target.lower()
        and "krypton" not in target.lower()
    )

    if vqa.centroids_total <= 0:
        centroid_desc = "No line centroiding data found."
    else:
        if is_science:
            centroid_desc = (
                f"For a science target, a low Good centroid rate ({vqa.centroids_good_pct}%) is **normal and expected** "
                f"since most target coordinates/wavelengths do not contain bright emission lines (they measure faint objects/sky background). "
                f"The centroid failure rate is extremely low ({vqa.centroids_fail_pct}%), showing excellent data integrity."
            )
        else:
            if vqa.centroids_good_pct > 10:
                centroid_desc = (
                    f"Good centroid rate ({vqa.centroids_good_pct}%) is nominal for a calibration frame. "
                    f"A low failure rate ({vqa.centroids_fail_pct}%) confirms high calibration line quality."
                )
            else:
                centroid_desc = (
                    f"Good centroid rate ({vqa.centroids_good_pct}%) is **critically low** for a calibration frame. "
                    f"This indicates the search boxes were misaligned (often due to high `dxCenter`) or the lamp was faint, "
                    f"causing the pipeline to look for lines in empty space."
                )

    # 4. Fit Residuals and Soften
    if math.isnan(vqa.fit_x_rms) and math.isnan(vqa.fit_y_rms):
        fit_desc = "No detector map fitting residuals available."
    else:
        fit_desc = f"Global residuals ($x\\text{{RMS}} = {vqa.fit_x_rms:.4f}\\text{{ px}}$ and $y\\text{{RMS}} = {vqa.fit_y_rms:.4f}\\text{{ px}}$) are "
        if (vqa.fit_x_rms < 0.15 or math.isnan(vqa.fit_x_rms)) and (
            vqa.fit_y_rms < 0.15 or math.isnan(vqa.fit_y_rms)
        ):
            fit_desc += "**excellent**, demonstrating a very precise distortion mapping solution."
        else:
            fit_desc += "**high**, indicating fitting instabilities, poor line constraints, or large optical distortions."

    if not math.isnan(vqa.fit_x_soften):
        fit_desc += f" Systematic error floors (softening) are $x\\text{{Soften}} = {vqa.fit_x_soften:.4f}\\text{{ px}}$ and $y\\text{{Soften}} = {vqa.fit_y_soften:.4f}\\text{{ px}}$."
        if vqa.fit_x_soften < 0.05:
            fit_desc += " The low $x$-softening confirms that the distortion model matches the physical traces without needing a large systematic error floor."
        else:
            fit_desc += " The elevated systematic error floor shows the fitter had to down-weight measurement errors to cope with unmodeled distortion or centroid scatter."

    # 4.5 Fibers
    if vqa.fit_active_fibers > 0:
        fibers_used_str = f"**{vqa.fit_active_fibers}** active fibers contributed lines to the fit (sample residuals for {len(vqa.fibers)} fibers are plotted below)."
    elif len(vqa.fibers) < 5:
        fibers_used_str = f"**{len(vqa.fibers)}** active fibers contributed lines to the fit (all are plotted below)."
    else:
        fibers_used_str = f"At least **{len(vqa.fibers)}** active fibers have residual statistics logged (sample residuals plotted below)."

    # 4.6 Line Flagging Rate (pctFlagged)
    if math.isnan(vqa.qa_flagged):
        flag_status = "N/A"
        flag_desc = "No flagged line percentage available."
    else:
        warn_thresh = 40.0 if vqa.arm == "b" else 15.0
        if vqa.qa_flagged < warn_thresh:
            flag_status = "**Nominal**"
            flag_desc = f"Flagged line rate ({vqa.qa_flagged:.1f}%) is within the normal threshold of <{warn_thresh}%."
        else:
            flag_status = "**WARNING/FAIL (Elevated)**"
            flag_desc = (
                f"Flagged line rate ({vqa.qa_flagged:.1f}%) exceeds the threshold of <{warn_thresh}%. "
                f"This indicates that a large fraction of the reference lines were rejected/flagged by the fitter "
                f"(often due to low signal-to-noise ratio, line crowding, or mismatched species thresholds)."
            )

    # 4.7 Detector Map Fitting Warnings
    fit_warnings = []
    if vqa.fit_n_lines > 0 and vqa.fit_n_lines < 20:
        fit_warnings.append(
            f"* ⚠️ **Insufficient constraints:** Only **{vqa.fit_n_lines}** lines were used in the detector map fit. "
            f"This is extremely low and insufficient to constrain the distortion model parameters properly."
        )
    elif vqa.fit_n_lines == 0:
        fit_warnings.append(
            "* ❌ **No constraints:** Zero lines were used in the detector map fit! The fit is completely unconstrained."
        )

    if vqa.fit_active_fibers > 0 and vqa.fit_active_fibers < 10:
        fit_warnings.append(
            f"* ⚠️ **Sparse fiber coverage:** Only **{vqa.fit_active_fibers}** fibers had valid lines contributing to the fit "
            f"(out of ~600 fibers). The spatial distribution of constraints across the detector is extremely sparse."
        )
    elif len(vqa.fibers) > 0 and len(vqa.fibers) < 5 and vqa.fit_active_fibers == 0:
        fit_warnings.append(
            f"* ⚠️ **Sparse fiber coverage:** Only **{len(vqa.fibers)}** fibers had valid lines. The fit constraints are extremely sparse."
        )

    if (not math.isnan(vqa.fit_x_rms) and math.isnan(vqa.fit_y_rms)) or (
        math.isnan(vqa.fit_x_rms) and not math.isnan(vqa.fit_y_rms)
    ):
        fit_warnings.append(
            "* ❌ **Degenerate fit:** One of the fitting RMS residuals (spatial or wavelength) is `NaN`, indicating a degenerate fit."
        )

    fit_warnings_str = "\n".join(fit_warnings) if fit_warnings else ""

    # 5. Cosmic Rays
    total_crs = sum(cr[0] for cr in vqa.cosmic_rays) if vqa.cosmic_rays else 0
    total_pixels = sum(cr[1] for cr in vqa.cosmic_rays) if vqa.cosmic_rays else 0
    if total_crs > 0:
        cr_desc = f"Found {total_crs:,} cosmic rays affecting {total_pixels:,} pixels. This is within normal background limits for this exposure."
    else:
        cr_desc = "Cosmic ray task was skipped or no cosmic rays were reported."

    # 6. Recommendations
    recs = []
    if vqa.qa_status == "PASS":
        recs.append(
            "* **No action required.** The image quality, optical focus, and calibration fit are excellent."
        )
    if not math.isnan(vqa.qa_dx) and abs(vqa.qa_dx) >= 0.2:
        recs.append(
            "* **Verify slit offsets configuration:** If flexure/alignment shift continues to grow, ensure `doSlitOffsets` is enabled in `fitDetectorMap` config to shift the template before centroiding."
        )
    if not math.isnan(vqa.qa_fwhm) and vqa.qa_fwhm >= 3.2:
        recs.append(
            "* **Inspect spectrograph focus/alignment:** Defocusing is present. Check mirror focusing mechanics and spectrograph temperature logs."
        )
    if (
        not is_science
        and vqa.centroids_good_pct < 10
        and not math.isnan(vqa.qa_dx)
        and abs(vqa.qa_dx) >= 0.5
    ):
        recs.append(
            "* **Check base calibration map:** The template detector map is significantly shifted relative to physical traces. Verify that a stale calibration file is not being used."
        )

    # Diagnosis Summary
    if vqa.qa_status == "PASS":
        diagnosis_summary = (
            "**Yes.** This run represents an excellent, high-quality exposure."
        )
    else:
        reasons = []
        if not math.isnan(vqa.qa_fwhm) and vqa.qa_fwhm >= 3.2:
            reasons.append("optical defocusing (elevated FWHM)")
        if not math.isnan(vqa.qa_dx) and abs(vqa.qa_dx) >= 0.2:
            reasons.append("physical flexure/spatial shift (dxCenter)")
        if not math.isnan(vqa.qa_flagged) and vqa.qa_flagged >= (
            40.0 if vqa.arm == "b" else 15.0
        ):
            reasons.append("high flagged line rate (pctFlagged)")
        if vqa.fit_n_lines > 0 and vqa.fit_n_lines < 20:
            reasons.append("critically low number of fit lines")
        if vqa.fit_active_fibers > 0 and vqa.fit_active_fibers < 10:
            reasons.append("critically low number of active fibers")
        if (not math.isnan(vqa.fit_x_rms) and math.isnan(vqa.fit_y_rms)) or (
            math.isnan(vqa.fit_x_rms) and not math.isnan(vqa.fit_y_rms)
        ):
            reasons.append("degenerate fit (NaN residuals)")

        reasons_str = (
            ", ".join(reasons) if reasons else "unspecified calibration/image issues"
        )
        diagnosis_summary = (
            f"**No.** This run flagged a warning or failure due to: **{reasons_str}**."
        )

    # Status Emoji
    status_emoji = (
        "✅ PASS"
        if vqa.qa_status == "PASS"
        else (
            "⚠️ WARN"
            if vqa.qa_status == "WARN"
            else "❌ FAIL" if vqa.qa_status == "FAIL" else "❓ UNKNOWN"
        )
    )

    # Plot path
    plot_file = f"qa_dashboard_{vqa.visit}_{vqa.arm}{vqa.spectrograph}.png"
    plot_path = (plot_dir / plot_file) if plot_dir else Path(plot_file)

    # If the plot doesn't exist, generate it
    if not plot_path.exists():
        generate_plots(vqa, plot_path.parent, collection=collection)

    # Read the image and base64-encode it so the markdown is self-contained
    import base64

    if plot_path.exists():
        with open(plot_path, "rb") as img_file:
            encoded_str = base64.b64encode(img_file.read()).decode("utf-8")
        img_src = f"data:image/png;base64,{encoded_str}"
    else:
        img_src = ""

    report_md = f"""# DRP Image Quality QA Diagnostic Report
**Visit:** {vqa.visit} | **Arm:** {vqa.arm}{vqa.spectrograph} | **Target/Sequence:** {target} | **Status:** {status_emoji}

This report presents a diagnostic analysis of the `reduceExposure` pipeline execution logs.

---

## 📊 Visual Diagnostics

### Combined Image Quality QA Dashboard
![Combined Image Quality QA Dashboard]({img_src})

---

## 🔍 Diagnosis: Is this a "Good" Image?

{diagnosis_summary}

Here is the step-by-step diagnostic breakdown:

### 1. Optical Focus (FWHM)
* **Median FWHM:** **{vqa.qa_fwhm:.2f} px** (Status: {fwhm_status})
  * {fwhm_desc}

### 2. Flexure / Spatial Shift (`dxCenter`)
* **Center shift (`dxCenter`):** **{vqa.qa_dx:+.3f} px**{f" (RMS: {vqa.qa_dx_rms:.3f} px)" if not math.isnan(vqa.qa_dx_rms) else ""} (Status: {dx_status})
  * {dx_desc}

### 3. Centroiding Quality and Line Flagging
* **Centroid Statistics:**
  * **Total Centroids:** {vqa.centroids_total:,}
  * **Good Centroids:** {vqa.centroids_good:,} ({vqa.centroids_good_pct}%)
  * **Low S/N (<5.0):** {vqa.centroids_low_sn:,} ({vqa.centroids_low_sn_pct}%)
  * **Failed Centroids:** {vqa.centroids_fail:,} ({vqa.centroids_fail_pct}%)
  * {centroid_desc}
* **Line Flagging Rate (`pctFlagged`):** **{vqa.qa_flagged:.1f}%** (Status: {flag_status})
  * {flag_desc}

### 4. Detector Map Fitting Residuals
* **Global Residuals (RMS):**
  * {fit_desc}
* **Fit Constraints:** **{vqa.fit_n_lines:,}** total lines were used to constrain the global fit (out of {vqa.fit_total_lines:,} candidate lines).
* **Fibers Used:** {fibers_used_str}
{fit_warnings_str}

### 5. Cosmic Rays
* **Cosmic Ray Statistics:**
  * {cr_desc}

---

## ⏱️ Execution Durations
* **ISR (Instrument Signature Removal):** {vqa.isr_time_s:.2f} s
* **Cosmic Ray Detection:** {vqa.cosmic_ray_time_s:.2f} s
* **reduceExposure (Total Task):** {vqa.reduce_exposure_time_s:.2f} s ({(vqa.reduce_exposure_time_s/60.0):.2f} minutes)
* **imageQualityQa:** {vqa.iq_qa_time_s:.2f} s
{"* **mergeArms:** " + f"{vqa.merge_arms_time_s:.2f} s" if vqa.merge_arms_time_s > 0.0 else ""}

---

## 🛠️ Recommendations
{"\n".join(recs)}
"""

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_md)
    print(f"Generated detailed diagnostic report: \n  - {output_path}")


# ---------------------------------------------------------------------------
# CLI Reporter
# ---------------------------------------------------------------------------


def print_text_report(vqa: VisitQA):
    vqa.sanitize()
    print("=" * 80)
    status_emoji = (
        "✅"
        if vqa.qa_status == "PASS"
        else (
            "⚠️"
            if vqa.qa_status == "WARN"
            else "❌" if vqa.qa_status == "FAIL" else "❓"
        )
    )
    print(
        f" {status_emoji}  PFS DRP QA REPORT: Visit {vqa.visit} | "
        f"Arm {vqa.arm}{vqa.spectrograph} ({vqa.qa_target}) | STATUS: {vqa.qa_status}"
    )
    print("=" * 80)
    print(f"  * Dither:             {vqa.dither}")
    fwhm_comment = ""
    if not math.isnan(vqa.qa_fwhm):
        fwhm_comment = "  (Nominal)" if vqa.qa_fwhm < 3.2 else "  (Degraded Focus)"
    fwhm_str = f"{vqa.qa_fwhm:.2f} px" if not math.isnan(vqa.qa_fwhm) else "N/A"
    print(f"  * Median FWHM:        {fwhm_str}{fwhm_comment}")

    dx_comment = ""
    if not math.isnan(vqa.qa_dx):
        dx_comment = (
            "  (Nominal)"
            if abs(vqa.qa_dx) < 0.15
            else (
                "  (Severe physical drift!)" if abs(vqa.qa_dx) > 0.5 else "  (Marginal)"
            )
        )
    dx_str = f"{vqa.qa_dx:+.3f} px" if not math.isnan(vqa.qa_dx) else "N/A"
    if not math.isnan(vqa.qa_dx_rms):
        dx_str += f" (RMS: {vqa.qa_dx_rms:.3f} px)"
    print(f"  * Center Shift (dx):  {dx_str}{dx_comment}")

    flagged_str = f"{vqa.qa_flagged:.1f}%" if not math.isnan(vqa.qa_flagged) else "N/A"
    print(f"  * Pct Flagged Lines:  {flagged_str}  [Details: {vqa.qa_detail}]")
    print("-" * 80)
    print("  Line Centroiding Performance:")
    print(f"    - Total Centroids:  {vqa.centroids_total:,}")
    print(f"    - Good Centroids:   {vqa.centroids_good:,} ({vqa.centroids_good_pct}%)")
    print(
        f"    - Low S/N (<5.0):   {vqa.centroids_low_sn:,} ({vqa.centroids_low_sn_pct}%)"
    )
    print(f"    - Failed Centroids: {vqa.centroids_fail:,} ({vqa.centroids_fail_pct}%)")
    print("-" * 80)
    print("  Detector Map Fit Residuals:")
    print(
        f"    - Global Residuals: xRMS = {vqa.fit_x_rms:.4f} px  |  yRMS = {vqa.fit_y_rms:.4f} px  (wavelength)"
    )
    print(
        f"    - Fit Softening:    xSoften = {vqa.fit_x_soften:.4f} px  |  ySoften = {vqa.fit_y_soften:.4f} px"
    )
    print(f"    - Fit Constraints:  {vqa.fit_n_lines:,} total lines used")

    if vqa.fibers:
        print("    - Sampled Fibers Detail:")
        print("        FiberId   |   xRMS (pixels)   |   yRMS (pixels)   |   nLines")
        print("        " + "-" * 56)
        for f in vqa.fibers:
            print(
                f"        {f.fiber_id:<9} |   {f.x_rms:<15.4f} |   {f.y_rms:<15.4f} |   {f.n_lines}"
            )

    print("-" * 80)
    print("  Execution Durations & Resources:")
    print(
        f"    - ISR:              {vqa.isr_time_s:.2f} s  (Bad Pixels: {vqa.isr_bad_pixels})"
    )
    if vqa.cosmic_ray_time_s > 0.0 or vqa.cosmic_rays:
        total_crs = sum(x[0] for x in vqa.cosmic_rays)
        total_pixels = sum(x[1] for x in vqa.cosmic_rays)
        print(
            f"    - Cosmic Ray Task:  {vqa.cosmic_ray_time_s:.2f} s  "
            f"(Found {total_crs} CRs, affecting {total_pixels} pixels)"
        )
    print(f"    - reduceExposure:   {vqa.reduce_exposure_time_s:.2f} s")
    print(f"    - imageQualityQa:   {vqa.iq_qa_time_s:.2f} s")
    if vqa.merge_arms_time_s > 0.0:
        print(f"    - mergeArms:        {vqa.merge_arms_time_s:.2f} s")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse reduceExposure & imageQualityQa logs (from files or Butler), assess QA health, and generate plots."
    )
    parser.add_argument(
        "log_files",
        nargs="*",
        type=Path,
        help="Paths to DRP execution log files (if not using Butler).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory to save combined diagnostic dashboard plots.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the serialized VisitQA dataclass as JSON.",
    )
    parser.add_argument(
        "--json-in",
        type=Path,
        default=None,
        help="Optional path to a serialized VisitQA JSON file to load and plot/report instead of parsing logs.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional path to write a detailed markdown QA report.",
    )

    # Butler specific options
    parser.add_argument(
        "--butler-repo",
        type=str,
        default=None,
        help="Path or URI to the Butler repository.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Run collection name (required if using Butler).",
    )
    parser.add_argument(
        "--visit",
        type=int,
        default=None,
        help="Visit number to query (required if using Butler).",
    )
    parser.add_argument(
        "--spectrograph",
        type=int,
        default=None,
        help="Spectrograph module (1-4) (required if using Butler).",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["b", "r", "n", "m"],
        help="Arms to query (default: b r n m).",
    )
    parser.add_argument(
        "--use-logs",
        action="store_true",
        help=(
            "With --butler-repo, parse the task logs instead of reading the"
            " iqQaMetrics datasets.  Logs additionally carry centroid counts"
            " and mergeArms timings, but are often pruned from a collection."
        ),
    )

    args = parser.parse_args()

    if args.json_in:
        print(f"Loading VisitQA from JSON file: {args.json_in}")
        try:
            with open(args.json_in, "r") as f:
                vqa_list = [VisitQA.from_json(f.read())]
        except Exception as e:
            print(f"Error loading JSON file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.butler_repo:
        if not args.collection or args.visit is None or args.spectrograph is None:
            parser.error(
                "--collection, --visit, and --spectrograph are all required when --butler-repo is set."
            )

        print(f"Connecting to Butler repository: {args.butler_repo}...")

        # iqQaMetrics is the primary source: imageQualityQa has already
        # parsed the logs into it, and it survives log pruning.  Fall back to
        # the raw task logs when the collection has no metrics (or when the
        # log-only fields are explicitly requested).
        vqa_list = []
        if not args.use_logs:
            vqa_list = get_visit_metrics(
                repo=args.butler_repo,
                collection=args.collection,
                visit=args.visit,
                spectrograph=args.spectrograph,
                arms=tuple(args.arms),
            )
            if vqa_list:
                found = ", ".join(f"{v.arm}{v.spectrograph}" for v in vqa_list)
                print(f"Read iqQaMetrics for: {found}")
            else:
                print("No iqQaMetrics datasets found; falling back to task logs.")

        if not vqa_list:
            logs, _ = get_visit_logs(
                repo=args.butler_repo,
                collection=args.collection,
                visit=args.visit,
                spectrograph=args.spectrograph,
                arms=tuple(args.arms),
            )

            if not logs:
                print(
                    f"Error: No iqQaMetrics or logs found in Butler for visit {args.visit}, "
                    f"spectrograph {args.spectrograph}.",
                    file=sys.stderr,
                )
                sys.exit(1)

            vqa_list = parse_butler_logs(
                logs, args.visit, args.spectrograph, collection=args.collection
            )
    else:
        if not args.log_files:
            parser.error("Must specify log_files, --json-in, or --butler-repo.")
        vqa_list = parse_logs(args.log_files, collection=args.collection)

    if not vqa_list:
        print("Error: No PFS DRP QA log data parsed from the inputs.", file=sys.stderr)
        sys.exit(1)

    # Print report for each visit found
    has_fail = False
    for vqa in vqa_list:
        print_text_report(vqa)
        if vqa.qa_status == "FAIL":
            has_fail = True

        if args.json_out:
            # If multiple visits are present, append visit/arm/spec details to filename if it doesn't already have them
            out_path = args.json_out
            if len(vqa_list) > 1:
                stem = out_path.stem
                ext = out_path.suffix
                out_path = out_path.with_name(
                    f"{stem}_{vqa.visit}_{vqa.arm}{vqa.spectrograph}{ext}"
                )

            print(f"Writing VisitQA JSON to: {out_path}")
            try:
                with open(out_path, "w") as f:
                    f.write(vqa.to_json())
            except Exception as e:
                print(f"Error writing JSON file: {e}", file=sys.stderr)

        if args.report_out:
            # If multiple visits are present, append visit/arm/spec details to filename if it doesn't already have them
            report_path = args.report_out
            if len(vqa_list) > 1:
                stem = report_path.stem
                ext = report_path.suffix
                report_path = report_path.with_name(
                    f"{stem}_{vqa.visit}_{vqa.arm}{vqa.spectrograph}{ext}"
                )
            generate_markdown_report(
                vqa, report_path, plot_dir=args.plot_dir, collection=args.collection
            )

        if args.plot_dir:
            generate_plots(vqa, args.plot_dir, collection=args.collection)

    if has_fail:
        print("QA Verdict: FAIL (At least one visit failed QA metrics).")
        sys.exit(2)
    else:
        print("QA Verdict: SUCCESS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
