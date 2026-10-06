from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import traceback
from types import SimpleNamespace
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
for path in (str(HERE),):
    if path not in sys.path:
        sys.path.insert(0, path)

from mtk_containers import ModemParts, UnwrapError, unwrap_path  # noqa: E402


class BackendError(RuntimeError):
    """Expected input/configuration failure that can be shown in the GUI."""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

UNIVERSAL = _load_module("mtk_gui_universal", HERE / "mtk_universal.py")


def human_size(size: int | None) -> str:
    if size is None:
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{size:,} B"
        size /= 1024
    return str(size)


def safe_stem(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
    return value[:80] or "mtk"


def parse_profile(value: str, *, field_name: str = "profile") -> str:
    value = value.strip().lower()
    if value == "all":
        return "all"
    if value.isdecimal():
        return str(int(value))
    raise BackendError(f"{field_name} must be 'all' or a non-negative integer")


def parse_profile_list(value: str) -> list[int]:
    """Parse an explicit comparison list; unlike extraction, `all` is invalid."""
    values = [part.strip() for part in value.split(",")]
    if not values or any(not part.isdecimal() for part in values):
        raise BackendError("comparison profiles must be comma-separated non-negative integers")
    numbers = [int(part) for part in values]
    if len(set(numbers)) != len(numbers):
        raise BackendError("comparison profiles must not contain duplicates")
    return numbers


EXPORT_FORMATS = ("b0cd", "b826", "cap_prune", "mtk_nr", "mtk_lte")


def parse_export_formats(values: Iterable[str]) -> frozenset[str]:
    formats = frozenset(value.strip().lower() for value in values if value.strip())
    unknown = formats.difference(EXPORT_FORMATS)
    if unknown:
        raise BackendError("unsupported export format(s): " + ", ".join(sorted(unknown)))
    if not formats:
        raise BackendError("choose at least one export format")
    return formats


@dataclass
class ModemRecord:
    source: Path
    capability_bank_index: int | None = None
    capability_bank: str | None = None
    lte_bank_index: int | None = None
    profile: int | None = None
    packaging: str = ""
    layers: tuple[str, ...] = ()
    rom_bytes: int | None = None
    drdi_bytes: int | None = None
    split_bytes: int | None = None
    source_summary: str = ""
    selected: bool = True
    status: str = "Ready"
    loader: str = "—"
    counts: dict[str, int] = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    combo_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def key(self) -> str:
        return str(self.source.resolve()).casefold()

    @property
    def display_name(self) -> str:
        base = self.source.name or str(self.source)
        if self.profile is None:
            return base
        bank = f"bank {self.capability_bank_index}" if self.capability_bank_index is not None else "capability bank"
        return f"{base} — {bank} / profile {self.profile}"


class MtkBackend:
    """No-Tk boundary around unwrapping, extraction, export and comparison."""

    @staticmethod
    def inspect(source: Path) -> ModemRecord:
        source = source.resolve()
        try:
            parts = unwrap_path(source)
        except UnwrapError as exc:
            raise BackendError(str(exc)) from exc
        selected = parts.report["selected"]
        layer_names = tuple(layer["format"] for layer in parts.report.get("layers", ()))
        return ModemRecord(
            source=source,
            packaging=parts.report.get("packaging", "unknown"),
            layers=layer_names,
            rom_bytes=selected["md1rom"]["bytes"],
            drdi_bytes=selected["md1drdi_hdr" if parts.drdi_data is not None else "md1drdi"]["bytes"],
            split_bytes=selected.get("md1drdi_data", {}).get("bytes"),
            source_summary=" → ".join(layer_names) if layer_names else "extracted parts",
            details=parts.report,
        )

    @staticmethod
    def parts(source: Path) -> ModemParts:
        try:
            return unwrap_path(source)
        except UnwrapError as exc:
            raise BackendError(str(exc)) from exc

    @classmethod
    def summarize(cls, record: ModemRecord, *, loader: str = "auto", retain_combos: bool = False) -> dict:
        """Decode one imported source to discover its exportable bank/profiles."""
        parts = cls.parts(record.source)
        args = SimpleNamespace(loader=loader, profile="all", out=None,
                               device=record.source.stem, stem=safe_stem(record.source.stem),
                               drdi_data=SimpleNamespace(read_bytes=lambda: parts.drdi_data)
                               if parts.drdi_data is not None else None)
        reporter = UNIVERSAL.Reporter()
        try:
            active, attempts, _data = UNIVERSAL.select_loader(
                args, parts.rom, parts.drdi, reporter, drdi_data=parts.drdi_data)
            cap, states, per_profile, union, lte_bank, lte_profiles, lte_union, unresolved = \
                UNIVERSAL.extract_capability(active, "all", reporter)
            secondary_combos = {} if retain_combos else None
            secondary_profiles = UNIVERSAL.tensor_secondary_summaries(
                active, "all", reporter, lte_profiles=lte_profiles,
                lte_count=len(lte_union), combo_profiles=secondary_combos)
            supported_bands = UNIVERSAL.annotate_band_participation(
                UNIVERSAL.discover_supported_bands(active, cap, lte_bank, reporter),
                union, lte_union)
            if retain_combos:
                cache = {}
                physical = isinstance(active, UNIVERSAL.TensorCdfLoader)
                notes = ["Static firmware capability variants, not a captured or advertised modem log.",
                         "NRDC is a presentation category for mixed FR1/FR2 NR rows."]
                notes.extend(dict.fromkeys(issue.message for issue in reporter.issues
                                           if issue.level in {"warn", "warning", "fail", "error"}))
                if isinstance(active, UNIVERSAL.Nr15Loader):
                    projection = active.decoder.projection_info()
                    notes.extend(projection["limitations"])
                    notes.append("BB customization: " + projection["bb_customization_activation"])
                for p, combos in per_profile.items():
                    lte_rows = (active.lte_rows_by_bank.get(cap.table_index, {}).get(p, ())
                                if physical else lte_profiles.get(p, ()))
                    cache[(cap.table_index, p)] = {
                        "nr": tuple(combos), "lte": tuple(UNIVERSAL.dedup_exact(lte_rows)),
                        "loader": active.name, "lte_bank_index": cap.table_index if physical
                        else lte_bank.table_index if lte_bank else None,
                        "notes": tuple(notes + (["This profile's NR features could not be fully resolved."]
                                                if p in unresolved else []))}
                for (bank, p), combos in (secondary_combos or {}).items():
                    cache[(bank, p)] = {"nr": tuple(combos), "lte": (), "loader": active.name,
                                       "lte_bank_index": bank, "notes": tuple(notes)}
                if physical:
                    for bank, profiles in active.lte_rows_by_bank.items():
                        for p, rows in profiles.items():
                            entry = cache.setdefault((bank, p), {"nr": (), "loader": active.name,
                                                   "lte_bank_index": bank, "notes": tuple(notes)})
                            entry["lte"] = tuple(UNIVERSAL.dedup_exact(rows))
                record.combo_cache = cache
        except Exception as exc:
            raise BackendError(str(exc)) from exc
        return {
            "loader": active.name, "loader_selection": attempts,
            "banks": [bank.to_dict() for bank in active.banks],
            "capability_bank": hex(cap.bank_va), "capability_bank_index": cap.table_index,
            "profiles": UNIVERSAL.serialize_profile_summary(states, per_profile),
            "secondary_profiles": secondary_profiles,
            "lte_bank": hex(lte_bank.bank_va) if lte_bank else None,
            "lte_bank_index": lte_bank.table_index if lte_bank else None,
            "lte_profiles": {str(key): len(value) for key, value in lte_profiles.items()},
            "supported_bands": supported_bands,
            "union": {"exact_rows": len(union),
                      "kinds": dict(zip(("endc", "nrca", "lte"),
                                         map(len, UNIVERSAL.export.classify(union, 1)))),
                      "complete": not unresolved, "unresolved_profiles": unresolved},
            "lte_union_exact_rows": len(lte_union), "validation": reporter.as_dict(),
            "physical_lte_profiles": ({str(b): {str(p): len(UNIVERSAL.dedup_exact(rows))
                                               for p, rows in profiles.items()}
                                       for b, profiles in active.lte_rows_by_bank.items()}
                                      if isinstance(active, UNIVERSAL.TensorCdfLoader) else None),
        }

    @staticmethod
    def profile_records(source_record: ModemRecord, summary: dict) -> list[ModemRecord]:
        """Expand one physical modem into the capability-bank profiles it proves."""
        records = []
        for profile in summary["profiles"]:
            number = int(profile["profile"])
            kinds = profile.get("gui_counts", profile["kinds"])
            records.append(ModemRecord(
                source=source_record.source, packaging=source_record.packaging,
                layers=source_record.layers, rom_bytes=source_record.rom_bytes,
                drdi_bytes=source_record.drdi_bytes, split_bytes=source_record.split_bytes,
                source_summary=source_record.source_summary, capability_bank_index=summary["capability_bank_index"],
                capability_bank=summary["capability_bank"], lte_bank_index=summary["lte_bank_index"],
                profile=number, loader=summary["loader"], status="Ready to export",
                counts={"lte": summary["lte_profiles"].get(str(number), 0),
                        "endc": kinds.get("endc", 0), "nr_sa": kinds.get("nr_sa", 0), "nrca": kinds.get("nrca", 0),
                        "nrdc": kinds.get("nrdc", 0)},
                details={**source_record.details, "summary": summary},
            ))
        # Tensor split-CDF keeps the FR2/NRDC namespace in a sibling bank
        # (normally Bank 8), so expose those profiles as independently
        # selectable rows instead of pretending their number belongs to Bank 6.
        for profile in summary.get("secondary_profiles", ()):
            number = int(profile["profile"])
            kinds = profile.get("gui_counts", profile.get("kinds", {}))
            records.append(ModemRecord(
                source=source_record.source, packaging=source_record.packaging,
                layers=source_record.layers, rom_bytes=source_record.rom_bytes,
                drdi_bytes=source_record.drdi_bytes, split_bytes=source_record.split_bytes,
                source_summary=source_record.source_summary,
                capability_bank_index=int(profile.get("bank_index", 8)),
                capability_bank=profile.get("bank_va"),
                lte_bank_index=summary.get("lte_bank_index"), profile=number,
                loader=summary["loader"], status="Ready to export",
                counts={"lte": int(profile.get("lte_count", 0)),
                        "endc": kinds.get("endc", 0), "nr_sa": kinds.get("nr_sa", 0), "nrca": kinds.get("nrca", 0),
                        "nrdc": kinds.get("nrdc", 0)},
                details={**source_record.details, "summary": summary,
                         "secondary_bank": int(profile.get("bank_index", 8))},
            ))
        physical = summary.get("physical_lte_profiles")
        if physical is not None:
            # Each Tensor row represents one physical bank. LTE from a sibling
            # bank must not appear under its NR bank's address/profile label.
            for record in records:
                record.counts["lte"] = physical.get(str(record.capability_bank_index), {}).get(str(record.profile), 0)
                record.details["bank_only"] = True
            indexed = {(r.capability_bank_index, r.profile) for r in records}
            banks = {b["table_index"]: b for b in summary["banks"]}
            for bank_number, profiles in physical.items():
                bank_number = int(bank_number)
                for profile, count in profiles.items():
                    number = int(profile)
                    if (bank_number, number) in indexed:
                        continue
                    records.append(ModemRecord(
                        source=source_record.source, packaging=source_record.packaging,
                        layers=source_record.layers, source_summary=source_record.source_summary,
                        capability_bank_index=bank_number,
                        capability_bank=banks[bank_number]["bank_va"], lte_bank_index=bank_number,
                        profile=number, loader=summary["loader"], status="Ready to export",
                        counts={"lte": count, "endc": 0, "nrca": 0, "nrdc": 0},
                        details={**source_record.details, "summary": summary, "bank_only": True}))
            records.sort(key=lambda r: (r.capability_bank_index, r.profile))
        if not records:
            raise BackendError("no validated capability profiles were discovered")
        for record in records:
            record.combo_cache = source_record.combo_cache
        return records

    @classmethod
    def extract_and_export(cls, record: ModemRecord, output: Path, *, device: str,
                           loader: str, extraction_profile: str,
                           export_formats: frozenset[str] = frozenset(EXPORT_FORMATS),
                           exclude_mimo_subsets: bool = False) -> dict:
        """Run the universal core with memory-resident, provenance-checked parts."""
        parts = cls.parts(record.source)
        output.mkdir(parents=True, exist_ok=True)
        args = SimpleNamespace(loader=loader, profile=extraction_profile, out=output,
                               device=device, stem=safe_stem(record.source.stem),
                               bank_index=record.capability_bank_index,
                               bank_only=record.details.get("bank_only", False),
                               export_formats=export_formats,
                               exclude_mimo_subsets=exclude_mimo_subsets,
                               drdi_data=SimpleNamespace(read_bytes=lambda: parts.drdi_data)
                               if parts.drdi_data is not None else None)
        reporter = UNIVERSAL.Reporter()
        report = {
            "tool": "mtk-gui / mtk_universal.py", "version": UNIVERSAL.VERSION,
            "inputs": {"source": "mtk-gui", "image": str(record.source),
                       "unwrapping": parts.report},
            "requested_profile": extraction_profile, "requested_loader": loader,
        }
        try:
            active, attempts, data = UNIVERSAL.select_loader(
                args, parts.rom, parts.drdi, reporter, drdi_data=parts.drdi_data)
            report["loader_selection"] = attempts
            report.update(UNIVERSAL.run_extraction(active, args, reporter))
            report["status"] = "ok"
        except Exception as exc:
            report["status"] = "failed"
            report["fatal"] = {"type": type(exc).__name__, "message": str(exc)}
            reporter.fail("fatal", str(exc), exception=type(exc).__name__)
        report["validation"] = reporter.as_dict()
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        if report["status"] != "ok":
            raise BackendError(report["fatal"]["message"])
        return report

    @classmethod
    def compare_profiles(cls, record: ModemRecord, profiles: list[int], output: Path,
                         *, device: str, loader: str) -> dict:
        if not profiles or len(profiles) != len(set(profiles)) or any(p < 0 for p in profiles):
            raise BackendError("comparison profiles must be unique non-negative integers")
        parts = cls.parts(record.source)
        args = SimpleNamespace(loader=loader, profile="all", out=output.parent, device=device,
                               stem=safe_stem(record.source.stem),
                               drdi_data=SimpleNamespace(read_bytes=lambda: parts.drdi_data)
                               if parts.drdi_data is not None else None)
        reporter = UNIVERSAL.Reporter()
        try:
            active, _attempts, _data = UNIVERSAL.select_loader(
                args, parts.rom, parts.drdi, reporter, drdi_data=parts.drdi_data)
            if record.details.get("bank_only"):
                bank = next(b for b in active.banks if b.table_index == record.capability_bank_index)
                lte_profiles = UNIVERSAL.scan_lte_rows_bank(bank, active.tables, reporter)
                if bank.table_index == active.capability_bank().table_index:
                    _, _, nr_profiles, _, _, _, _, _ = UNIVERSAL.extract_capability(active, "all", reporter)
                elif bank.table_index == 8:
                    nr_profiles = {p.profile: UNIVERSAL._secondary_combos(p)
                                   for p in UNIVERSAL.decode_tensor_secondary(active, 8, reporter)}
                else:
                    nr_profiles = {}
                available = set(nr_profiles) | set(lte_profiles)
                for p in available:
                    nr_profiles.setdefault(p, [])
                    lte_profiles.setdefault(p, [])
            else:
                _cap, _states, nr_profiles, _union, _lte_bank, lte_profiles, _lte_union, _unresolved = \
                    UNIVERSAL.extract_capability(active, "all", reporter)
            available = set(nr_profiles) & set(lte_profiles)
            missing = sorted(set(profiles) - available)
            if missing:
                raise BackendError(f"profiles not available in both LTE and NR data: {missing}")
            result = _write_profile_comparison(profiles, nr_profiles, lte_profiles, output)
            result["validation"] = reporter.as_dict()
            return result
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(str(exc)) from exc


def _class_name(value: int, absent: int) -> str:
    if value == absent:
        return "-"
    return UNIVERSAL.export.CLASS_LETTERS[value] if value < len(UNIVERSAL.export.CLASS_LETTERS) else f"class{value}"


def _combo_label(combo) -> str:
    tokens = [UNIVERSAL.export.lte_token(component, with_mimo=True) for component in combo.lte]
    tokens.extend(UNIVERSAL.export.nr_token(component) for component in combo.nr)
    return "-".join(tokens)


def _radio_details(combo) -> str:
    value = {
        "lte": [{"band": comp.band,
                 "dl_class": _class_name(comp.dl_class, UNIVERSAL.LTE_UL_ABSENT),
                 "ul_class": _class_name(comp.ul_class, UNIVERSAL.LTE_UL_ABSENT),
                 "dl_mimo": comp.dl_mimo} for comp in combo.lte],
        "nr": [{"band": comp.band,
                "dl_class": _class_name(comp.dl_class, UNIVERSAL.NR_UL_ABSENT_CANON),
                "ul_class": _class_name(comp.ul_class, UNIVERSAL.NR_UL_ABSENT_CANON),
                "carriers": [{"scs_khz": cc.scs_khz, "dl_mimo": cc.dl_mimo,
                              "dl_bw_mhz": cc.dl_bw_mhz, "ul_mimo": cc.ul_mimo,
                              "ul_bw_mhz": cc.ul_bw_mhz} for cc in comp.ccs]} for comp in combo.nr],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_profile_comparison(profiles: list[int], nr_profiles: dict, lte_profiles: dict,
                              output: Path) -> dict:
    rows = {}
    counts = {}
    for profile in profiles:
        lte = UNIVERSAL.dedup_exact(lte_profiles[profile])
        endc, nrca, _lte_from_nr = UNIVERSAL.export.classify(nr_profiles[profile], 1)
        nrdc = [c for c in nrca if any(n.band < 257 for n in c.nr) and any(n.band >= 257 for n in c.nr)]
        families = {"LTE": lte, "ENDC": endc, "NRCA": [c for c in nrca if c not in nrdc], "NRDC": nrdc}
        counts[profile] = {name: len(combos) for name, combos in families.items()}
        for family, combos in families.items():
            for combo in combos:
                key = (family, UNIVERSAL.combo_key(combo))
                rows.setdefault(key, {"family": family, "combination": _combo_label(combo),
                                      "exact_radio": _radio_details(combo), "profiles": set()})["profiles"].add(profile)
    ordered = sorted(rows.values(), key=lambda row: ({"LTE": 0, "ENDC": 1, "NRCA": 2, "NRDC": 3}[row["family"]],
                                                       row["combination"], row["exact_radio"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["family", "combination", "exact_radio"] + [f"profile_{p}" for p in profiles] + ["profile_count", "present_in"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in ordered:
            present = row["profiles"]
            writer.writerow({"family": row["family"], "combination": row["combination"],
                             "exact_radio": row["exact_radio"],
                             **{f"profile_{p}": int(p in present) for p in profiles},
                             "profile_count": len(present),
                             "present_in": ",".join(map(str, (p for p in profiles if p in present)))})
    return {"file": str(output), "rows": len(ordered), "counts": counts}


def format_profile_topology(report: dict) -> str:
    """Render the discovered banks without hiding small/default profiles."""
    banks = report.get("banks", ())
    cap_index = report.get("capability_bank_index")
    lte_index = report.get("lte_bank_index")
    lines = [f"Loader: {report.get('loader', 'unknown')}", "", "Discovered banks:"]
    for bank in banks:
        index = bank.get("table_index")
        roles = []
        if index == cap_index:
            roles.append("capability")
        if index == lte_index:
            roles.append("LTE")
        role = f" ({', '.join(roles)})" if roles else ""
        profiles = ", ".join(map(str, bank.get("live_profiles", ()))) or "none"
        lines.append(f"  bank {index}: {bank.get('bank_va', '?')}{role}; live profiles: {profiles}")
    lines.extend(("", "Decoded capability profiles:"))
    for profile in report.get("profiles", ()):
        kinds = profile.get("gui_counts", profile.get("kinds", {}))
        lines.append("  profile {profile}: {rows} NR/EN-DC rows "
                     "(EN-DC {endc}, NR SA {nr_sa}, NR-CA {nrca}, NRDC {nrdc}); LTE table rows {lte}".format(
                         profile=profile.get("profile", "?"),
                         rows=profile.get("decoded_rows", 0),
                         endc=kinds.get("endc", 0), nr_sa=kinds.get("nr_sa", 0), nrca=kinds.get("nrca", 0),
                         nrdc=kinds.get("nrdc", 0),
                         lte=(report["physical_lte_profiles"].get(str(cap_index), {}).get(str(profile.get("profile")), 0)
                              if report.get("physical_lte_profiles") is not None
                              else report.get("lte_profiles", {}).get(str(profile.get("profile")), 0))))
    if report.get("physical_lte_profiles") is not None:
        lines.extend(("", "LTE tables by physical bank (decoded variants):"))
        for bank, profiles in report["physical_lte_profiles"].items():
            for profile, count in profiles.items():
                lines.append(f"  bank {bank} / profile {profile}: {count} LTE rows")
    support = report.get("supported_bands", {})
    if support:
        lines.extend(("", "Firmware-supported bands (separate from combination rows):"))
        for key, label, missing_key in (
                ("lte", "LTE", "supported_without_lte_ca_row"),
                ("nr", "NR", "supported_without_nr_combination_row")):
            item = support.get(key)
            if not item:
                continue
            lines.append(f"  {label}: " + ", ".join(map(str, item.get("union", ()))))
            missing = item.get(missing_key, ())
            if missing:
                lines.append(f"    supported without a static combination row: "
                             + ", ".join(map(str, missing)))
    if not report.get("profiles"):
        lines.append("  No capability profile was decoded.")
    secondary = report.get("secondary_profiles", ())
    if secondary:
        lines.extend(("", "Decoded secondary-bank profiles:"))
        for profile in secondary:
            kinds = profile.get("gui_counts", profile.get("kinds", {}))
            lines.append("  bank {bank} / profile {profile}: {rows} rows "
                         "(EN-DC {endc}, NR SA {nr_sa}, NR-CA {nrca}, NRDC {nrdc})".format(
                             bank=profile.get("bank_index", "?"),
                             profile=profile.get("profile", "?"),
                             rows=profile.get("decoded_rows", 0),
                             endc=kinds.get("endc", 0), nr_sa=kinds.get("nr_sa", 0), nrca=kinds.get("nrca", 0),
                             nrdc=kinds.get("nrdc", 0)))
    return "\n".join(lines)


def selected_count_summary(records: Iterable[ModemRecord]) -> str:
    selected = [record for record in records if record.selected]
    totals = {key: sum(record.counts.get(key, 0) for record in selected)
              for key in ("lte", "endc", "nr_sa", "nrca", "nrdc")}
    return (f"{len(selected)} selected — row sums (overlap included): LTE {totals['lte']:,}; "
            f"EN-DC {totals['endc']:,}; NR SA {totals['nr_sa']:,}; NR-CA {totals['nrca']:,}; NRDC {totals['nrdc']:,}")


class MtkParserGUI:
    def __init__(self) -> None:
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.scale = self._detect_scale()
        self.root.tk.call("tk", "scaling", self.scale * 1.333333)
        self.root.title("MediaTek modem capability parser")
        self.root.geometry(f"{self.s(1180)}x{self.s(760)}")
        self.root.minsize(self.s(900), self.s(520))
        style = ttk.Style()
        font = tkfont.nametofont("TkDefaultFont")
        style.configure("Treeview", rowheight=round(font.metrics("linespace") * 1.55))

        self.records: list[ModemRecord] = []
        self.visible: dict[str, ModemRecord] = {}
        self.viewers: dict[tuple, Any] = {}
        self.busy = False
        self.status_var = tk.StringVar(value="Import a modem image or parts directory; double-click a profile to view combinations.")
        self.loader_var = tk.StringVar(value="auto")
        self.format_vars = {name: tk.BooleanVar(value=name != "cap_prune") for name in EXPORT_FORMATS}
        self.exclude_mimo_subsets_var = tk.BooleanVar(value=False)
        self._build()

    def _detect_scale(self) -> float:
        try:
            scale = self.root.winfo_fpixels("1i") / 96.0
            return scale if .5 <= scale <= 4 else 1.0
        except Exception:
            return 1.0

    def s(self, value: int) -> int:
        return round(value * self.scale)

    def _build(self) -> None:
        ttk, tk = self.ttk, self.tk
        outer = ttk.Frame(self.root, padding=self.s(12))
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, self.s(10)))
        self.import_button = ttk.Button(top, text="Import modem image(s)", command=self.choose_sources)
        self.import_button.pack(side="left")
        self.import_folder_button = ttk.Button(top, text="Import parts folder", command=self.choose_folder)
        self.import_folder_button.pack(side="left", padx=(self.s(8), 0))
        ttk.Label(top, text="Container loader").pack(side="left", padx=(self.s(16), self.s(6)))
        ttk.Combobox(top, textvariable=self.loader_var, values=("auto", "nr15", "grid", "tensor", "flat"),
                     width=9, state="readonly").pack(side="left")
        self.clear_button = ttk.Button(top, text="Clear imports", state="disabled", command=self.clear)
        self.clear_button.pack(side="right")

        tree_frame = ttk.Frame(outer)
        tree_frame.pack(fill="both", expand=True)
        columns = ("extract", "name", "bank", "profile", "lte", "endc", "nr_sa", "nrca", "nrdc", "packaging", "path")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        labels = {"extract": "Run", "name": "Modem source", "bank": "Bank address",
                  "profile": "Profile", "lte": "LTE", "endc": "EN-DC", "nr_sa": "NR SA", "nrca": "NR-CA",
                  "nrdc": "NRDC", "packaging": "DRDI packaging", "path": "Source path"}
        widths = {"extract": 55, "name": 210, "bank": 120, "profile": 70, "lte": 75,
                  "endc": 75, "nr_sa": 75, "nrca": 75, "nrdc": 75, "packaging": 115, "path": 350}
        for column in columns:
            self.tree.heading(column, text=labels[column])
            self.tree.column(column, width=self.s(widths[column]), minwidth=self.s(45),
                             stretch=column in {"name", "path"},
                             anchor="center" if column not in {"name", "path"} else "w")
        yscroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.bind("<Button-1>", self.toggle_checkbox)
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Return>", self.open_selected_viewer)
        self.tree.bind("<space>", self.toggle_selected)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        formats = ttk.LabelFrame(outer, text="Export formats", padding=self.s(10))
        formats.pack(fill="x", pady=(self.s(10), self.s(8)))
        for column, (key, label) in enumerate((("b0cd", "0xB0CD v41 (LTE)"),
                           ("b826", "0xB826 v21 (NR / EN-DC)"),
                           ("mtk_nr", "MTK NR Trace Log"),
                           ("mtk_lte", "MTK LTE CA_COMB_INFO"),
                           # ("cap_prune", "Exact-MIMO LTE cap-prune"),
                           )):
            ttk.Checkbutton(formats, text=label, variable=self.format_vars[key]).grid(
                row=0, column=column, sticky="w", padx=(0, self.s(18)))
        options = ttk.LabelFrame(outer, text="Export options", padding=self.s(10))
        options.pack(fill="x", pady=(0, self.s(8)))
        ttk.Checkbutton(options, text="Exclude MIMO subsets",
                        variable=self.exclude_mimo_subsets_var).pack(side="left")
        ttk.Label(options, text="(Only include if feature sets are non-identical)").pack(
            side="left", padx=(self.s(16), 0))

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, self.s(7)))
        self.deselect_button = ttk.Button(actions, text="Deselect all", command=self.deselect_all)
        self.deselect_button.pack(side="left")
        self.select_button = ttk.Button(actions, text="Select all", command=self.select_all)
        self.select_button.pack(side="left", padx=(self.s(8), 0))
        self.view_button = ttk.Button(actions, text="View combos", command=self.open_selected_viewer)
        self.view_button.pack(side="left", padx=(self.s(8), 0))
        self.compare_button = ttk.Button(actions, text="Compare profiles", command=self.choose_compare)
        self.compare_button.pack(side="right")
        self.topology_button = ttk.Button(actions, text="Profile topology", command=self.show_topology)
        self.topology_button.pack(side="right", padx=(0, self.s(8)))
        self.export_button = ttk.Button(actions, text="Export selected", command=self.choose_export)
        self.export_button.pack(side="right", padx=(0, self.s(8)))

        ttk.Label(outer, textvariable=self.status_var).pack(fill="x")
        self.log = tk.Text(outer, height=8, wrap="word", state="disabled")
        self.log.pack(fill="x", pady=(self.s(5), 0))

    def append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.import_button, self.select_button, self.deselect_button,
                       self.import_folder_button, self.compare_button, self.topology_button, self.export_button,
                       self.view_button):
            button.configure(state=state)
        self.clear_button.configure(state="disabled" if busy or not self.records else "normal")
        self.root.configure(cursor="watch" if busy else "")
        if status:
            self.status_var.set(status)

    def choose_sources(self) -> None:
        from tkinter import filedialog
        if self.busy: return
        names = filedialog.askopenfilenames(
            title="Choose MediaTek modem images or extracted parts",
            filetypes=(("Modem images and container files", "*.img *.bin *.mbn *.gz *.xz *.sparse"),
                       ("All files", "*.*")),
        )
        if not names: return
        self.import_sources([Path(name) for name in names])

    def import_sources(self, sources: Iterable[Path]) -> None:
        paths = list(dict.fromkeys(path.resolve() for path in sources))
        if not paths: return
        existing = {record.key for record in self.records}
        paths = [path for path in paths if str(path).casefold() not in existing]
        if not paths:
            self.status_var.set("All selected sources are already imported.")
            return
        loader = self.loader_var.get()
        self.set_busy(True, f"Inspecting and discovering profiles in {len(paths)} source(s)…")
        for path in paths: self.append_log(f"Inspecting {path}")

        def work() -> None:
            good, failed = [], []
            for path in paths:
                try:
                    source_record = MtkBackend.inspect(path)
                    summary = MtkBackend.summarize(source_record, loader=loader, retain_combos=True)
                    good.extend(MtkBackend.profile_records(source_record, summary))
                except Exception: failed.append((path, traceback.format_exc()))
            self.root.after(0, lambda: self.import_finished(good, failed))
        threading.Thread(target=work, daemon=True).start()

    def choose_folder(self) -> None:
        from tkinter import filedialog
        if self.busy: return
        folder = filedialog.askdirectory(title="Choose extracted modem-parts directory")
        if folder:
            self.import_sources([Path(folder)])

    def import_finished(self, good: list[ModemRecord], failed: list[tuple[Path, str]]) -> None:
        from tkinter import messagebox
        self.records.extend(good)
        for path, error in failed:
            self.append_log(f"Could not inspect {path}: {error.strip().splitlines()[-1]}")
        self.refresh()
        self.set_busy(False, f"Imported {len(good)} capability profile(s); {selected_count_summary(self.records)}.")
        if failed:
            messagebox.showwarning("Some imports failed", "See the log for details. No source was guessed.")

    def clear(self) -> None:
        for window in list(self.viewers.values()):
            if window.winfo_exists():
                window.destroy()
        self.viewers.clear()
        self.records.clear(); self.visible.clear(); self.refresh()
        self.status_var.set("Imports cleared.")
        self.append_log("Cleared all modem sources.")

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.visible.clear()
        for index, record in enumerate(self.records):
            iid = f"source-{index}"
            self.visible[iid] = record
            count = record.counts
            self.tree.insert("", "end", iid=iid, values=("☑" if record.selected else "☐", record.display_name,
                             record.capability_bank or "—", record.profile if record.profile is not None else "—",
                             f"{count.get('lte', 0):,}", f"{count.get('endc', 0):,}", f"{count.get('nr_sa', 0):,}",
                             f"{count.get('nrca', 0):,}",
                             f"{count.get('nrdc', 0):,}", record.packaging, str(record.source)))
        self.clear_button.configure(state="normal" if self.records and not self.busy else "disabled")

    def toggle_checkbox(self, event: Any) -> str | None:
        if self.busy: return "break"
        if self.tree.identify_region(event.x, event.y) == "cell" and self.tree.identify_column(event.x) == "#1":
            iid = self.tree.identify_row(event.y)
            if iid and iid in self.visible:
                self.visible[iid].selected = not self.visible[iid].selected
                self.refresh()
            return "break"
        return None

    def toggle_selected(self, _event=None) -> str:
        if not self.busy and self.tree.selection():
            record = self.visible.get(self.tree.selection()[0])
            if record: record.selected = not record.selected; self.refresh()
        return "break"

    def on_double_click(self, event: Any) -> str | None:
        if self.busy:
            return "break"
        column = self.tree.identify_column(event.x)
        if self.tree.identify_region(event.x, event.y) not in {"cell", "tree"}:
            return None
        if column == "#1":
            return "break"
        iid = self.tree.identify_row(event.y)
        record = self.visible.get(iid)
        if record is not None:
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            name = self.tree.column(column, "id")
            tab = {"lte": "LTE", "endc": "EN-DC", "nr_sa": "NR SA (1CC)",
                   "nrca": "NR-CA", "nrdc": "NRDC"}.get(name)
            self.open_viewer_for_record(record, initial_tab=tab)
            return "break"
        return None

    def open_selected_viewer(self, _event=None) -> str:
        from tkinter import messagebox
        if self.busy:
            return "break"
        selection = self.tree.selection()
        record = self.visible.get(selection[0]) if selection else None
        if record is None:
            messagebox.showinfo("View combos", "Select a capability-profile row first.", parent=self.root)
        else:
            self.open_viewer_for_record(record)
        return "break"

    def open_viewer_for_record(self, record: ModemRecord, *, initial_tab: str | None = None) -> None:
        from tkinter import messagebox
        from mtk_viewer import ComboViewerWindow
        key = (record.key, record.capability_bank_index, record.profile)
        window = self.viewers.get(key)
        if window is not None and window.winfo_exists():
            if initial_tab:
                window.select_family(initial_tab)
            window.deiconify()
            window.lift()
            window.focus_set()
            return
        data = record.combo_cache.get((record.capability_bank_index, record.profile))
        if data is None:
            messagebox.showwarning("View combos", "Re-import this source to create its in-memory combo snapshot.",
                                   parent=self.root)
            return
        try:
            window = ComboViewerWindow(self.root, record=record, data=data, scale=self.scale,
                                       initial_tab=initial_tab)
        except Exception as exc:
            self.append_log(f"Could not view {record.display_name}: {exc}")
            messagebox.showerror("Viewer failed", str(exc), parent=self.root)
            return
        self.viewers[key] = window
        window.bind("<Destroy>", lambda event: self.viewers.pop(key, None)
                    if event.widget is window else None, add="+")

    def select_all(self) -> None:
        for record in self.records: record.selected = True
        self.refresh()

    def deselect_all(self) -> None:
        for record in self.records: record.selected = False
        self.refresh()

    def selected(self) -> list[ModemRecord]:
        return [record for record in self.records if record.selected]

    def _settings(self) -> str:
        loader = self.loader_var.get()
        if loader not in {"auto", "nr15", "grid", "tensor", "flat"}: raise BackendError("choose a supported loader")
        return loader

    def _export_formats(self) -> frozenset[str]:
        return parse_export_formats(key for key, variable in self.format_vars.items() if variable.get())

    def choose_export(self) -> None:
        from tkinter import filedialog, messagebox
        selected = self.selected()
        if not selected:
            messagebox.showwarning("No modem selected", "Check at least one row in the Run column."); return
        try:
            loader = self._settings()
            export_formats = self._export_formats()
            exclude_mimo_subsets = self.exclude_mimo_subsets_var.get()
        except BackendError as exc: messagebox.showerror("Invalid settings", str(exc)); return
        folder = filedialog.askdirectory(title="Choose export directory")
        if not folder: return
        output = Path(folder)
        self.set_busy(True, f"Extracting {len(selected)} selected capability profile(s)…")
        self.append_log(f"Exporting to {output}")

        def work() -> None:
            done, failed = [], []
            for record in selected:
                suffix = (f"_bank{record.capability_bank_index}_profile{record.profile}"
                          if record.profile is not None else "_all_profiles")
                target = output / safe_stem(record.source.stem + suffix)
                try:
                    report = MtkBackend.extract_and_export(record, target, device=record.source.stem,
                                                           loader=loader, extraction_profile=str(record.profile) if record.profile is not None else "all",
                                                           export_formats=export_formats,
                                                           exclude_mimo_subsets=exclude_mimo_subsets)
                    done.append((record, target, report))
                except Exception: failed.append((record, traceback.format_exc()))
            self.root.after(0, lambda: self.export_finished(done, failed, output))
        threading.Thread(target=work, daemon=True).start()

    def export_finished(self, done, failed, output: Path) -> None:
        from tkinter import messagebox
        for record, target, report in done:
            record.status = f"Exported to {target}"
            record.loader = report["loader"]
            record.counts = {**report["gui_counts"], "lte": report["lte_union_exact_rows"]}
            record.details["extraction"] = report
            self.append_log(f"Exported {record.display_name} → {target}")
            if report["export"].get("skipped_formats"):
                self.append_log("No matching data in this bank for: " + ", ".join(report["export"]["skipped_formats"]))
        for record, error in failed:
            record.status = "Failed — see log"
            self.append_log(f"Failed {record.display_name}: {error.strip().splitlines()[-1]}")
        self.refresh(); self.set_busy(False, f"Exported {len(done)} capability profile(s) to {output}; {len(failed)} failed.")
        if failed: messagebox.showwarning("Some exports failed", "See the log and each output report.json for details.")
        elif done: messagebox.showinfo("Export complete", f"Exported {len(done)} MTK capability profile(s).\n\n{output}")

    def show_topology(self) -> None:
        from tkinter import messagebox
        selection = self.tree.selection()
        if len(selection) != 1:
            messagebox.showwarning("Choose one profile", "Select one capability-profile row to view its discovered topology.")
            return
        record = self.visible.get(selection[0])
        report = record and (record.details.get("extraction") or record.details.get("summary"))
        if not report:
            messagebox.showinfo("Profile topology", "This source was not summarized successfully. Re-import it with a supported loader.")
            return
        text = format_profile_topology(report)
        self.append_log(f"Profile topology for {record.display_name}:\n{text}")
        messagebox.showinfo(f"Profile topology — {record.display_name}", text)

    def choose_compare(self) -> None:
        from tkinter import filedialog, messagebox, simpledialog
        selected = self.selected()
        if len(selected) != 1:
            messagebox.showwarning("Choose one modem", "Profile comparison is within one MTK modem source; select exactly one row."); return
        raw = simpledialog.askstring("Compare profiles", "Profiles to compare (for example: 0,1,2):", initialvalue="0,1")
        if raw is None: return
        try: profiles = parse_profile_list(raw)
        except BackendError as exc: messagebox.showerror("Invalid profiles", str(exc)); return
        if len(profiles) < 2:
            messagebox.showerror("Invalid profiles", "Choose at least two profiles to compare."); return
        directory = filedialog.askdirectory(title="Choose comparison report directory")
        if not directory: return
        loader = self.loader_var.get()
        record, output = selected[0], Path(directory) / f"{safe_stem(selected[0].source.stem)}_profile_comparison.csv"
        self.set_busy(True, f"Comparing profiles {','.join(map(str, profiles))}…")

        def work() -> None:
            try:
                result = MtkBackend.compare_profiles(record, profiles, output, device=record.source.stem, loader=loader)
            except Exception:
                error = traceback.format_exc(); self.root.after(0, lambda: self.compare_finished(record, None, error)); return
            self.root.after(0, lambda: self.compare_finished(record, result, None))
        threading.Thread(target=work, daemon=True).start()

    def compare_finished(self, record: ModemRecord, result: dict | None, error: str | None) -> None:
        from tkinter import messagebox
        if error:
            self.set_busy(False, "Profile comparison failed."); self.append_log(error)
            messagebox.showerror("Profile comparison failed", error.strip().splitlines()[-1]); return
        self.set_busy(False, f"Wrote {result['rows']:,} exact profile-comparison rows.")
        self.append_log(f"Compared profiles for {record.display_name}: {result['file']}")
        messagebox.showinfo("Profile comparison complete", f"{result['rows']:,} exact LTE/EN-DC/NR-CA/NRDC rows\n\n{result['file']}")

    def run(self) -> None:
        self.root.mainloop()


def cli_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", type=Path, help="modem image or extracted-parts directory")
    ap.add_argument("--list", action="store_true", help="inspect the source and print container provenance")
    ap.add_argument("--out", type=Path, help="output directory; runs extraction instead of launching the GUI")
    ap.add_argument("--device", default="MediaTek modem")
    ap.add_argument("--loader", default="auto", choices=("auto", "nr15", "grid", "tensor", "flat"))
    ap.add_argument("--profile", default="all")
    ap.add_argument("--formats", default=",".join(EXPORT_FORMATS),
                    help="comma-separated: b0cd,b826,mtk_nr,mtk_lte,cap_prune")
    args = ap.parse_args(argv)
    if args.source is None:
        MtkParserGUI().run(); return 0
    try:
        record = MtkBackend.inspect(args.source)
        if args.list or args.out is None:
            print(json.dumps({"source": str(record.source), "packaging": record.packaging,
                              "layers": record.layers, "rom_bytes": record.rom_bytes,
                              "drdi_bytes": record.drdi_bytes, "drdi_data_bytes": record.split_bytes,
                              "unwrapping": record.details}, indent=2))
            if args.out is None: return 0
        report = MtkBackend.extract_and_export(record, args.out, device=args.device, loader=args.loader,
                                               extraction_profile=parse_profile(args.profile),
                                               export_formats=parse_export_formats(args.formats.split(",")))
        print(json.dumps({"status": "ok", "loader": report["loader"],
                          "counts": {**report["union"]["kinds"], "lte": report["lte_union_exact_rows"]},
                          "out": str(args.out)}, indent=2))
        return 0
    except (BackendError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
