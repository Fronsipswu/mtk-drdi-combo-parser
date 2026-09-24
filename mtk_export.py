from __future__ import annotations
import collections
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterable

CLASS_LETTERS = "ABCDEFGHIJKL"

# Physical carriers per LTE bandwidth class, indexed by the 0-based class byte.
# A=1 B=2 C=2 D=3 E=4 F=5.
LTE_CLASS_CCS = (1, 2, 2, 3, 4, 5)
LTE_UL_ABSENT = 6
NR_UL_ABSENT = 0x1C

VERSION = 21
SOURCE_ENDC = 3
SOURCE_NRCA = 4
SOURCE_NRDC = 5
@dataclass
class NrCC:
    """One physical NR carrier, already resolved through the feature tables."""
    scs_khz: int
    dl_mimo: int | None            # layers: 2 / 4 / 8
    dl_bw_mhz: int | None
    ul_mimo: int | None = None     # layers: 1 / 2 / 4 ; None = no UL on this CC
    ul_bw_mhz: int | None = None


@dataclass
class NrComponent:
    band: int
    dl_class: int                  # RAW enum: 0=A, 1=B, 2=C ...
    ul_class: int                  # RAW enum, or NR_UL_ABSENT
    ccs: list[NrCC] = field(default_factory=list)

    @property
    def has_ul(self) -> bool:
        return self.ul_class != NR_UL_ABSENT


@dataclass
class LteComponent:
    band: int                      # already mapped to 3GPP
    dl_class: int                  # RAW enum
    ul_class: int                  # RAW enum, or LTE_UL_ABSENT
    dl_mimo: list[int] = field(default_factory=list)   # LAYERS per physical CC

    @property
    def has_ul(self) -> bool:
        return self.ul_class < LTE_UL_ABSENT


@dataclass
class Combo:
    lte: list[LteComponent] = field(default_factory=list)
    nr: list[NrComponent] = field(default_factory=list)

    @property
    def kind(self) -> str:
        if self.lte and self.nr:
            return "ENDC"
        if self.nr:
            return "NR"
        return "LTE"

    @property
    def nr_physical_ccs(self) -> int:
        return sum(len(c.ccs) for c in self.nr)


# -------------------------------------------------------------------- sorting
def _sort_key(tok: str):
    m = re.match(r"([bn])(\d+)", tok)
    return (m.group(1) != "b", int(m.group(2)), tok) if m else (True, 9999, tok)


def _join(tokens: list[str]) -> str:
    return "-".join(sorted(tokens, key=_sort_key))


# ------------------------------------------------------------------ cap-prune
def lte_token(c: LteComponent, with_mimo: bool = True) -> str:
    """Render one LTE component."""
    dl_let = CLASS_LETTERS[c.dl_class]
    ul_let = CLASS_LETTERS[c.ul_class] if c.has_ul else ""
    if not with_mimo:
        return f"b{c.band}{dl_let}{ul_let}"
    mimo_digits = "".join(str(m) for m in sorted(c.dl_mimo, reverse=True)) if c.dl_mimo else ""
    return f"b{c.band}{dl_let}{mimo_digits}{ul_let}"


def nr_token(c: NrComponent) -> str:
    """Render one NR component for cap-prune (no MIMO digits, no SCS/BW)."""
    dl_let = CLASS_LETTERS[c.dl_class] if c.dl_class < len(CLASS_LETTERS) else f"[{c.dl_class}]"
    ul_let = (CLASS_LETTERS[c.ul_class] if c.ul_class < len(CLASS_LETTERS) else f"[{c.ul_class}]") if c.has_ul else ""
    return f"n{c.band}{dl_let}{ul_let}"


def render_lte_capprune(combos, with_mimo: bool = True) -> str:
    """Cap-prune format for pure LTE. '-mAll' appended once per combination."""
    out = []
    for cb in combos:
        toks = [lte_token(c, with_mimo) for c in cb.lte]
        if toks:
            out.append(_join(toks) + "-mAll")
    return ";".join(_dedup_sorted(out)) + ";\n"


def render_nr_capprune(combos) -> str:
    """NR-CA / EN-DC cap-prune. No MIMO digits, no '-mAll'."""
    out = []
    for cb in combos:
        toks = [lte_token(c, with_mimo=False) for c in cb.lte] + [nr_token(c) for c in cb.nr]
        if toks:
            out.append(_join(toks))
    return ";".join(_dedup_sorted(out)) + ";\n"


def _dedup_sorted(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return sorted(out, key=lambda s: [_sort_key(t) for t in s.replace("-mAll", "").split("-")])


# ------------------------------------------------------------- B826 tables & encoder
BW_NAMES = [
    "DEFAULT", "5", "10", "15", "20", "20_20", "20_20_20",
    "20_20_20_20", "20_20_20_20_20", "25", "30", "40", "50",
    "50_50", "50_50_50", "50_50_50_50", "50_50_50_50_50", "60",
    "70", "80", "90", "100", "100_60", "100_100", "100_100_100",
    "100_100_100_100", "100_100_100_100_100",
    "100_100_100_100_100_100", "100_100_100_100_100_100_100",
    "100_100_100_100_100_100_100_100", "40_40", "60_40",
    "100_40", "200", "200_200", "200_200_200", "200_200_200_200",
    "10_10", "25_25", "40_10", "40_20", "35", "30_20", "60_60",
    "30_30", "45", "50_5", "50_10", "50_15", "50_20", "40_15",
    "15_15", "30_25", "20_10", "20_15", "5_5", "80_80", "80_20",
    "40_30", "100_90", "30_10", "100_20", "80_40", "50_40",
    "100_50", "100_80",
    "35_35", "45_45",
]
BW_TO_INDEX = {
    tuple(int(x) for x in name.split("_")): idx
    for idx, name in enumerate(BW_NAMES)
    if name != "DEFAULT"
}
SCS_TO_INDEX = {15: 1, 30: 2, 60: 3, 120: 4, 240: 5}


def antenna_tables():
    names = ["INVALID", "1", "2", "4"]
    for count in range(2, 9):
        names.append("_".join(["1"] * count))
        names.extend(
            "_".join(["2"] * leading + ["1"] * (count - leading))
            for leading in range(1, count + 1)
        )
        names.extend(
            "_".join(["4"] * leading + ["2"] * (count - leading))
            for leading in range(1, count + 1)
        )
    names.extend(["8", "8_4", "8_4_4", "8_8", "6", "4_8", "6_4", "4_6", "6_6"])

    fwd = {}
    rev = {}
    for idx, name in enumerate(names):
        layers = () if name == "INVALID" else tuple(int(x) for x in name.split("_"))
        fwd[layers] = idx
        rev[idx] = layers
    return fwd, rev


ANT_TO_INDEX, INDEX_TO_ANT = antenna_tables()
ANTENNA_INDEX = ANT_TO_INDEX


def mimo_index(layers):
    if not layers:
        return 0
    key = tuple(int(x) for x in layers)
    if key in ANT_TO_INDEX:
        return ANT_TO_INDEX[key]
    canonical = tuple(sorted(key, reverse=True))
    if canonical in ANT_TO_INDEX:
        return ANT_TO_INDEX[canonical]
    raise ValueError(f"B826 antenna enum cannot encode MIMO vector {layers}")


BW_EXT_ENABLE = False
BW_EXT_INDEX = {
    (100, 100, 20): 66,
    (100, 100, 50): 67,
    (100, 100, 40): 68,
    (100, 50, 20): 69,
}


def bw_index(values):
    if not values:
        return 0, True, (), None
    key = tuple(int(x) for x in values)
    if key in BW_TO_INDEX:
        return BW_TO_INDEX[key], True, key, None
    canonical = tuple(sorted(key, reverse=True))
    if canonical in BW_TO_INDEX:
        return BW_TO_INDEX[canonical], True, canonical, None
    if len(set(key)) == 1 and (key[0],) in BW_TO_INDEX:
        return BW_TO_INDEX[(key[0],)], True, (key[0],), None
    canon_desc = tuple(sorted(key, reverse=True))
    if BW_EXT_ENABLE and canon_desc in BW_EXT_INDEX:
        return BW_EXT_INDEX[canon_desc], True, canon_desc, None
    distinct = tuple(sorted(set(key), reverse=True))
    if len(distinct) == 2 and distinct in BW_TO_INDEX:
        return BW_TO_INDEX[distinct], True, distinct, key
    return 0, False, key, None


def encode_component(component, unsupported_counter):
    band = int(component["band"])
    if not 0 < band < 512:
        raise ValueError(f"B826 v21 band out of 9-bit range: {band}")

    is_nr = component["rat"] == "NR"
    dl_class = int(component["dl_class"])
    ul_class = int(component["ul_class"])
    dl_mimo = mimo_index(component["dl_mimo"])
    ul_mimo = mimo_index(component["ul_mimo"]) if ul_class else 0

    if dl_mimo > 0x7F:
        raise ValueError(f"DL MIMO index {dl_mimo} exceeds B826 field")
    if ul_mimo > 0x1F:
        raise ValueError(f"UL MIMO index {ul_mimo} exceeds B826 field")

    head = (
        band
        | ((1 if is_nr else 0) << 9)
        | ((dl_class & 0x1F) << 10)
        | ((dl_mimo & 1) << 15)
    )
    byte1 = ((dl_mimo >> 1) & 0x3F) | ((ul_class & 0x03) << 6)
    byte2 = ((ul_class >> 2) & 0x07) | ((ul_mimo & 0x1F) << 3)

    byte3 = 0
    byte4 = 0
    byte5 = 0

    if is_nr:
        scs_idx = SCS_TO_INDEX[int(component["scs"])]
        dl_idx, dl_ok, dl_raw, dl_collapsed = bw_index(component["dl_bw"])
        ul_idx, ul_ok, ul_raw, ul_collapsed = bw_index(component["ul_bw"])

        if not dl_ok:
            unsupported_counter[("DL", band, tuple(component["dl_bw"]))] += 1
        elif dl_collapsed is not None:
            unsupported_counter[("DL-collapsed", band, dl_collapsed, dl_raw)] += 1
        if ul_class and not ul_ok:
            unsupported_counter[("UL", band, tuple(component["ul_bw"]))] += 1

        byte3 |= (scs_idx & 0x01) << 7
        byte4 |= (scs_idx >> 1) & 0x03
        byte4 |= (dl_idx & 0x3F) << 2
        byte5 |= (dl_idx >> 6) & 0x01
        byte5 |= (ul_idx & 0x7F) << 1

    return struct.pack("<HBBBBB", head, byte1, byte2, byte3, byte4, byte5) + b"\x00\x00"


def encode_combo(components, unsupported_counter):
    count = len(components)
    if not 1 <= count <= 15:
        raise ValueError(f"B826 v21 supports 1..15 components, got {count}")

    combo_features = (count & 0x0F) << 3
    return (
        b"\x00" * 3
        + struct.pack("<H", combo_features)
        + b"\x00" * 24
        + b"".join(encode_component(c, unsupported_counter) for c in components)
    )


def build_log(component_rows, source):
    unsupported = collections.Counter()
    encoded = []
    seen = set()

    for components in component_rows:
        raw = encode_combo(components, unsupported)
        if raw not in seen:
            seen.add(raw)
            encoded.append(raw)

    total = len(encoded)
    if total > 0xFFFF:
        raise ValueError("B826 log item exceeds uint16 combo count")

    header = struct.pack("<HHHHHB", VERSION, 0, total, 0, total, source)
    return header + b"".join(encoded), unsupported, len(component_rows), total


def decode_header_and_first(blob):
    version, reserved, total, index, num, source = struct.unpack_from("<HHHHHB", blob, 0)
    pos = 11 + 3
    features = struct.unpack_from("<H", blob, pos)[0]
    pos += 2
    count = (features >> 3) & 0x0F
    pos += 24

    components = []
    for _ in range(count):
        head, b1, b2, b3, b4, b5 = struct.unpack_from("<HBBBBB", blob, pos)
        pos += 9
        band = head & 0x1FF
        is_nr = (head >> 9) & 1
        dl_class = (head >> 10) & 0x1F
        dl_mimo_idx = ((b1 & 0x3F) << 1) | ((head >> 15) & 1)
        ul_class = ((b2 & 0x07) << 2) | ((b1 >> 6) & 0x03)
        ul_mimo_idx = (b2 >> 3) & 0x1F

        item = {
            "rat": "NR" if is_nr else "LTE",
            "band": band,
            "dl_class_index": dl_class,
            "ul_class_index": ul_class,
            "dl_mimo_index": dl_mimo_idx,
            "dl_mimo_layers": list(INDEX_TO_ANT.get(dl_mimo_idx, ())),
            "ul_mimo_index": ul_mimo_idx,
            "ul_mimo_layers": list(INDEX_TO_ANT.get(ul_mimo_idx, ())),
        }
        if is_nr:
            scs_idx = ((b4 & 0x03) << 1) | ((b3 >> 7) & 1)
            dl_bw_idx = ((b5 & 1) << 6) | ((b4 >> 2) & 0x3F)
            ul_bw_idx = (b5 >> 1) & 0x7F
            item.update({
                "scs_index": scs_idx,
                "scs_khz": (1 << (scs_idx - 1)) * 15,
                "dl_bw_index": dl_bw_idx,
                "ul_bw_index": ul_bw_idx,
            })
        components.append(item)

    return {
        "version": version,
        "total": total,
        "index": index,
        "num": num,
        "source": source,
        "first_combo": components,
    }


# Backwards compatibility alias
_enc = sys.modules[__name__]


# ----------------------------------------------------------------------- B826
@dataclass
class B826Result:
    """Pure result object - no file I/O has happened."""
    tag: str
    source: int
    blob: bytes
    records: int
    input_rows: int
    unsupported: dict

    @property
    def hex(self) -> str:
        return self.blob.hex().upper()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.blob).hexdigest()

    def block(self, device: str) -> str:
        """The '# header / Payload:' text block for this family."""
        return (f"# 0xB826 v21 {self.tag} (source={self.source}) {device}\n"
                + f"# records={self.records}\n"
                + "Payload: " + self.hex + "\n")

    def verify(self) -> bool:
        try:
            decode_header_and_first(self.blob)
            return True
        except Exception:
            return False


def _b826_components(cb: Combo) -> list[dict]:
    comps = []
    for c in cb.lte:
        n = max(1, len(c.dl_mimo))
        ul_ccs = LTE_CLASS_CCS[c.ul_class] if c.ul_class < len(LTE_CLASS_CCS) else 1
        comps.append({
            "rat": "LTE", "band": c.band,
            "dl_class": c.dl_class + 1,
            "ul_class": (c.ul_class + 1) if c.has_ul else 0,
            "dl_mimo": c.dl_mimo or [2] * n,
            "ul_mimo": [1] * ul_ccs if c.has_ul else [],
        })
    for c in cb.nr:
        if not c.ccs:
            return []
        ul = [cc for cc in c.ccs if cc.ul_mimo]
        comps.append({
            "rat": "NR", "band": c.band,
            "dl_class": c.dl_class + 1,
            "ul_class": (c.ul_class + 1) if c.has_ul else 0,
            "dl_mimo": [cc.dl_mimo or 2 for cc in c.ccs],
            "dl_bw":   [cc.dl_bw_mhz or 20 for cc in c.ccs],
            "ul_mimo": [cc.ul_mimo for cc in ul] if (c.has_ul and ul) else [],
            "ul_bw":   [cc.ul_bw_mhz for cc in ul] if (c.has_ul and ul) else [],
            "scs": c.ccs[0].scs_khz,
        })
    return comps


def build_b826(combos, source: int, tag: str | None = None) -> B826Result:
    """Encode combos -> B826Result. Pure: returns bytes, writes nothing."""
    tag = tag or {3: "RF_ENDC", 4: "RF_NRCA", 5: "RF_NRDC"}.get(source, f"SOURCE{source}")
    rows, seen = [], set()
    for cb in combos:
        comps = _b826_components(cb)
        if not comps:
            continue
        key = repr(comps)
        if key in seen:
            continue
        seen.add(key)
        rows.append(comps)
    blob, unsupported, n_in, n_out = build_log(rows, source)
    return B826Result(tag, source, blob, n_out, n_in,
                      {str(k): v for k, v in unsupported.items()})


def combined_b826_text(results, device: str) -> str:
    """Join B826Results into the single combined-file text (EN-DC first)."""
    return "\n".join(r.block(device) for r in results)


def classify(combos, nrca_min_ccs: int = 1):
    """Split a flat combo list into (endc, nrca, lte_only)."""
    endc = [c for c in combos if c.kind == "ENDC"]
    nrca = [c for c in combos if c.kind == "NR" and c.nr_physical_ccs >= nrca_min_ccs]
    lte  = [c for c in combos if c.kind == "LTE"]
    return endc, nrca, lte


# ----------------------------------------------------------------------- B0CD v41
class B0cdError(ValueError):
    """A normalized LTE row cannot be represented by the v41 layout."""


def _b0cd_antenna_index(layers: Iterable[int]) -> int:
    values = tuple(sorted((int(value) for value in layers), reverse=True))
    try:
        return ANT_TO_INDEX[values]
    except KeyError as exc:
        raise B0cdError(f"0xB0CD v41 has no antenna enum for LTE MIMO {values}") from exc


def _b0cd_component(component) -> bytes:
    band = int(component.band)
    dl_class = int(component.dl_class)
    ul_class = int(component.ul_class)
    if not 1 <= band <= 0x1FF:
        raise B0cdError(f"0xB0CD v41 LTE band is out of range: {band}")
    if not 0 <= dl_class < 26:
        raise B0cdError(f"0xB0CD v41 DL class is out of range: {dl_class}")
    if ul_class != LTE_UL_ABSENT and not 0 <= ul_class < 26:
        raise B0cdError(f"0xB0CD v41 UL class is out of range: {ul_class}")

    dl_mimo = _b0cd_antenna_index(component.dl_mimo or (2,))
    if ul_class == LTE_UL_ABSENT:
        qcom_ul_class, ul_mimo = 0, 0
    else:
        qcom_ul_class = ul_class + 1
        ul_mimo = _b0cd_antenna_index((1,) * LTE_CLASS_CCS[ul_class])
    return struct.pack("<HBBBBB", band, dl_class + 1, qcom_ul_class,
                       dl_mimo, ul_mimo, 0)


@dataclass(frozen=True)
class B0cdResult:
    packets: tuple[bytes, ...]
    records: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(b"".join(self.packets)).hexdigest()


def build_b0cd_v41(lte_combos, packet_combos: int = 100) -> B0cdResult:
    """Build headerless v41 payloads from LTE-only normalized combinations."""
    if not 1 <= packet_combos <= 0xFF:
        raise B0cdError("packet_combos must fit in one byte")
    records, seen = [], set()
    for combo in lte_combos:
        if getattr(combo, "nr", ()):
            continue
        components = tuple(_b0cd_component(component) for component in combo.lte)
        if not components:
            continue
        if len(components) > 6:
            raise B0cdError("0xB0CD v41 supports at most six LTE components per combination")
        record = bytes([len(components)]) + b"".join(components)
        if record not in seen:
            seen.add(record)
            records.append(record)
    packets = tuple(
        bytes([41, len(records[start:start + packet_combos])])
        + b"".join(records[start:start + packet_combos])
        for start in range(0, len(records), packet_combos)
    )
    return B0cdResult(packets, len(records))


def write_b0cd_v41(lte_combos, destination, device: str) -> dict:
    """Write an importer-friendly B0CD v41 payload text file."""
    result = build_b0cd_v41(lte_combos)
    dest_path = Path(destination)
    lines = [
        "# Headerless 0xB0CD v41 LTE capability payloads.",
        f"# Device: {device}",
        "# Derived from MediaTek DRDI, not captured Qualcomm DIAG data.",
        "# MTK supplies band/class/DL-MIMO; BCS is omitted. UL MIMO is one layer per UL CC and UL-QAM is 0 (unknown).",
        f"# records={result.records}; packets={len(result.packets)}; sha256={result.sha256}",
        "",
    ]
    for index, packet in enumerate(result.packets, 1):
        lines.extend((f"# LTE CA packet {index}/{len(result.packets)}", "Payload: " + packet.hex().upper(), ""))
    dest_path.write_text("\n".join(lines), encoding="ascii")
    return {"file": str(dest_path), "records": result.records,
            "packets": len(result.packets), "sha256": result.sha256,
            "derived": True, "ul_qam": "unknown=0"}


# --------------------------------------------------------------- file writing
def export_all(combos, device: str, out_dir: str, stem: str,
               lte_combos=None, write_per_family: bool = False,
               nrca_min_ccs: int = 1) -> dict:
    """One call: classify, render cap-prune, encode B826, write everything."""
    os.makedirs(out_dir, exist_ok=True)
    endc, nrca, lte_from_combos = classify(combos, nrca_min_ccs)
    lte = lte_combos if lte_combos is not None else lte_from_combos
    files, meta = {}, {"device": device,
                       "counts": {"endc": len(endc), "nrca": len(nrca), "lte": len(lte)}}

    def _w(name, text):
        path = os.path.join(out_dir, name)
        open(path, "w", encoding="utf-8").write(text)
        files[name] = path

    if lte:
        _w(f"{stem}_lte_ca_exact_mimo_cap_prune.txt", render_lte_capprune(lte, with_mimo=True))

    results = [build_b826(endc, 3), build_b826(nrca, 4)]
    _w(f"{stem}_0xB826_v21_combined.txt", combined_b826_text(results, device))
    if write_per_family:
        for r in results:
            _w(f"{stem}_0xB826_v21_{r.tag}.txt", r.block(device))
    for r in results:
        meta[r.tag] = {"source": r.source, "input_rows": r.input_rows, "records": r.records,
                       "bytes": len(r.blob), "sha256": r.sha256,
                       "unsupported": r.unsupported, "decodes": r.verify()}
    meta["files"] = files
    return meta


def build_b826_combined(endc_combos, nrca_combos, device: str, out_dir: str,
                        stem: str, write_per_family: bool = False) -> dict:
    """Backwards-compatible wrapper around the newer API."""
    os.makedirs(out_dir, exist_ok=True)
    results = [build_b826(endc_combos, 3), build_b826(nrca_combos, 4)]
    path = os.path.join(out_dir, f"{stem}_combined.txt")
    open(path, "w", encoding="utf-8").write(combined_b826_text(results, device))
    meta = {"device": device, "combined_file": path}
    if write_per_family:
        for r in results:
            open(os.path.join(out_dir, f"{stem}_{r.tag}.txt"), "w").write(r.block(device))
    for r in results:
        meta[r.tag] = {"source": r.source, "input_rows": r.input_rows, "records": r.records,
                       "bytes": len(r.blob), "sha256": r.sha256, "unsupported": r.unsupported}
    return meta


def verify_b826(path: str) -> bool:
    """Round-trip every payload in a combined file through the reference decoder."""
    ok = True
    for line in open(path, encoding="utf-8"):
        if not line.startswith("Payload:"):
            continue
        try:
            decode_header_and_first(bytes.fromhex(line.split("Payload:")[1].strip()))
        except Exception as e:
            print(f"  decode FAILED: {e}")
            ok = False
    return ok
