"""Text exports matching the supplied ImportMtkNr.kt / ImportMTKLte.kt.

These are reconstructed capability logs, not captured runtime trace messages.
IDs belong to each output file and reference the normalized firmware features.
"""
from __future__ import annotations

from collections import Counter
import hashlib


class TraceError(ValueError):
    pass


LTE_WEIGHTS = (1, 2, 2, 3, 4, 5)
LAYERS = {1: "ONE", 2: "TWO", 4: "FOUR", 8: "EIGHT"}


def _layer(value):
    if value is None:
        return "UNKNOWN"
    if value not in LAYERS:
        raise TraceError(f"unsupported MIMO layer count: {value}")
    return LAYERS[value] + "_LAYERS"


def _class(value):
    if not 0 <= value < 26:
        raise TraceError(f"MTK trace importer requires a single-letter class; got {value}")
    return chr(65 + value)


def _header(device):
    return ["# Reconstructed from MediaTek firmware; not a captured modem log.",
            "# Device: " + str(device).replace("\n", " ").replace("\r", " "),
            "# Modulation is unknown. LTE UL MIMO uses one layer per configured carrier."]


def render_nr_trace(combos, device):
    """Preserve ordered NR DL/UL per-CC bandwidth, SCS, MIMO and classes."""
    lines = _header(device)
    features, sets = {}, {}
    bodies = []
    seen = set()
    families = Counter()

    def feature(rat, direction, scs, bw, mimo):
        if rat == "N" and scs not in (15, 30, 60, 120):
            # ImportMtkNr silently maps any other SCS string to 15 kHz.
            raise TraceError(f"MTK NR importer cannot represent SCS {scs} kHz")
        namespace = (rat, direction)
        table = features.setdefault(namespace, {})
        key = (scs, bw, mimo)
        if key not in table:
            idx = len(table) + 1
            table[key] = idx
            if rat == "N":
                if bw is not None and (not isinstance(bw, int) or bw <= 0):
                    raise TraceError(f"invalid native bandwidth: {bw}")
                field = "mimo" if direction == "DL" else "cb_mimo"
                lines.append(f"[CAP] NR {direction} FSpCC[{idx}], scs[NL1_CAP_SCS_{scs}KHZ], "
                             f"bw[NL1_CAP_BW{bw or 0}], bw90m[NL1_CAP_NOT_SUPPORT], "
                             f"{field}[NL1_CAP_MIMO_{_layer(mimo)}], modulation[UNKNOWN]")
            else:
                lines.append(f"[CAP] EUTRA {direction} FSpCC[{idx}], mimo[NL1_CAP_MIMO_{_layer(mimo)}]")
        return table[key]

    def feature_set(rat, direction, ids):
        if not ids:
            return "_0"
        limit = (8 if direction == "DL" else 4) if rat == "N" else 5
        if len(ids) > limit:
            raise TraceError(f"MTK importer supports at most {limit} {rat} {direction} carriers per FS")
        table = sets.setdefault((rat, direction), {})
        key = tuple(ids)
        if key not in table:
            idx = len(table) + 1
            table[key] = idx
            name = "NR" if rat == "N" else "EUTRA"
            lines.append(f"[CAP] {name} {direction} FS[{idx}], FS{direction}pCC ID" +
                         "".join(f"[{i}]" for i in ids))
        return f"{rat}{table[key]}"

    for combo in combos:
        if not combo.nr:
            continue
        dl, ul, pairs = [], [], []
        for rat, components in (("E", combo.lte), ("N", combo.nr)):
            for component in components:
                prefix = "B" if rat == "E" else "N"
                dl.append(f"{prefix}{component.band}{_class(component.dl_class)}")
                ul.append(f"{prefix}{component.band}{_class(component.ul_class)}" if component.has_ul else "0")
                if rat == "E":
                    dl_ids = [feature(rat, "DL", None, None, m) for m in component.dl_mimo]
                    ul_ids = ([feature(rat, "UL", None, None, 1)] * LTE_WEIGHTS[component.ul_class]
                              if component.has_ul else [])
                else:
                    dl_ids = [feature(rat, "DL", cc.scs_khz, cc.dl_bw_mhz, cc.dl_mimo)
                              for cc in component.ccs]
                    ul_ids = [feature(rat, "UL", cc.scs_khz, cc.ul_bw_mhz, cc.ul_mimo)
                              for cc in component.ccs if component.has_ul and cc.ul_mimo is not None]
                if not dl_ids:
                    raise TraceError("missing DL per-carrier features")
                pairs.append(f"[{feature_set(rat, 'DL', dl_ids)}/{feature_set(rat, 'UL', ul_ids)}]")
        key = (tuple(dl), tuple(ul), tuple(pairs))
        if key in seen:
            continue
        seen.add(key)
        idx = len(seen)
        bodies.extend((f"[CAP] FSC[{idx}], D/U{''.join(pairs)}",
                       f"[CAP] CA idx [{idx - 1}] NL1 bc, num[{len(dl)}] DL: {'_'.join(dl)} "
                       f"UL: {'_'.join(ul)} FSC[{idx}]"))
        family = ("endc" if combo.lte else "nrdc" if
                  any(c.band < 257 for c in combo.nr) and any(c.band >= 257 for c in combo.nr) else "nrca")
        families[family] += 1
    return "\n".join(lines + bodies) + "\n", {"records": len(seen), "families": dict(families),
        "modulation": "unknown", "unknown_bandwidth": "NL1_CAP_BW0"}


def render_lte_log(combos, device):
    """CA_COMB_INFO importer supports one 2/4-layer value per component."""
    records = []
    seen = set()
    mixed = 0
    for combo in combos:
        if combo.nr or not combo.lte:
            continue
        row = []
        for c in combo.lte:
            if not (0 <= c.dl_class < 6 and 0 <= c.ul_class <= 6):
                raise TraceError("LTE importer requires MTK A..F classes / UL-absent=6")
            if not c.dl_mimo or any(m not in (2, 4) for m in c.dl_mimo):
                raise TraceError("LTE CA_COMB_INFO importer supports only known 2/4-layer MIMO")
            if len(set(c.dl_mimo)) > 1:
                mixed += 1
            # Do not claim every carrier supports the strongest carrier's MIMO.
            row.append((c.band, c.ul_class, c.dl_class, min(c.dl_mimo)))
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            records.append(row)
    lines = _header(device) + [
        "# BCS unknown: zero placeholders are required by ImportMTKLte; no BCS support is asserted.",
        "# Mixed per-carrier DL MIMO is projected to the minimum per logical component.",
        "MSG_ID_ERRC_RCM_UE_PRE_CA_COMB_INFO",
        f"bandwidth_comb_set = Array[{len(records)}]"]
    lines.extend(f"bandwidth_comb_set[{i}] = 0x0" for i in range(len(records)))
    for idx, row in enumerate(records):
        lines.extend((f"band_comb[{idx}]", f"band_param_num = {len(row)}", f"band_param = Array[{len(row)}]"))
        for i, (band, ul, dl, mimo) in enumerate(row):
            lines.extend((f"band_param[{i}]", f"band = {band}", f"class_ul = {ul}", f"class_dl = {dl}"))
        lines.append(f"band_mimo = Array[{len(row)}]")
        for i, (_, _, _, mimo) in enumerate(row):
            lines.extend((f"band_mimo[{i}]", f"mimo = ERRC_CAPA_CA_MIMO_CAPA_{_layer(mimo)}"))
    return "\n".join(lines) + "\n", {"records": len(records), "bcs": "unknown; zero placeholder",
        "mimo_projection": "minimum per logical component", "mixed_mimo_components_projected": mixed}


def write_trace(combos, path, device, *, nr):
    content, meta = (render_nr_trace if nr else render_lte_log)(combos, device)
    path.write_text(content, encoding="utf-8")
    return {**meta, "file": str(path), "derived": True,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
