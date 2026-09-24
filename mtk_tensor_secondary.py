"""Structural decoder for Tensor split-CDF secondary capability banks.

Pixel/Tensor images keep the ordinary FR1 CandidateNode grammar in bank 6,
but the FR2/NR-DC catalogue in bank 8 is rooted by small pointer tables in
md1rom and uses a different NR class-weight table for mmWave bands.  This
module deliberately discovers those roots from the image rather than relying
on a Pixel build address.  A result is returned only after every candidate,
descriptor, FSC cursor, and feature reference closes inside its bank image.

The return value contains plain dictionaries so the GUI core can turn it into
the shared ``mtk_export`` objects without creating an import cycle.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

try:  # Optional; the universal parser already uses numpy when available.
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


SCS_KHZ = {0: 15, 1: 30, 2: 60, 3: 120, 4: 240}
DL_MIMO = {0: 2, 1: 4, 2: 8}
UL_MIMO = {0: 1, 1: 2, 2: 4}
LTE_WEIGHT = (1, 2, 2, 3, 4, 5)
NR_OMIT_UL = frozenset((0x1C, 0xFF))

# Tensor FR2 uses a native G..M namespace whose multiplicity is not the
# generic FR1 table.  These are raw firmware class bytes, not B826 letters.
FR2_WEIGHT = {0: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8}


@dataclass
class SecondaryProfile:
    bank_index: int
    bank_va: int
    profile: int
    combos: list[dict]
    candidate_count: int
    expanded_rows: int
    excluded_single_fr2: int
    roots: dict


def _u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def _runtime_ptr(value, alias):
    return value - alias if alias and value >= alias else value


def _read_ptr_array(im, ptr, *, alias, max_count):
    """Read a zero-terminated pointer array wholly contained by ``im``."""
    raw = _runtime_ptr(ptr, alias)
    off = raw - im.relocation
    if off < im.source_offset or off + 4 > im.end_source:
        return None
    out = []
    for i in range(max_count):
        pos = off + i * 4
        if pos + 4 > im.end_source:
            return None
        value = _u32(im.drdi, pos)
        if value == 0:
            return out
        out.append(value)
    return None


def _weight(band, cls, generic):
    if band >= 257:
        return FR2_WEIGHT.get(cls, 0)
    return generic[cls] if 0 <= cls < len(generic) else 0


class _Decoder:
    def __init__(self, loader, bank, reporter=None):
        self.loader = loader
        self.bank = bank
        self.reporter = reporter
        self.alias = int(getattr(loader, "ALIAS", 0x60000000))
        self.images = {im.profile: im for im in bank.images}
        self.generic_nr_weight = tuple(loader.tables.nr_weights)
        self.lte_band_map = tuple(loader.tables.lte_band_map)
        self.bw = tuple(loader.tables.bw)
        self._desc_cache = {}

    def _candidate_values(self):
        """ROM words that can point into this bank (small, bounded search set)."""
        lo = self.bank.bank_va
        hi = lo + max(im.length for im in self.images.values())
        alo, ahi = lo + self.alias, hi + self.alias
        if _np is not None:
            words = _np.frombuffer(self.loader.rom,
                                   dtype="<u4", count=len(self.loader.rom) // 4)
            # Tensor images normally store all DRDI references with the
            # 0x60000000 alias.  Prefer that compact set; fall back to raw
            # runtime addresses for firmware that omits the alias entirely.
            aliased = words[(words >= alo) & (words < ahi)]
            if len(aliased):
                return sorted(set(int(x) for x in aliased))
            return sorted(set(int(x) for x in words[(words >= lo) & (words < hi)]))
        vals = set()
        raw_vals = set()
        for off in range(0, len(self.loader.rom) - 3, 4):
            value = _u32(self.loader.rom, off)
            if alo <= value < ahi:
                vals.add(value)
            elif lo <= value < hi:
                raw_vals.add(value)
        return sorted(vals or raw_vals)

    def _decode_desc(self, im, ptr, nr):
        key = (im.profile, ptr, nr)
        if key in self._desc_cache:
            return self._desc_cache[key]
        off = im.resolve(ptr, 16)
        if off is None:
            return None
        count, records_ptr, fsc_count, fsc_ptr = struct.unpack_from("<4I", im.drdi, off)
        if not (1 <= count <= 16 and fsc_count > 0):
            return None
        rec_size = 4 if nr else 3
        records_off = im.resolve(records_ptr, count * rec_size)
        if records_off is None:
            return None
        records = []
        units = 0
        for i in range(count):
            if nr:
                band, ul, dl = struct.unpack_from("<HBB", im.drdi, records_off + i * 4)
                w = _weight(band, dl, self.generic_nr_weight)
                if not (1 <= band <= 1024 and w > 0):
                    return None
                if ul not in NR_OMIT_UL and _weight(band, ul, self.generic_nr_weight) <= 0:
                    return None
            else:
                idx, ul, dl = struct.unpack_from("<BBB", im.drdi, records_off + i * 3)
                if not (0 <= idx < len(self.lte_band_map)):
                    return None
                band = self.lte_band_map[idx]
                if not (1 <= band <= 90 and 0 <= dl < len(LTE_WEIGHT)):
                    return None
                if ul != 6 and not (0 <= ul < len(LTE_WEIGHT)):
                    return None
                w = LTE_WEIGHT[dl]
            records.append((band, ul, dl))
            units += w
        fsc_size = 3 if nr else 2
        fsc_off = im.resolve(fsc_ptr, fsc_count * fsc_size) if fsc_ptr else None
        if units <= 0 or fsc_off is None or fsc_count % units:
            return None
        result = {"count": count, "records": records, "fsc_count": fsc_count,
                  "fsc_off": fsc_off, "units": units,
                  "variants": fsc_count // units, "nr": nr}
        self._desc_cache[key] = result
        return result

    def _candidate_shape(self, im, ptr):
        off = im.resolve(ptr, 16)
        if off is None:
            return None
        meta0, meta1, lte_ptr, nr_ptr = struct.unpack_from("<4I", im.drdi, off)
        if nr_ptr == 0:
            return None
        nr = self._decode_desc(im, nr_ptr, True)
        if nr is None:
            return None
        lte = self._decode_desc(im, lte_ptr, False) if lte_ptr else None
        if lte_ptr and lte is None:
            return None
        return {"ptr": ptr, "meta0": meta0, "meta1": meta1, "lte": lte, "nr": nr}

    def _candidate_roots(self, values):
        found = {}
        for profile, im in self.images.items():
            best = None
            for root in values:
                pointers = _read_ptr_array(im, root, alias=self.alias, max_count=100000)
                # A secondary capability root is a large array.  This excludes
                # feature arrays and incidental pointer lists early.
                if not pointers or len(pointers) < 32:
                    continue
                sample = pointers[: min(128, len(pointers))]
                if any(self._candidate_shape(im, p) is None for p in sample):
                    continue
                decoded = []
                for p in pointers:
                    shape = self._candidate_shape(im, p)
                    if shape is None:
                        decoded = []
                        break
                    decoded.append(shape)
                if decoded and (best is None or len(decoded) > len(best[1])):
                    best = (root, decoded)
            if best is not None:
                found[profile] = best
        return found

    def _feature_roots(self, values):
        found = {}
        for profile, im in self.images.items():
            choices = []
            for root in values:
                pointers = _read_ptr_array(im, root, alias=self.alias, max_count=128)
                if not pointers or not (8 <= len(pointers) <= 64):
                    continue
                rows = []
                for p in pointers:
                    off = im.resolve(p, 3)
                    if off is None:
                        rows = []
                        break
                    status, bw, bw90 = im.drdi[off:off + 3]
                    # Unsupported feature objects conventionally carry a
                    # sentinel BW index, so only supported rows use the ROM
                    # dictionary domain check.
                    if status not in (0, 1, 2, 3) or (status != 3 and bw >= len(self.bw)):
                        rows = []
                        break
                    rows.append((status, bw, int(bool(bw90))))
                if rows:
                    choices.append((root, rows))
            if choices:
                found[profile] = choices
        return found

    def _decode_profile(self, im, candidates, dl_features, ul_features):
        combos = []
        single_fr2 = 0
        for candidate in candidates:
            nr = candidate["nr"]
            lte = candidate["lte"]
            lte_variants = lte["variants"] if lte else 1
            nr_variants = nr["variants"]
            if (lte and lte_variants not in (1, nr_variants)
                    and nr_variants != 1):
                return None
            variants = max(lte_variants, nr_variants)
            lte_raw = (bytes(im.drdi[lte["fsc_off"]:lte["fsc_off"] + lte["fsc_count"] * 2])
                       if lte else b"")
            nr_raw = bytes(im.drdi[nr["fsc_off"]:nr["fsc_off"] + nr["fsc_count"] * 3])
            bands = [x[0] for x in nr["records"]]
            has_fr1 = any(x < 257 for x in bands)
            has_fr2 = any(x >= 257 for x in bands)
            if not lte and not (has_fr1 and has_fr2):
                single_fr2 += variants
                continue
            for variant in range(variants):
                out_lte = []
                if lte:
                    lv = 0 if lte_variants == 1 else variant
                    cursor = lv * lte["units"]
                    for band, ul, dl in lte["records"]:
                        mimo = []
                        for _ in range(LTE_WEIGHT[dl]):
                            _aux, status = lte_raw[cursor * 2:cursor * 2 + 2]
                            if status not in (0, 1, 2):
                                return None
                            mimo.append({0: 2, 1: 4, 2: 8}[status]); cursor += 1
                        out_lte.append({"band": int(band), "dl_class": int(dl),
                                        "ul_class": int(ul), "dl_mimo": mimo})
                    if cursor != (lv + 1) * lte["units"]:
                        return None
                out_nr = []
                nv = 0 if nr_variants == 1 else variant
                cursor = nv * nr["units"]
                for band, ul, dl in nr["records"]:
                    cc_count = _weight(band, dl, self.generic_nr_weight)
                    active_ul_expected = (0 if ul in NR_OMIT_UL
                                          else _weight(band, ul, self.generic_nr_weight))
                    ccs = []; active = 0
                    for _ in range(cc_count):
                        scs, ui, di = nr_raw[cursor * 3:cursor * 3 + 3]; cursor += 1
                        if scs not in SCS_KHZ or di >= len(dl_features) or ui >= len(ul_features):
                            return None
                        dstatus, dbw, dbw90 = dl_features[di]
                        ustatus, ubw, ubw90 = ul_features[ui]
                        if dstatus not in DL_MIMO or dbw >= len(self.bw):
                            return None
                        if ustatus not in (0, 1, 2, 3):
                            return None
                        if ustatus != 3:
                            active += 1
                        ccs.append({"scs_khz": SCS_KHZ[scs],
                                    "dl_mimo": DL_MIMO[dstatus],
                                    "dl_bw_mhz": int(self.bw[dbw]),
                                    "ul_mimo": UL_MIMO.get(ustatus) if ustatus != 3 else None,
                                    "ul_bw_mhz": int(self.bw[ubw]) if ustatus != 3 and ubw < len(self.bw) else None})
                    if active != active_ul_expected:
                        return None
                    out_nr.append({"band": int(band), "dl_class": int(dl),
                                   "ul_class": (0x1C if ul in NR_OMIT_UL else int(ul)),
                                   "ccs": ccs})
                if cursor != (nv + 1) * nr["units"]:
                    return None
                combos.append({"lte": out_lte, "nr": out_nr})
        return combos, single_fr2

    def decode(self):
        values = self._candidate_values()
        roots = self._candidate_roots(values)
        feature_roots = self._feature_roots(values)
        results = []
        for profile, im in self.images.items():
            if profile not in roots or profile not in feature_roots:
                continue
            candidates = roots[profile][1]
            choices = feature_roots[profile]
            best = None
            # There are normally two choices (DL and UL); try both directions
            # and retain only the one for which all FSC references close.
            for dl_root, dl in choices:
                for ul_root, ul in choices:
                    if dl_root == ul_root:
                        continue
                    decoded = self._decode_profile(im, candidates, dl, ul)
                    if decoded is not None:
                        combos, excluded = decoded
                        if best is None or len(combos) > len(best[0]):
                            best = (combos, excluded, dl_root, ul_root)
            if best is None:
                continue
            combos, excluded, dl_root, ul_root = best
            results.append(SecondaryProfile(
                bank_index=self.bank.table_index, bank_va=self.bank.bank_va,
                profile=profile, combos=combos, candidate_count=len(candidates),
                expanded_rows=len(combos) + excluded, excluded_single_fr2=excluded,
                roots={"candidate": roots[profile][0], "dl_features": dl_root,
                       "ul_features": ul_root}))
            if self.reporter is not None:
                self.reporter.info("tensor_secondary_bank",
                                   "decoded Tensor secondary FR2/NRDC bank",
                                   bank_index=self.bank.table_index,
                                   profile=profile,
                                   candidate_count=len(candidates),
                                   expanded_rows=len(combos) + excluded,
                                   exported_rows=len(combos),
                                   excluded_single_fr2=excluded,
                                   roots={k: hex(v) for k, v in results[-1].roots.items()})
        return results


def decode_tensor_secondary(loader, bank_index=8, reporter=None):
    """Return validated secondary-bank profiles, or an empty list.

    Secondary banks are optional across Tensor releases.  A failed proof is a
    warning and leaves the ordinary capability extraction untouched; callers
    can still present the regular Bank-5/6 results.
    """
    bank = next((b for b in loader.banks if b.table_index == bank_index and b.images), None)
    if bank is None:
        return []
    try:
        results = _Decoder(loader, bank, reporter).decode()
    except (IndexError, struct.error, ValueError) as exc:
        if reporter is not None:
            reporter.warn("tensor_secondary_unresolved",
                          "secondary Tensor bank did not pass structural proof",
                          bank_index=bank_index, reason=str(exc))
        return []
    if not results and reporter is not None:
        reporter.warn("tensor_secondary_unresolved",
                      "secondary Tensor bank roots or FSC tables were not proved",
                      bank_index=bank_index)
    return results
