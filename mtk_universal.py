"""Inputs:
    md1rom + md1drdi, or md1rom + Tensor-style md1drdi_hdr + md1drdi_data.
    --image accepts a packaged modem image or extracted-parts directory;
    mtk_containers unwraps recognized layers before container/grammar discovery."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import struct
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

HERE = Path(__file__).resolve().parent

def _import_shared():
    """Locate the project's shared exporter.

    The exporter is deliberately NOT vendored: mtk_export.py and its b826
    encoder are the validated rendering path and must stay single-sourced.
    Search order: MTK_SHARED env var, ./shared, ../parsers/shared, then the
    repository default.  Fail loudly rather than silently parsing with no
    ability to export.
    """
    cands = [HERE]
    env = os.environ.get("MTK_SHARED")
    if env:
        cands.append(Path(env))
    cands += [HERE / "shared", HERE.parent / "parsers" / "shared",
              Path(r"D:/MTK/parsers/shared")]
    tried = []
    for c in cands:
        tried.append(str(c))
        if (c / "mtk_export.py").is_file():
            sys.path.insert(0, str(c))
            import mtk_export as _m
            return _m, str(c)
    raise SystemExit("cannot locate mtk_export.py (shared exporter). Tried:"
                     + "".join("\n  " + t for t in tried)
                     + "\nSet MTK_SHARED to the directory that holds it.")

export, SHARED = _import_shared()

from mtk_export import write_b0cd_v41
from mtk_tensor_secondary import decode_tensor_secondary
from mtk_trace import write_trace

VERSION = "0.2-universal"

U32 = struct.Struct("<I")
D16 = struct.Struct("<IIII")
NRREC = struct.Struct("<HBB")
# Two bandwidth-enum families are proven in the corpus.  They are disjoint at
# index 10 (100 vs 90), so a hit is never ambiguous.  A third family must be
# added here explicitly -- the tool refuses to extrapolate an enum it has not
# seen, because a wrong bandwidth dictionary produces plausible-looking but
# wrong output, which is the exact failure mode this project treats as worst.
BW_FAMILIES = {
    "modern20": (5, 10, 15, 20, 25, 30, 40, 50, 60, 80,
                 100, 200, 400, 35, 45, 70, 90, 800, 1600, 2000),
    "legacy14": (5, 10, 15, 20, 25, 30, 40, 50, 60, 80,
                 90, 100, 200, 400),
}
BW20 = BW_FAMILIES["modern20"]

LTE_WEIGHT_PREFIX = bytes((1, 2, 2, 3, 4, 5))
NR_WEIGHT_PREFIX = bytes((1, 2, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8))
LTE_WEIGHTS_EXPECTED = (1, 2, 2, 3, 4, 5)
BANDMAP_PREFIX = bytes(list(range(49)) + list(range(65, 72)))
BANDMAP_LEN = 96
MAX_CLASS_WEIGHT = 32        # no observed class aggregates more than 32 carriers
MAX_LTE_BAND = 90            # highest real entry in every observed band map
# Entry 0 of a FeatureObj table is an "unsupported" object: mimo_status == 3.
# This literal is only the modern spelling of it (bandwidth code 0x14 is one
# past a 20-entry enum); MD800 writes 03 0d 01 instead.  Used as a cheap
# prefilter for modern bank selection only -- never as the universal anchor.
FEATURE_SENTINEL = b"\x03\x14\x01"
SCS = {0: 15, 1: 30, 2: 60, 3: 120, 4: 240}
DL_MIMO = {0: 2, 1: 4, 2: 8}
UL_MIMO = {0: 1, 1: 2, 2: 4}
LTE_UL_ABSENT = 6
NR_UL_ABSENT_CANON = 0x1C    # value the shared exporter expects; see RomTables.nr_ul_absent
VA_LO = 0x60000000           # DRDI runtime address window (all solved devices)
VA_HI = 0x80000000


class UniversalError(RuntimeError):
    pass


class CheckError(UniversalError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def find_all(buf: bytes, pat: bytes, limit: Optional[int] = None) -> list[int]:
    out, start = [], 0
    while True:
        p = buf.find(pat, start)
        if p < 0:
            break
        out.append(p)
        if limit is not None and len(out) >= limit:
            break
        start = p + 1
    return out


def u16(buf: bytes, o: int) -> int:
    return struct.unpack_from("<H", buf, o)[0]


def u32(buf: bytes, o: int) -> int:
    return struct.unpack_from("<I", buf, o)[0]


@dataclass
class Issue:
    level: str
    code: str
    message: str
    context: dict = field(default_factory=dict)


@dataclass
class Reporter:
    issues: list[Issue] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)

    def info(self, code: str, msg: str, **ctx):
        self.issues.append(Issue("info", code, msg, ctx))

    def warn(self, code: str, msg: str, **ctx):
        self.issues.append(Issue("warning", code, msg, ctx))

    def fail(self, code: str, msg: str, **ctx):
        self.issues.append(Issue("error", code, msg, ctx))

    def check(self, name: str, passed: bool, **data):
        self.checks.append({"name": name, "passed": bool(passed), **data})
        if not passed:
            raise CheckError(f"check failed: {name}: {data}")

    def as_dict(self):
        return {
            "issues": [dataclasses.asdict(x) for x in self.issues],
            "checks": self.checks,
        }


@dataclass(frozen=True)
class RomTables:
    bw: tuple[int, ...]
    nr_weights: tuple[int, ...]
    lte_weights: tuple[int, ...]
    lte_band_map: tuple[int, ...]
    bw_off: int
    nr_weights_off: int
    lte_weights_off: int
    band_map_off: int
    bw_family: str = "modern20"
    band_map_off_alternatives: tuple[int, ...] = ()

    @property
    def nr_ul_absent(self) -> frozenset:
        """UL-class byte values meaning "this component carries no uplink".

        Firmware-dependent, and never to be hardcoded: 0x1c on modern
        Dimensity, additionally 0xff on Tensor, 0x11 on the MD800 legacy
        family.  The rule that generalises is structural rather than numeric --
        a UL byte with no weight in the discovered NR class table cannot denote
        carriers, so it must be an absent marker.  The claim is then *proved*
        per profile by the UL feature-closure check in FeatureResolver, which
        requires exactly zero active UL feature objects for every component
        that uses one.
        """
        w = self.nr_weights
        return frozenset(v for v in range(256) if v >= len(w) or w[v] == 0)

    def as_dict(self):
        d = dataclasses.asdict(self)
        d["nr_class_table_len"] = len(self.nr_weights)
        d["nr_ul_absent_low_values"] = sorted(v for v in self.nr_ul_absent if v < 64)[:12]
        for k in ("bw_off", "nr_weights_off", "lte_weights_off", "band_map_off"):
            d[k] = hex(d[k])
        d["band_map_off_alternatives"] = [hex(x) for x in self.band_map_off_alternatives]
        return d


def _bw_sites(rom: bytes) -> list:
    """Locate every occurrence of a known bandwidth enum family."""
    out = []
    for fam, tbl in BW_FAMILIES.items():
        pat = struct.pack("<%dH" % len(tbl), *tbl)
        for o in find_all(rom, pat):
            # A short family must be terminated, otherwise we would clip a
            # longer table that merely shares a prefix.
            if len(tbl) < 20:
                if o + len(pat) + 2 > len(rom) or u16(rom, o + len(pat)) != 0:
                    continue
            out.append((fam, o, tbl))
    return out


def _read_nr_weights(rom: bytes, off: int, cap: int = 96) -> tuple:
    """Read the NR class-weight table without assuming a terminator byte.

    Modern firmware ends the table with 0xff.  The MD800 legacy family does
    not -- it simply runs into unrelated data.  The portable rule is that a
    class weight is a small carrier count, so the table ends at the first byte
    that cannot be one.  Interior zeros are kept: they are unused class slots,
    and dropping them would shift every later class index.
    """
    vals = []
    for i in range(cap):
        if off + i >= len(rom):
            break
        b = rom[off + i]
        if b == 0xFF or b > MAX_CLASS_WEIGHT:
            break
        vals.append(b)
    while len(vals) > len(NR_WEIGHT_PREFIX) and vals[-1] == 0:
        # Trailing zeros carry no information; keeping the table minimal makes
        # "no weight" and "index out of range" the same statement.
        vals.pop()
    return tuple(vals)


def _lte_weight_sites(rom: bytes) -> list:
    return [o for o in find_all(rom, LTE_WEIGHT_PREFIX)
            if o + 8 <= len(rom) and rom[o + 6] == 0 and rom[o + 7] in (0, 0xFF)]


def _band_map_sites(rom: bytes) -> list:
    out = []
    for o in find_all(rom, BANDMAP_PREFIX):
        if o + BANDMAP_LEN > len(rom):
            continue
        # Byte 95 terminates the map.  Both 0x00 and 0xff are observed.
        if rom[o + BANDMAP_LEN - 1] not in (0, 0xFF):
            continue
        out.append(o)
    return out


def discover_rom_tables(rom: bytes, rep: Reporter) -> RomTables:
    """Discover and validate the ROM dictionaries without device offsets."""
    bws = _bw_sites(rom)
    if not bws:
        raise UniversalError(
            "no known MTK bandwidth enum found in md1rom. Known families: "
            + ", ".join("%s(%d entries)" % (k, len(v)) for k, v in BW_FAMILIES.items())
            + ". A new family must be added explicitly rather than inferred.")
    fams = set(f for f, _, _ in bws)
    if len(fams) > 1:
        raise UniversalError("md1rom matches more than one bandwidth enum family: %s" % sorted(fams))

    maps = _band_map_sites(rom)
    if not maps:
        raise UniversalError("no validated 96-byte LTE internal-band map found in md1rom")
    ltes = _lte_weight_sites(rom)
    nrs = set(find_all(rom, NR_WEIGHT_PREFIX))
    if not nrs:
        raise UniversalError("NR class-weight prefix 1,2,2,3,4,2,3,4,5,6,7,8 not found in md1rom")

    # Pick the bandwidth site whose surroundings resolve into a complete,
    # self-consistent dictionary.  Distance is only a tiebreak, never a gate:
    # the observed enum-to-map distance ranges from 0x378 to 0xe93c.
    errors = []
    for fam, bw_off, tbl in bws:
        bw_end = bw_off + 2 * len(tbl)
        # NR weights sit immediately after the enum on Tensor/Oppo/17T-Pro, or
        # eight bytes past a detached LTE table on 17T/Poco/Samsung/MD800.
        nr_off = None
        adj = [l for l in ltes if (l + 8) in nrs]
        if bw_end in nrs:
            nr_off = bw_end
        elif len(adj) == 1:
            nr_off = adj[0] + 8
        elif len(nrs) == 1:
            nr_off = next(iter(nrs))
        elif adj:
            nr_off = min(adj, key=lambda l: abs(l - bw_off)) + 8
        if nr_off is None:
            errors.append("%s@%#x: could not disambiguate %d NR class-weight candidates"
                          % (fam, bw_off, len(nrs)))
            continue
        if (nr_off - 8) in ltes:
            lte_off = nr_off - 8
        elif len(ltes) == 1:
            lte_off = ltes[0]
        elif ltes:
            lte_off = min(ltes, key=lambda o: abs(o - bw_off))
        else:
            errors.append("%s@%#x: no validated LTE class-weight table" % (fam, bw_off))
            continue

        nrw = _read_nr_weights(rom, nr_off)
        ltew = tuple(rom[lte_off:lte_off + 6])
        bw = tuple(tbl)
        if len(nrw) < len(NR_WEIGHT_PREFIX) or nrw[:len(NR_WEIGHT_PREFIX)] != tuple(NR_WEIGHT_PREFIX):
            errors.append("%s@%#x: NR weights failed prefix validation: %s" % (fam, bw_off, nrw[:16]))
            continue
        if ltew != LTE_WEIGHTS_EXPECTED:
            errors.append("%s@%#x: LTE class weights invalid: %s" % (fam, bw_off, ltew))
            continue

        # Band map: prefer the copy nearest the enum.  The copies are NOT
        # identical -- on Xiaomi 17T the far copy stores 252..255 where the
        # near copy stores 0 at indices 56..59, and 252..255 are reserved
        # markers rather than bands.  The near copy keeps those slots
        # rejectable; MAX_LTE_BAND rejects them either way.
        ordered = sorted(maps, key=lambda o: abs(o - bw_off))
        map_off = ordered[0]
        bandmap = tuple(rom[map_off:map_off + BANDMAP_LEN])

        rep.info("rom_tables", "discovered and validated ROM dictionaries",
                 bw_family=fam, bw_entries=len(bw), bw_off=hex(bw_off),
                 nr_weights_off=hex(nr_off), lte_weights_off=hex(lte_off),
                 band_map_off=hex(map_off), nr_class_table_len=len(nrw),
                 band_map_alternatives=[hex(o) for o in ordered[1:]],
                 enum_to_map_distance=hex(abs(map_off - bw_off)))
        return RomTables(bw, nrw, ltew, bandmap, bw_off, nr_off, lte_off, map_off,
                         bw_family=fam, band_map_off_alternatives=tuple(ordered[1:]))
    raise UniversalError("no bandwidth-enum site produced a complete dictionary: " + "; ".join(errors))


@dataclass
class Image:
    bank_va: int
    profile: int
    source_offset: int
    length: int
    relocation: int
    drdi: bytes
    label: str = ""
    alias: int = 0
    # Set by loaders that already know where a profile's CandidateNode pointer
    # array lives (the flat/legacy loader derives it while proving the image's
    # relocation).  It is a hint only: every pointer in it is still validated
    # and the run is still trimmed to its maximal fully-valid subrun.
    candidate_hint: Optional[tuple] = None

    @property
    def end_source(self) -> int:
        return self.source_offset + self.length

    @property
    def end_va(self) -> int:
        return self.bank_va + self.length

    def resolve(self, va: int, size: int = 1) -> Optional[int]:
        # Runtime pointers may carry a loader-defined alias (Tensor uses 0x60000000).
        raw = va - self.alias if self.alias and va >= self.alias else va
        off = raw - self.relocation
        if self.source_offset <= off and off + size <= self.end_source:
            return off
        return None

    def contains_va(self, va: int, size: int = 1) -> bool:
        return self.resolve(va, size) is not None

    def read(self, va: int, size: int) -> Optional[bytes]:
        o = self.resolve(va, size)
        return None if o is None else self.drdi[o:o + size]

    def u32(self, va: int) -> Optional[int]:
        o = self.resolve(va, 4)
        return None if o is None else u32(self.drdi, o)

    def u16(self, va: int) -> Optional[int]:
        o = self.resolve(va, 2)
        return None if o is None else u16(self.drdi, o)

    def to_dict(self):
        return {
            "bank_va": hex(self.bank_va), "profile": self.profile,
            "source_offset": hex(self.source_offset), "length": self.length,
            "relocation": hex(self.relocation), "alias": hex(self.alias), "label": self.label,
        }


@dataclass
class Bank:
    bank_va: int
    images: list[Image]
    table_index: int = -1

    @property
    def live_profiles(self):
        return [x.profile for x in self.images]

    def to_dict(self):
        return {"table_index": self.table_index, "bank_va": hex(self.bank_va),
                "live_profiles": self.live_profiles,
                "images": [x.to_dict() for x in self.images]}


class BaseLoader:
    name = "base"
    def __init__(self, rom: bytes, drdi: bytes, rep: Reporter):
        self.rom, self.drdi, self.rep = rom, drdi, rep
        self.tables = discover_rom_tables(rom, rep)
        self.banks: list[Bank] = []
        self._candidate_arrays = {}

    def list_banks(self) -> list[Bank]:
        return self.banks

    def capability_bank(self) -> Bank:
        raise NotImplementedError

    def lte_tables(self, cap: Bank, rep: Reporter):
        """Return (bank, {profile: [Combo]}) for the LTE CA namespace.

        Container-specific, like everything else a loader owns: the grid and
        CDF families store contiguous 32-byte rows in a sibling bank, the flat
        family stores a pointer array of row objects.
        """
        return choose_lte_bank(self, cap, rep)


class GridLoader(BaseLoader):
    """Modern {source_field, bank_va, bank_len} 12-byte descriptor matrix."""
    name = "grid"

    def __init__(self, rom: bytes, drdi: bytes, rep: Reporter, *, descriptor_hits=None):
        super().__init__(rom, drdi, rep)
        self._discover(descriptor_hits)

    @staticmethod
    def descriptor_hits(rom: bytes, drdi: bytes) -> list:
        """Scan aligned descriptors once, without copying the ROM.

        Both implementations include a descriptor ending exactly at EOF and
        ignore an incomplete trailing word. Numpy only prefilters; the bounds
        and runtime-address requirements are the same as the scalar scanner.
        """
        if np is not None:
            words = np.frombuffer(rom, dtype="<u4", count=len(rom) // 4)
            if len(words) < 3:
                return []
            sf, va, ln = words[:-2], words[1:-1], words[2:]
            src = sf & 0x0FFFFFFF
            valid = ((sf >> 28 == 3) & (ln >= 0x10) & (ln <= 0x800000)
                     & (src + np.minimum(ln, 0x800001) <= len(drdi))
                     & (va >= VA_LO) & (va < VA_HI))
            return [(int(i) * 4, int(src[i]), int(va[i]), int(ln[i]))
                    for i in np.flatnonzero(valid)]
        hits = []
        for off in range(0, len(rom) - 11, 4):
            sf, va, ln = struct.unpack_from("<III", rom, off)
            src = sf & 0x0FFFFFFF
            if ((sf >> 28) == 3 and 0x10 <= ln <= 0x800000
                    and src + ln <= len(drdi) and VA_LO <= va < VA_HI):
                hits.append((off, src, va, ln))
        return hits

    @staticmethod
    def _dense_score(hits) -> int:
        last = -100
        run = best = 0
        for off, *_ in hits:
            run = run + 1 if off - last <= 0x40 else 1
            best = max(best, run)
            last = off
        return best

    @staticmethod
    def probe(rom: bytes, drdi: bytes) -> int:
        return GridLoader._dense_score(GridLoader.descriptor_hits(rom, drdi))

    def _discover(self, descriptor_hits=None):
        raw = (self.descriptor_hits(self.rom, self.drdi)
               if descriptor_hits is None else descriptor_hits)
        if not raw:
            raise UniversalError("no modern bank descriptors found")
        # Dense cluster: real table is 12-byte-stride but small gaps can exist in false scans.
        clusters, cur = [], [raw[0]]
        for a, b in zip(raw, raw[1:]):
            if 0 < b[0] - a[0] <= 0x40:
                cur.append(b)
            else:
                if len(cur) >= 4:
                    clusters.append(cur)
                cur = [b]
        if len(cur) >= 4:
            clusters.append(cur)
        if not clusters:
            raise UniversalError("raw bank-descriptor hits did not form a coherent table")
        table = max(clusters, key=len)
        # Require a uniform matrix: each distinct bank VA has same declared column count.
        by_va: dict[int, list] = defaultdict(list)
        for x in table:
            by_va[x[2]].append(x)
        counts = Counter(len(v) for v in by_va.values())
        col_count, freq = counts.most_common(1)[0]
        coherent = {va: xs for va, xs in by_va.items() if len(xs) == col_count}
        if len(coherent) < 1:
            raise UniversalError("descriptor table does not contain a coherent bank/profile matrix")

        self.descriptor_table_off = table[0][0]
        self.columns = col_count
        self.banks = []
        for bi, (va, xs) in enumerate(sorted(coherent.items(), key=lambda kv: min(x[0] for x in kv[1]))):
            xs = sorted(xs, key=lambda x: x[0])
            images = []
            seen_stub = False
            for pi, (_ro, src, _va, ln) in enumerate(xs):
                if ln <= 0x40:
                    seen_stub = True
                    continue
                if seen_stub:
                    # Live profiles are a contiguous prefix by observed MTK contract.
                    raise UniversalError(f"bank {va:#x} has live profile {pi} after a stub; descriptor geometry is likely wrong")
                images.append(Image(va, pi, src, ln, va - src, self.drdi,
                                    label=f"bank{bi}/profile{pi}"))
            self.banks.append(Bank(va, images, bi))
        self.rep.info("grid_loader", "discovered modern bank/profile descriptor matrix",
                      descriptor_table_off=hex(self.descriptor_table_off), columns=self.columns,
                      banks=len(self.banks), live_counts=[len(b.images) for b in self.banks])

    def capability_bank(self) -> Bank:
        if hasattr(self, "_capability_bank"):
            return self._capability_bank
        candidates = []
        for b in self.banks:
            if not b.images:
                continue
            sent = 0
            for im in b.images:
                sent += self.drdi.count(FEATURE_SENTINEL, im.source_offset, im.end_source)
            avg = sum(i.length for i in b.images) / len(b.images)
            candidates.append((avg, sent, b))
        parser = GrammarParser(self, self.rep)
        # Preserve the existing ranking as a search order, but require grammar
        # evidence before accepting it. A sentinel is not a container contract;
        # try banks without it too if none of the preferred banks validate.
        for avg, sent, b in sorted(candidates, key=lambda t: (bool(t[1]), t[0], t[1]), reverse=True):
            for im in b.images:
                try:
                    _, _, info = parser.find_candidate_array(im)
                except UniversalError:
                    continue
                self._capability_bank = b
                self.rep.info("capability_bank", "selected grid capability bank by CandidateNode validation",
                              bank_va=hex(b.bank_va), live_profiles=len(b.images), sentinel_hits=sent,
                              average_live_length=int(avg), proof_profile=im.profile,
                              proof_candidate_count=info["count"])
                return b
            self.rep.info("capability_bank_rejected", "bank has no validated CandidateNode array",
                          bank_va=hex(b.bank_va), sentinel_hits=sent)
        raise UniversalError("no live grid bank contains a structurally valid CandidateNode array")


class TensorCdfLoader(BaseLoader):
    """Tensor/Pixel split CDF: 0x30000 header + concatenated DRDI slot data.

    The loader exposes CDF slots as the same Image abstraction used by the
    modern descriptor grid.  Runtime VAs are aliased by 0x60000000; the shared
    grammar parser never needs to know this.
    """
    name = "tensor"
    ALIAS = 0x60000000

    def __init__(self, rom: bytes, header: bytes, data: bytes, rep: Reporter):
        self.header = header
        self._validate_header_shape()
        super().__init__(rom, data, rep)
        self.sections = [struct.unpack_from("<II", header, 4 + i*8) for i in range(20)]
        oo, osz = self.sections[0]
        self.slot_offsets = struct.unpack_from("<641I", header, oo)
        bo, bsz = self.sections[1]
        self.bank_bounds = struct.unpack_from("<11I", header, bo)
        self._validate_slots()
        self._build_banks()

    @staticmethod
    def probe(header: bytes) -> bool:
        if len(header) != 0x30000:
            return False
        try:
            sections=[struct.unpack_from("<II",header,4+i*8) for i in range(20)]
        except struct.error:
            return False
        return (sections[0][1] == 641*4 and sections[1][1] == 11*4
                and sections[2][1] == 640*48
                and all(164 <= off <= len(header) and size <= len(header) - off
                        for off, size in sections[:3]))

    def _validate_header_shape(self):
        if not self.probe(self.header):
            raise UniversalError("split-CDF header failed section geometry (expected 0x30000, 641 offsets, 11 bounds, 640 SHA-384 digests)")

    def _validate_slots(self):
        # Bounds and offsets must be monotone; every slot gets cryptographic
        # validation against the header before any capability bytes are trusted.
        if any(a >= b for a, b in zip(self.bank_bounds, self.bank_bounds[1:])):
            raise UniversalError("CDF bank bounds are not strictly increasing")
        if any(a>b for a,b in zip(self.slot_offsets,self.slot_offsets[1:])):
            raise UniversalError("CDF slot offsets are not monotone")
        if self.slot_offsets[-1] > len(self.drdi):
            raise UniversalError(f"CDF final slot offset {self.slot_offsets[-1]:#x} exceeds data size {len(self.drdi):#x}")
        if self.slot_offsets[-1] < len(self.drdi):
            self.rep.info("cdf_trailing_data","CDF data contains bytes after the 640 indexed slots",
                          indexed_end=hex(self.slot_offsets[-1]),data_size=hex(len(self.drdi)),
                          trailing_bytes=len(self.drdi)-self.slot_offsets[-1])
        digest_off,digest_size=self.sections[2]
        ok=bad=0
        for slot in range(640):
            st,en=self.slot_offsets[slot],self.slot_offsets[slot+1]
            exp=self.header[digest_off+slot*48:digest_off+(slot+1)*48]
            if hashlib.sha384(self.drdi[st:en]).digest()==exp: ok+=1
            else: bad+=1
        if bad:
            raise UniversalError(f"CDF SHA-384 validation failed for {bad}/640 slots")
        self.rep.info("cdf_integrity","validated all split-CDF slot SHA-384 digests",ok=ok,bad=bad)

    def _build_banks(self):
        banks=[]
        for bi in range(10):
            va=self.bank_bounds[bi]
            images=[]; seen_stub=False
            for pi in range(64):
                slot=bi*64+pi
                src=self.slot_offsets[slot]; ln=self.slot_offsets[slot+1]-src
                if ln <= 0x40:
                    seen_stub=True
                    continue
                if seen_stub:
                    raise UniversalError(f"CDF bank {bi} has live image after stub at profile {pi}")
                images.append(Image(va,pi,src,ln,va-src,self.drdi,label=f"cdf-bank{bi}/profile{pi}",alias=self.ALIAS))
            banks.append(Bank(va,images,bi))
        self.banks=banks
        self.rep.info("tensor_loader","discovered split-CDF banks/profiles",banks=len(banks),live_counts=[len(b.images) for b in banks],alias=hex(self.ALIAS))

    def capability_bank(self) -> Bank:
        # Sentinel is a cheap prefilter; CandidateNode structural validity is
        # the authority.  This avoids choosing a physically larger unrelated CDF
        # bank that happens to contain the same 3-byte sentinel.
        parser=GrammarParser(self,self.rep)
        candidates=[]
        for b in self.banks:
            sentinel_images=[]
            for im in b.images:
                chunk=self.drdi[im.source_offset:im.end_source]
                if FEATURE_SENTINEL in chunk:
                    sentinel_images.append(im)
            if not sentinel_images:
                continue
            best_count=0; proved_profile=None
            for im in sentinel_images:
                try:
                    _o,_rows,info=parser.find_candidate_array(im)
                    if info["count"]>best_count:
                        best_count=info["count"]; proved_profile=im.profile
                except UniversalError:
                    continue
            if best_count:
                avg=sum(i.length for i in b.images)/len(b.images)
                candidates.append((best_count,avg,b,proved_profile,len(sentinel_images)))
        if not candidates:
            raise UniversalError("no CDF bank containing feature sentinels also contains a 100%-valid CandidateNode array")
        best_count,avg,b,prof,ns=max(candidates,key=lambda x:(x[0],x[1]))
        self.rep.info("capability_bank","selected CDF capability bank by sentinel + CandidateNode proof",
                      bank_index=b.table_index,bank_va=hex(b.bank_va),live_profiles=len(b.images),
                      proof_profile=prof,proof_candidate_count=best_count,sentinel_profiles=ns)
        return b


class FlatLoader(BaseLoader):
    """MD800-class legacy container: no 12-byte bank descriptor matrix.

    The modern families publish a {source_field, bank_va, bank_len} grid in
    md1rom, so a loader can read the geometry.  This family publishes nothing
    comparable; profiles are simply regions of md1drdi, each mounted at its own
    relocation.  Rather than porting per-firmware roots, the loader recovers the
    geometry from two facts that hold structurally:

      1. A profile's CandidateNode pointer array is a maximal run of words in
         the DRDI runtime window, and the node objects it addresses are laid
         out immediately after it (4-byte bias).  That fixes a *hypothesis*

             relocation = array[0] - (array_off + 4*count + 4)

      2. The hypothesis is accepted only if the shared grammar then closes:
         the pointers parse as CandidateNodes and the class-weight invariant
         holds.  A wrong relocation produces essentially zero valid nodes, so
         this is a proof rather than a fit.

    On the RG620T-EG and VIVO X300 MAX images this recovers exactly the fourteen
    profile arrays and nothing else, and every recovered runtime address is then
    independently corroborated against the profile pointer table in md1rom.
    """
    name = "flat"
    MIN_ARRAY = 64
    SAMPLE = 48
    SAMPLE_RATE = 0.9

    def __init__(self, rom: bytes, drdi: bytes, rep: Reporter):
        super().__init__(rom, drdi, rep)
        self.runs = self._runtime_pointer_runs()
        self._discover()

    @staticmethod
    def _runs_from_words(words, minrun: int):
        if np is None:
            raise UniversalError("the flat loader needs numpy for whole-image pointer scanning")
        mask = (words >= VA_LO) & (words < VA_HI)
        idx = np.flatnonzero(mask)
        out = []
        if not len(idx):
            return out
        start = prev = int(idx[0])
        for z in idx[1:]:
            z = int(z)
            if z != prev + 1:
                if prev - start + 1 >= minrun:
                    out.append((start * 4, prev - start + 1))
                start = z
            prev = z
        if prev - start + 1 >= minrun:
            out.append((start * 4, prev - start + 1))
        return out

    def _runtime_pointer_runs(self):
        w = np.frombuffer(self.drdi[:len(self.drdi) // 4 * 4], dtype="<u4")
        self._words = w
        return self._runs_from_words(w, self.MIN_ARRAY)

    def _probe_image(self, reloc: int) -> Image:
        return Image(bank_va=reloc, profile=-1, source_offset=0, length=len(self.drdi),
                     relocation=reloc, drdi=self.drdi, label="flat-probe")

    def _hypotheses(self, off: int, n: int):
        """Relocation hypotheses for one pointer run, cheapest first."""
        v0 = int(self._words[off // 4])
        # Nodes immediately follow the array; the 4-byte bias is the observed
        # packing on every MD800 profile examined.  The no-bias variant is kept
        # as a second guess so a repack does not silently defeat discovery.
        for bias in (4, 0):
            r = v0 - (off + 4 * n + bias)
            if 0 < r < VA_HI and r + len(self.drdi) < (1 << 32):
                yield r, bias

    def _discover(self):
        parser = GrammarParser(self, self.rep)
        found = []
        for off, n in self.runs:
            for reloc, bias in self._hypotheses(off, n):
                im = self._probe_image(reloc)
                k = min(n, self.SAMPLE)
                ok = 0
                for i in range(k):
                    if parser.parse_candidate_va(im, int(self._words[off // 4 + i])) is not None:
                        ok += 1
                if ok >= k * self.SAMPLE_RATE:
                    found.append({"array_off": off, "count": n, "relocation": reloc,
                                  "bias": bias, "sample_ok": ok, "sample": k,
                                  "array_va": off + reloc})
                    break
        if not found:
            raise UniversalError(
                "flat loader found no pointer run whose adjacency-derived relocation "
                "yields structurally valid CandidateNodes (scanned %d runs of >=%d words)"
                % (len(self.runs), self.MIN_ARRAY))

        order, table_off = self._rom_profile_order([f["array_va"] for f in found])
        if order:
            found.sort(key=lambda f: order.get(f["array_va"], 10_000))
            for f in found:
                f["profile"] = order.get(f["array_va"])
        # The ROM table can name the same runtime array twice (two source
        # regions mounting at one address).  Vendor numbering is kept for
        # provenance, but the parser needs a unique key per image.
        used = set()
        nxt = max([f.get("profile") or 0 for f in found]) + 1
        for i, f in enumerate(found):
            f["vendor_profile"] = f.get("profile")
            v = f.get("profile")
            if v is None or v in used:
                v = nxt
                nxt += 1
            used.add(v)
            f["profile"] = v

        images = []
        for f in found:
            images.append(Image(bank_va=f["relocation"], profile=f["profile"],
                                source_offset=0, length=len(self.drdi),
                                relocation=f["relocation"], drdi=self.drdi,
                                label="flat/profile%d" % f["profile"],
                                candidate_hint=(f["array_off"], f["count"])))
        self.banks = [Bank(min(f["array_va"] for f in found), images, 0)]
        self.discovery = found
        self.rom_profile_table_off = table_off
        self.rep.info("flat_loader", "recovered flat capability profiles by relocation proof",
                      profiles=len(found), rom_profile_table=hex(table_off) if table_off else None,
                      corroborated=sum(1 for f in found if order and f["array_va"] in order),
                      detail=[{"profile": f["profile"], "array_off": hex(f["array_off"]),
                               "count": f["count"], "relocation": hex(f["relocation"]),
                               "array_va": hex(f["array_va"]), "bias": f["bias"]} for f in found])

    def _rom_profile_order(self, array_vas):
        """Corroborate the recovered arrays against md1rom's profile table.

        md1rom carries, per profile, the runtime address of that profile's node
        pointer array.  Nothing in the discovery above used it, so agreement is
        a genuine second witness -- and it also supplies the vendor's own
        profile numbering, which is what earlier device-specific work reported.
        """
        want = set(array_vas)
        if np is None or not want:
            return {}, None
        rw = np.frombuffer(self.rom[:len(self.rom) // 4 * 4], dtype="<u4")
        hit = np.isin(rw, np.fromiter(sorted(want), dtype="<u8").astype("<u4"))
        idx = np.flatnonzero(hit)
        if not len(idx):
            return {}, None
        # Longest run of consecutive words that are all recovered array addresses.
        best = (0, 0, 0)
        start = prev = int(idx[0])
        for z in list(idx[1:]) + [None]:
            z = None if z is None else int(z)
            if z != (prev + 1 if prev is not None else None):
                ln = prev - start + 1
                if ln > best[0]:
                    best = (ln, start, prev)
                if z is None:
                    break
                start = z
            prev = z
        ln, w0, w1 = best
        if ln < 2:
            return {}, None
        order = {}
        for i in range(ln):
            va = int(rw[w0 + i])
            order.setdefault(va, i)
        return order, w0 * 4

    def capability_bank(self) -> Bank:
        b = self.banks[0]
        self.rep.info("capability_bank", "flat container exposes one synthetic capability bank",
                      bank_va=hex(b.bank_va), live_profiles=len(b.images))
        return b

    def lte_tables(self, cap: Bank, rep: Reporter):
        """LTE CA rows in this family are a pointer array of row objects."""
        parser_tables = self.tables
        results = {}
        chosen = []
        for off, n in self.runs:
            for reloc, _bias in self._hypotheses(off, n):
                im = self._probe_image(reloc)
                for row_bias in (4, 0):
                    combos = []
                    okall = True
                    probe = min(n, 32)
                    for i in range(probe):
                        va = int(self._words[off // 4 + i])
                        base = im.resolve(va, 4)
                        if base is None:
                            okall = False
                            break
                        cb = parse_lte_fields(im, base + row_bias, parser_tables)
                        if cb is None:
                            okall = False
                            break
                    if not okall:
                        continue
                    for i in range(n):
                        va = int(self._words[off // 4 + i])
                        base = im.resolve(va, 4)
                        cb = None if base is None else parse_lte_fields(im, base + row_bias, parser_tables)
                        if cb is None:
                            combos = []
                            break
                        combos.append(cb)
                    if combos:
                        chosen.append({"array_off": off, "count": n, "relocation": reloc,
                                       "row_bias": row_bias, "rows": combos})
                        break
                if chosen and chosen[-1]["array_off"] == off:
                    break
        if not chosen:
            rep.warn("lte_table_missing", "flat loader found no fully valid LTE CA pointer array")
            return None, {}
        chosen.sort(key=lambda c: -len(c["rows"]))
        for i, c in enumerate(chosen):
            results[i] = c["rows"]
        rep.info("lte_table", "flat LTE CA row arrays recovered by relocation proof",
                 arrays=[{"index": i, "array_off": hex(c["array_off"]), "rows": len(c["rows"]),
                          "relocation": hex(c["relocation"]), "row_bias": c["row_bias"]}
                         for i, c in enumerate(chosen)])
        return self.banks[0], results

    @staticmethod
    def probe(rom: bytes, drdi: bytes) -> int:
        if np is None:
            return 0
        w = np.frombuffer(drdi[:len(drdi) // 4 * 4], dtype="<u4")
        return len(FlatLoader._runs_from_words(w, FlatLoader.MIN_ARRAY))


@dataclass
class RawDescriptor:
    count: int
    records_ptr: int
    variant_count: int
    variant_ptr: int
    records: list[tuple]
    units: int
    fsc: list[tuple]
    is_nr: bool

    @property
    def variants(self) -> int:
        return self.variant_count // self.units if self.units else 0


@dataclass
class RawCandidate:
    va: int
    meta0: int
    meta1: int
    lte: Optional[RawDescriptor]
    nr: Optional[RawDescriptor]


class GrammarParser:
    def __init__(self, loader: BaseLoader, rep: Reporter):
        self.loader = loader
        self.tables = loader.tables
        self.rep = rep
        self._descriptor_image = None
        self._descriptors = {}

    def _parse_desc(self, im: Image, ptr: int, nr: bool, strict_fsc=True) -> Optional[RawDescriptor]:
        # Nodes share descriptors extensively. Scope the cache to the actual
        # Image object: profiles can reuse VAs with different source bytes.
        # Retain only one image so flat relocation probes cannot accumulate.
        if self._descriptor_image is not im:
            self._descriptors.clear()
            self._descriptor_image = im
        key = (ptr, nr, strict_fsc)
        if key not in self._descriptors:
            self._descriptors[key] = self._decode_desc(im, ptr, nr, strict_fsc)
        return self._descriptors[key]

    def _decode_desc(self, im: Image, ptr: int, nr: bool, strict_fsc=True) -> Optional[RawDescriptor]:
        do = im.resolve(ptr, 16)
        if do is None:
            return None
        cnt, rp, vc, vp = D16.unpack_from(im.drdi, do)
        if not (1 <= cnt <= 16):
            return None
        recsz = 4 if nr else 3
        ro = im.resolve(rp, cnt * recsz)
        if ro is None:
            return None
        records, units = [], 0
        if nr:
            for k in range(cnt):
                band, ul, dl = NRREC.unpack_from(im.drdi, ro + 4 * k)
                if not (1 <= band <= 1024):
                    return None
                if not (0 <= dl < len(self.tables.nr_weights)) or self.tables.nr_weights[dl] <= 0:
                    return None
                if ul not in self.tables.nr_ul_absent:
                    if not (0 <= ul < len(self.tables.nr_weights)) or self.tables.nr_weights[ul] <= 0:
                        return None
                records.append((band, ul, dl)); units += self.tables.nr_weights[dl]
            fs = 3
        else:
            for k in range(cnt):
                idx, ul, dl = struct.unpack_from("<BBB", im.drdi, ro + 3 * k)
                if not (0 <= idx < len(self.tables.lte_band_map)):
                    return None
                band = self.tables.lte_band_map[idx]
                # 0 and 0xff terminate/void a slot; 252..255 are reserved
                # markers present in one of the two band-map copies.  No real
                # LTE band exceeds MAX_LTE_BAND, so anything above it is a
                # mis-resolved index, not a capability.
                if band == 0 or band > MAX_LTE_BAND:
                    return None
                if not (0 <= dl < len(self.tables.lte_weights)) or self.tables.lte_weights[dl] <= 0:
                    return None
                if ul != LTE_UL_ABSENT and not (0 <= ul < len(self.tables.lte_weights)):
                    return None
                records.append((band, ul, dl)); units += self.tables.lte_weights[dl]
            fs = 2
        if units <= 0 or vc <= 0 or vc % units != 0:
            return None
        vo = im.resolve(vp, vc * fs)
        if vo is None:
            return None
        if nr:
            fsc = [tuple(im.drdi[vo + 3*k:vo + 3*k + 3]) for k in range(vc)]
            if strict_fsc and any(t[0] not in SCS for t in fsc):
                return None
        else:
            fsc = [tuple(im.drdi[vo + 2*k:vo + 2*k + 2]) for k in range(vc)]
            # second byte is code-proven DL-MIMO status; reject unknown/unsupported.
            if strict_fsc and any(t[1] not in (0, 1, 2, 3) for t in fsc):
                return None
        return RawDescriptor(cnt, rp, vc, vp, records, units, fsc, nr)

    def parse_candidate_va(self, im: Image, va: int) -> Optional[RawCandidate]:
        o = im.resolve(va, 16)
        if o is None:
            return None
        meta0, meta1, lp, nptr = D16.unpack_from(im.drdi, o)
        lte = self._parse_desc(im, lp, False) if lp else None
        nr = self._parse_desc(im, nptr, True) if nptr else None
        if lte is None and nr is None:
            return None
        # A nonzero *in-image* descriptor pointer that failed is evidence this is not a candidate.
        if lp and im.contains_va(lp, 16) and lte is None:
            return None
        if nptr and im.contains_va(nptr, 16) and nr is None:
            return None
        return RawCandidate(va, meta0, meta1, lte, nr)

    def _pointer_runs(self, im: Image, minrun=4):
        """Aligned runs of u32 values pointing into this same image."""
        data = im.drdi[im.source_offset:im.end_source]
        if np is not None:
            arr = np.frombuffer(data[:len(data)//4*4], dtype="<u4")
            lo = im.bank_va + im.alias
            hi = im.bank_va + im.length + im.alias
            mask = (arr >= lo) & (arr < hi)
            idx = np.flatnonzero(mask)
            if not len(idx):
                return []
            runs=[]; s=prev=int(idx[0])
            for z in idx[1:]:
                z=int(z)
                if z != prev+1:
                    if prev-s+1 >= minrun: runs.append((im.source_offset+s*4, prev-s+1))
                    s=z
                prev=z
            if prev-s+1 >= minrun: runs.append((im.source_offset+s*4, prev-s+1))
            return runs
        out=[]; o=im.source_offset
        while o+4<=im.end_source:
            v=u32(im.drdi,o)
            if im.contains_va(v,1):
                st=o; n=0
                while o+4<=im.end_source and im.contains_va(u32(im.drdi,o),1):
                    n+=1; o+=4
                if n>=minrun: out.append((st,n))
            else: o+=4
        return out

    def _validate_run(self, im: Image, off: int, n: int):
        """Parse a pointer run and return its maximal fully-valid subrun."""
        validity=[]; parsed=[]
        for k in range(n):
            va=u32(im.drdi, off+4*k)
            c=self.parse_candidate_va(im,va)
            validity.append(c is not None); parsed.append(c)
        best=None; st=None
        for i, ok in enumerate(validity+[False]):
            if ok and st is None: st=i
            elif not ok and st is not None:
                ln=i-st
                if ln>=4:
                    cand=(ln,off+4*st,parsed[st:i],n)
                    if best is None or (cand[0],-cand[1])>(best[0],-best[1]): best=cand
                st=None
        return best

    def find_candidate_array(self, im: Image) -> tuple[int, list[RawCandidate], dict]:
        # Bank selection and extraction use the same immutable image and ROM
        # dictionaries. Reuse the full validation result, not just a sampled
        # hint; this also preserves discovery warnings from the first scan.
        cache = self.loader._candidate_arrays
        key = id(im)
        if key not in cache:
            cache[key] = (im, self._find_candidate_array(im))
        return cache[key][1]

    def _find_candidate_array(self, im: Image) -> tuple[int, list[RawCandidate], dict]:
        if im.candidate_hint:
            hoff, hn = im.candidate_hint
            best = self._validate_run(im, hoff, hn)
            if best is None:
                raise UniversalError(
                    "%s: loader-supplied candidate array at %#x (%d entries) contains no "
                    "structurally valid CandidateNode subrun" % (im.label, hoff, hn))
            ln, off, rows, rawrun = best
            nr_desc=sum(c.nr is not None for c in rows); lte_desc=sum(c.lte is not None for c in rows)
            return off, rows, {"file_offset":off,"relative_offset":off-im.source_offset,"count":ln,
                               "raw_pointer_run_count":rawrun,"nr_descriptors":nr_desc,
                               "lte_descriptors":lte_desc,"invariant_pass":ln,"invariant_total":ln,
                               "invariant_rate":1.0,"source":"loader_hint"}
        best = None
        all_valid = []
        for off, n in self._pointer_runs(im, minrun=4):
            # Maximal valid subruns trim unrelated neighbours at either end.
            cand = self._validate_run(im, off, n)
            if cand is None:
                continue
            all_valid.append(cand)
            if best is None or (cand[0],-cand[1])>(best[0],-best[1]):
                best = cand
        if best is None:
            raise UniversalError(f"no structurally valid CandidateNode pointer array found in {im.label}")
        # The longest valid subrun wins, but say so when there was competition:
        # on the current corpus there never is, and if that changes the union
        # is probably incomplete rather than the runner-up being noise.
        if len(all_valid) > 1:
            self.rep.warn("candidate_subrun_discarded",
                          "more than one structurally valid CandidateNode subrun in this image; "
                          "only the longest is decoded, so the result may be incomplete",
                          image=im.label,
                          subrun_lengths=sorted((c[0] for c in all_valid), reverse=True)[:8])
        ln, off, rows, rawrun = best
        # Exact invariant is already enforced per descriptor; summarize it explicitly.
        nr_desc=sum(c.nr is not None for c in rows); lte_desc=sum(c.lte is not None for c in rows)
        info={"file_offset":off,"relative_offset":off-im.source_offset,"count":ln,
              "raw_pointer_run_count":rawrun,"nr_descriptors":nr_desc,"lte_descriptors":lte_desc,
              "invariant_pass":ln,"invariant_total":ln,"invariant_rate":1.0,"source":"scan"}
        return off, rows, info


@dataclass(frozen=True)
class FeatureObj:
    mimo_status: int
    bw_code: int
    channel_bw_90: int


@dataclass
class FeatureTable:
    root_off: int
    root_va: int
    objects: list[FeatureObj]

    def __len__(self): return len(self.objects)


class FeatureResolver:
    def __init__(self, parser: GrammarParser):
        self.p = parser

    def _valid_obj(self, b: bytes) -> bool:
        if len(b)!=3: return False
        st,bw,b90=b
        if st not in (0,1,2,3) or b90 not in (0,1): return False
        if st==3:
            return True  # sentinel 03 14 01 deliberately has BW code outside 20-entry enum
        return 0 <= bw < len(self.p.tables.bw)

    def _absent_objects(self, im: Image) -> list:
        """File offsets of FeatureObj values that mean "not supported".

        Entry 0 of every feature table is such an object, so these offsets are
        the only possible table anchors.  What identifies one is its *meaning*
        -- mimo_status == 3 -- not a byte pattern: modern firmware writes
        03 14 01 (bandwidth code one past a 20-entry enum) while the MD800
        legacy family writes 03 0d 01 (the last code of a 14-entry enum).
        Matching the literal modern triple finds nothing on MD800, and the two
        stray 03 14 01 byte sequences that do occur there are unrelated data.
        """
        lo, hi = im.source_offset, im.end_source
        max_bw = len(self.p.tables.bw) + 8
        if np is not None:
            a = np.frombuffer(im.drdi[lo:hi], dtype=np.uint8)
            if len(a) < 3:
                return []
            m = (a[:-2] == 3) & (a[1:-1] <= max_bw) & (a[2:] <= 1)
            return [lo + int(i) for i in np.flatnonzero(m)]
        out = []
        for o in range(lo, hi - 2):
            if im.drdi[o] == 3 and im.drdi[o + 1] <= max_bw and im.drdi[o + 2] <= 1:
                out.append(o)
        return out

    def find_tables(self, im: Image, min_len=4) -> list:
        """Locate FeatureObj pointer tables.

        Anchored on the table's entry-0 object rather than swept word by word:
        the only words that can root a table are those holding the runtime
        address of an "unsupported" object.  On a 26 MB flat image that is the
        difference between millions of resolve() calls and a few dozen.
        """
        data = im.drdi
        anchors = self._absent_objects(im)
        if not anchors:
            return []
        # file offset -> runtime address is exactly the inverse of Image.resolve
        want = sorted(set(o + im.relocation + im.alias for o in anchors))

        roots = []
        if np is not None:
            n_words = (im.end_source - im.source_offset) // 4
            words = np.frombuffer(data, dtype="<u4", count=n_words,
                                  offset=im.source_offset)
            hits = np.flatnonzero(np.isin(words, np.array(want, dtype="<u8").astype("<u4")))
            roots = [im.source_offset + int(h) * 4 for h in hits]
        else:
            wantset = set(want)
            for off in range(im.source_offset, im.end_source - 4, 4):
                if u32(data, off) in wantset:
                    roots.append(off)

        out = []
        covered = 0
        for off in sorted(roots):
            # A long array in which every entry happens to address an absent
            # object makes every one of its positions look like a table root,
            # and each suffix then looks like a shorter table.  Only the
            # longest one can be real, so skip roots already inside an
            # accepted table.  Genuine tables do not nest: entries 1..n-1 of a
            # real table address supported objects, not absent ones.
            if off < covered:
                continue
            objs = []
            k = 0
            while off + 4 * k + 4 <= im.end_source:
                q = u32(data, off + 4 * k)
                qo = im.resolve(q, 3)
                if qo is None:
                    break
                raw = data[qo:qo + 3]
                if not self._valid_obj(raw):
                    break
                objs.append(FeatureObj(*raw))
                k += 1
            if len(objs) < min_len:
                continue
            covered = off + 4 * len(objs)
            # A table of nothing but absent objects carries no capability and
            # cannot be the DL or UL side of anything.
            if all(o.mimo_status == 3 for o in objs):
                continue
            root_va = off + im.relocation + im.alias
            out.append(FeatureTable(off, root_va, objs))
        return out

    def collect_refs(self, rows: list):
        """Walk the grammar once and record every feature reference.

        Pair selection used to re-walk the whole candidate set for each of the
        T*(T-1) orderings.  On Pixel that is 46k references per walk.  The walk
        result does not depend on which tables are assigned, so it is done once
        and every pair is then scored with array arithmetic.
        """
        dl_ids = []
        ul_ids = []
        comp_start = []
        comp_expect = []
        for c in rows:
            if not c.nr:
                continue
            d = c.nr
            for var in range(d.variants):
                cur = var * d.units
                for band, ulcls, dlcls in d.records:
                    n = self.p.tables.nr_weights[dlcls]
                    sl = d.fsc[cur:cur + n]
                    cur += n
                    if len(sl) != n:
                        return None
                    comp_start.append(len(ul_ids))
                    comp_expect.append(0 if ulcls in self.p.tables.nr_ul_absent
                                       else self.p.tables.nr_weights[ulcls])
                    for scs, ui, di in sl:
                        dl_ids.append(di)
                        ul_ids.append(ui)
                if cur != (var + 1) * d.units:
                    return None
        if not dl_ids:
            return None
        if np is not None:
            dl_a = np.array(dl_ids, dtype=np.int64)
            ul_a = np.array(ul_ids, dtype=np.int64)
            return {"dl": dl_a, "ul": ul_a,
                    "start": np.array(comp_start, dtype=np.int64),
                    "expect": np.array(comp_expect, dtype=np.int64),
                    "uniq_dl": np.unique(dl_a), "uniq_ul": np.unique(ul_a),
                    "cache": {}, "np": True}
        return {"dl": dl_ids, "ul": ul_ids, "start": comp_start, "expect": comp_expect,
                "uniq_dl": sorted(set(dl_ids)), "uniq_ul": sorted(set(ul_ids)),
                "cache": {}, "np": False}

    def _evaluate_pair(self, refs, dl: FeatureTable, ul: FeatureTable):
        if refs is None:
            return False, {"reason": "grammar_walk_failed"}
        dl_ids, ul_ids = refs["dl"], refs["ul"]
        dl_status = [o.mimo_status for o in dl.objects]
        ul_status = [o.mimo_status for o in ul.objects]
        if refs["np"]:
            maxdi = int(refs["uniq_dl"][-1]); maxui = int(refs["uniq_ul"][-1])
            if maxdi >= len(dl) or maxui >= len(ul):
                return False, {"reason": "feature_id_oob", "max_dl": maxdi, "max_ul": maxui}
            dls = np.array(dl_status, dtype=np.int64)
            uls = np.array(ul_status, dtype=np.int64)
            # Cheap first: only the *distinct* referenced ids matter for the DL
            # sanity test, and there are at most a few dozen of them.
            if bool((dls[refs["uniq_dl"]] == 3).any()):
                return False, {"reason": "dl_references_unsupported"}
            # Two UL tables that classify every referenced id the same way give
            # identical closure results, so memoise on that signature.  Flat
            # images offer many near-duplicate tables and this collapses them.
            sig = ("detail",) + tuple(bool(x) for x in (uls[refs["uniq_ul"]] != 3))
            cached = refs["cache"].get(sig)
            if cached is None:
                active = (uls[ul_ids] != 3).astype(np.int64)
                sums = (np.add.reduceat(active, refs["start"]) if len(refs["start"])
                        else np.zeros(0, dtype=np.int64))
                bad = np.flatnonzero(sums != refs["expect"])
                cached = (int(len(sums)), int(len(sums)) - int(len(bad)),
                          None if not len(bad) else (int(refs["expect"][int(bad[0])]),
                                                     int(sums[int(bad[0])])))
                refs["cache"][sig] = cached
            checks, ok, firstbad = cached
            if firstbad is not None:
                return False, {"reason": "ul_class_feature_mismatch",
                               "expected": firstbad[0], "active": firstbad[1]}
            refs_n = int(len(dl_ids))
        else:
            maxdi = max(dl_ids); maxui = max(ul_ids)
            if maxdi >= len(dl) or maxui >= len(ul):
                return False, {"reason": "feature_id_oob", "max_dl": maxdi, "max_ul": maxui}
            if any(dl_status[i] == 3 for i in dl_ids):
                return False, {"reason": "dl_references_unsupported"}
            starts = refs["start"] + [len(ul_ids)]
            checks = ok = 0
            for i, e in enumerate(refs["expect"]):
                act = sum(1 for j in range(starts[i], starts[i + 1]) if ul_status[ul_ids[j]] != 3)
                checks += 1
                if act != e:
                    return False, {"reason": "ul_class_feature_mismatch", "expected": e, "active": act}
                ok += 1
            refs_n = len(dl_ids)
        slack = (len(dl) - (maxdi + 1)) + (len(ul) - (maxui + 1))
        exact = int(len(dl) == maxdi + 1) + int(len(ul) == maxui + 1)
        return True, {"refs": refs_n, "max_dl_id": maxdi, "max_ul_id": maxui,
                      "ul_checks": checks, "ul_checks_ok": ok, "slack": slack,
                      "exact_lengths": exact}

    def _ul_closure(self, refs, ul: FeatureTable):
        """Does this table satisfy the UL class check for every component?

        Depends only on the UL table, which is the whole point: the check can
        be answered per table instead of per (DL, UL) ordering.
        """
        maxui = int(refs["uniq_ul"][-1]) if refs["np"] else max(refs["uniq_ul"])
        if len(ul) <= maxui:
            return False
        status = [o.mimo_status for o in ul.objects]
        if refs["np"]:
            uls = np.array(status, dtype=np.int64)
            # Keyed separately from _evaluate_pair's cache: same signature,
            # different payload.
            sig = ("closure",) + tuple(bool(x) for x in (uls[refs["uniq_ul"]] != 3))
            cached = refs["cache"].get(sig)
            if cached is None:
                active = (uls[refs["ul"]] != 3).astype(np.int64)
                sums = (np.add.reduceat(active, refs["start"]) if len(refs["start"])
                        else np.zeros(0, dtype=np.int64))
                cached = bool((sums == refs["expect"]).all())
                refs["cache"][sig] = cached
            return cached
        starts = list(refs["start"]) + [len(refs["ul"])]
        for i, e in enumerate(refs["expect"]):
            act = sum(1 for j in range(starts[i], starts[i + 1])
                      if status[refs["ul"][j]] != 3)
            if act != e:
                return False
        return True

    def _dl_admissible(self, refs, dl: FeatureTable):
        maxdi = int(refs["uniq_dl"][-1]) if refs["np"] else max(refs["uniq_dl"])
        if len(dl) <= maxdi:
            return False
        return not any(dl.objects[int(i)].mimo_status == 3 for i in refs["uniq_dl"])

    def pair_candidates(self, rows: list, tables: list, cap: int = 16):
        """Rank DL/UL feature-table assignments that survive every check.

        Every condition in _evaluate_pair reads either the DL table or the UL
        table, never both, so the admissible tables can be found in O(T) per
        side rather than O(T^2) over orderings.  That matters: one MD800
        profile offers 7489 candidate tables, where the product form spends
        over a minute to reach the same answer.

        Only the best `cap` tables per side (fewest unused entries first) are
        combined and scored in full, which bounds the work while keeping the
        ranking of the winner unchanged -- the tiebreak beyond slack is root
        proximity, and a table with more slack can never outrank one with less.
        """
        refs = self.collect_refs(rows)
        if refs is None:
            self.last_pair_stats = {"reason": "grammar_walk_failed"}
            return []
        maxdi = int(refs["uniq_dl"][-1]) if refs["np"] else max(refs["uniq_dl"])
        maxui = int(refs["uniq_ul"][-1]) if refs["np"] else max(refs["uniq_ul"])
        dls = sorted(((len(t) - (maxdi + 1), t) for t in tables if self._dl_admissible(refs, t)),
                     key=lambda x: (x[0], x[1].root_off))
        uls = sorted(((len(t) - (maxui + 1), t) for t in tables if self._ul_closure(refs, t)),
                     key=lambda x: (x[0], x[1].root_off))
        self.last_pair_stats = {"tables": len(tables), "dl_admissible": len(dls),
                                "ul_admissible": len(uls), "max_dl_id": maxdi,
                                "max_ul_id": maxui, "capped": len(dls) > cap or len(uls) > cap}
        good = []
        for _sd, dl in dls[:cap]:
            for _su, ul in uls[:cap]:
                if dl is ul:
                    continue
                ok, detail = self._evaluate_pair(refs, dl, ul)
                if ok:
                    # Lower slack, more exact boundaries, closer roots preferred.
                    score = (detail["slack"], -detail["exact_lengths"],
                             abs(dl.root_off - ul.root_off))
                    good.append((score, dl, ul, detail))
        return sorted(good, key=lambda x: x[0])


@dataclass
class ProfileState:
    image: Image
    candidates: list[RawCandidate]
    candidate_info: dict
    feature_tables: list[FeatureTable]
    feature_pairs: list
    dl_table: Optional[FeatureTable]=None
    ul_table: Optional[FeatureTable]=None
    feature_detail: dict=field(default_factory=dict)
    pair_stats: dict=field(default_factory=dict)


def establish_feature_pairs(states: list[ProfileState], rep: Reporter):
    """Resolve DL/UL feature-table direction across all live profiles.

    Usually one assignment passes.  If a profile has a symmetric tie (same table
    lengths/reference domains), use the unanimous root-order direction proved by
    the unambiguous sibling profiles in the same bank.
    """
    orientations=[]
    for s in states:
        if len(s.feature_pairs)==1:
            _,dl,ul,_=s.feature_pairs[0]
            orientations.append("DL_FIRST" if dl.root_off<ul.root_off else "UL_FIRST")
        elif s.feature_pairs:
            bestscore=s.feature_pairs[0][0]
            tied=[x for x in s.feature_pairs if x[0][:2]==bestscore[:2]]
            if len(tied)==1:
                _,dl,ul,_=tied[0]; orientations.append("DL_FIRST" if dl.root_off<ul.root_off else "UL_FIRST")
    hint=Counter(orientations).most_common(1)[0][0] if orientations else None
    unresolved=[]
    for s in states:
        if not s.feature_pairs:
            # No table pair satisfies domain closure and the UL class check for
            # this profile, so its candidate array is not proved and it must
            # not contribute rows.  Dropping it quietly would under-report and
            # aborting would discard the profiles that *are* proved, so it is
            # recorded as an error and the union is marked incomplete.
            unresolved.append(s)
            rep.fail("feature_pair_unresolved",
                     "no DL/UL feature-table assignment passes domain and UL-class closure; "
                     "this profile is excluded from the union",
                     profile=s.image.profile, image=s.image.label,
                     candidates=s.candidate_info.get("count"),
                     feature_tables=len(s.feature_tables))
            continue
        candidates=s.feature_pairs
        if hint:
            matching=[x for x in candidates if ("DL_FIRST" if x[1].root_off<x[2].root_off else "UL_FIRST")==hint]
            if matching: candidates=matching
        score,dl,ul,detail=candidates[0]
        s.dl_table,s.ul_table,s.feature_detail=dl,ul,{**detail,"score":list(score),"order_hint":hint,
            "dl_root_relative":dl.root_off-s.image.source_offset,"ul_root_relative":ul.root_off-s.image.source_offset,
            "dl_len":len(dl),"ul_len":len(ul)}
    resolved=[x for x in states if x not in unresolved]
    if not resolved:
        raise UniversalError("no capability profile could be resolved: no DL/UL feature-table "
                             "assignment passes domain and UL-class closure on any profile")
    rep.info("feature_orientation","resolved DL/UL feature-table orientation across capability profiles",
             hint=hint, profiles=len(resolved), unresolved=[x.image.profile for x in unresolved])
    return resolved, [x.image.profile for x in unresolved]


def combo_key(cb: export.Combo):
    return (
        tuple((c.band,c.dl_class,c.ul_class,tuple(c.dl_mimo)) for c in cb.lte),
        tuple((c.band,c.dl_class,c.ul_class,
               tuple((x.scs_khz,x.dl_mimo,x.dl_bw_mhz,x.ul_mimo,x.ul_bw_mhz) for x in c.ccs)) for c in cb.nr)
    )


def decode_profile_combos(p: GrammarParser, s: ProfileState) -> list[export.Combo]:
    assert s.dl_table and s.ul_table
    out=[]
    for cand in s.candidates:
        nvar=cand.nr.variants if cand.nr else 1
        lvar=cand.lte.variants if cand.lte else 1
        if nvar!=lvar and nvar!=1 and lvar!=1:
            raise UniversalError(f"{s.image.label}: LTE/NR variant cardinalities incompatible: LTE={lvar} NR={nvar}")
        variants=max(nvar,lvar)
        for v in range(variants):
            lte_comps=[]; nr_comps=[]
            if cand.lte:
                d=cand.lte; var=0 if lvar==1 else v; cur=var*d.units
                for band,ul,dl in d.records:
                    n=p.tables.lte_weights[dl]; sl=d.fsc[cur:cur+n]; cur+=n
                    if len(sl)!=n: raise UniversalError("LTE cursor underflow")
                    mm=[]
                    for b0,b1 in sl:
                        if b1 not in DL_MIMO:
                            raise UniversalError(f"{s.image.label}: LTE DL MIMO status {b1} is unsupported/rejected")
                        mm.append(DL_MIMO[b1])
                    lte_comps.append(export.LteComponent(band,dl,ul,mm))
                if cur != (var+1)*d.units: raise UniversalError("LTE cursor exhaustion failed")
            if cand.nr:
                d=cand.nr; var=0 if nvar==1 else v; cur=var*d.units
                for band,ul,dl in d.records:
                    n=p.tables.nr_weights[dl]; sl=d.fsc[cur:cur+n]; cur+=n
                    ccs=[]; active_ul=0
                    for scs,ui,di in sl:
                        if scs not in SCS: raise UniversalError(f"invalid NR SCS enum {scs}")
                        if di>=len(s.dl_table) or ui>=len(s.ul_table): raise UniversalError("feature id escaped table")
                        dob=s.dl_table.objects[di]; uob=s.ul_table.objects[ui]
                        if dob.mimo_status not in DL_MIMO: raise UniversalError("DL feature references unsupported object")
                        dl_bw=p.tables.bw[dob.bw_code]
                        if uob.mimo_status in UL_MIMO:
                            active_ul+=1; um=UL_MIMO[uob.mimo_status]; ub=p.tables.bw[uob.bw_code]
                        else:
                            um=ub=None
                        ccs.append(export.NrCC(SCS[scs],DL_MIMO[dob.mimo_status],dl_bw,um,ub))
                    exp=0 if ul in p.tables.nr_ul_absent else p.tables.nr_weights[ul]
                    if active_ul!=exp:
                        raise UniversalError(f"UL feature/class mismatch after pair resolution: band n{band}, active={active_ul}, expected={exp}")
                    # Normalize all discovered absent sentinels to the shared exporter's canonical 0x1c.
                    ul_norm=NR_UL_ABSENT_CANON if ul in p.tables.nr_ul_absent else ul
                    nr_comps.append(export.NrComponent(band,dl,ul_norm,ccs))
                if cur != (var+1)*d.units: raise UniversalError("NR cursor exhaustion failed")
            out.append(export.Combo(lte_comps,nr_comps))
    return out


def dedup_exact(combos: Iterable[export.Combo]) -> list[export.Combo]:
    seen=set(); out=[]
    for cb in combos:
        k=combo_key(cb)
        if k not in seen: seen.add(k); out.append(cb)
    return out


# LTE 32-byte rows
@dataclass
class LteRow:
    off:int; combo:export.Combo


def parse_lte_fields(im: Image, c0_off: int, tables: RomTables) -> Optional[export.Combo]:
    """Validate one LTE CA row starting at the absolute offset of its count field.

    Two packagings of the same six fields are proven in the corpus:

      modern grid   contiguous 32-byte row, c0 at row+8
      MD800 legacy  pointer array -> row object, c0 at (va - relocation) + 4

    Only the framing differs, so the invariant work lives here and the two
    scanners below merely supply the address of the count field.
    """
    if c0_off < im.source_offset or c0_off + 24 > im.end_source: return None
    c0,p0,c1,p1,c2,p2 = struct.unpack_from("<6I", im.drdi, c0_off)
    if not (1<=c0<=16): return None
    ro=im.resolve(p0,c0*3); mo=im.resolve(p1,c1)
    if ro is None or mo is None: return None
    recs=[]; units=0
    for k in range(c0):
        idx,ul,dl=struct.unpack_from("<BBB",im.drdi,ro+3*k)
        if idx>=len(tables.lte_band_map): return None
        band=tables.lte_band_map[idx]
        if band == 0 or band > MAX_LTE_BAND or dl >= 6 or (ul != LTE_UL_ABSENT and ul >= 6): return None
        units+=tables.lte_weights[dl]; recs.append((band,ul,dl))
    if c1!=units or c1<=0: return None
    mmraw=list(im.drdi[mo:mo+c1]);
    if any(x not in (2,3,4) for x in mmraw): return None
    cur=0; comps=[]
    for band,ul,dl in recs:
        n=tables.lte_weights[dl]; sl=mmraw[cur:cur+n];cur+=n
        # Proven common encoding in the LTE row table is actual-ish status 2->2Rx,3->4Rx.
        # 4->8Rx is allowed for forward compatibility.
        mmap={2:2,3:4,4:8}
        comps.append(export.LteComponent(band,dl,ul,[mmap[x] for x in sl]))
    if cur!=c1:return None
    return export.Combo(comps,[])


def parse_lte_row(im: Image, off: int, tables: RomTables) -> Optional[LteRow]:
    """Modern contiguous 32-byte row: flags, sub, then the six shared fields."""
    if off < im.source_offset or off + 32 > im.end_source: return None
    cb = parse_lte_fields(im, off + 8, tables)
    return None if cb is None else LteRow(off, cb)


def scan_lte_rows_bank(bank:Bank,tables:RomTables,rep:Reporter) -> dict[int,list[export.Combo]]:
    """Find 32-byte LTE row tables in every live image by 100% invariant runs."""
    result={}
    for im in bank.images:
        valids=[]
        # Rows are 32-byte aligned relative to the source image in solved modern layouts.
        # Scan each residue modulo 32 because source offsets need not themselves be aligned.
        best=[]
        for residue in range(0,32,4):
            rows=[]; cur=[]
            start=im.source_offset+((residue-im.source_offset)%32)
            offsets = range(start, im.end_source - 31, 32)
            if np is not None and offsets:
                # This is exactly parse_lte_fields' first count check, not a
                # new signature. Keep every possible row for full validation.
                counts = np.ndarray((len(offsets),), dtype="<u4", buffer=im.drdi,
                                    offset=start + 8, strides=(32,))
                offsets = (start + int(i) * 32
                           for i in np.flatnonzero((counts >= 1) & (counts <= 16)))
            previous = None
            for off in offsets:
                # Skipped impossible rows still terminate invariant runs.
                if previous is not None and off != previous + 32:
                    if len(cur)>=4: rows.append(cur)
                    cur=[]
                previous = off
                r=parse_lte_row(im,off,tables)
                if r:
                    cur.append(r)
                else:
                    if len(cur)>=4: rows.append(cur)
                    cur=[]
            if len(cur)>=4: rows.append(cur)
            if rows:
                m=max(rows,key=len)
                if len(m)>len(best): best=m
        if best:
            result[im.profile]=[r.combo for r in best]
            rep.info("lte_row_table","found invariant-valid LTE CA row table",
                     bank_va=hex(bank.bank_va),profile=im.profile,relative_off=hex(best[0].off-im.source_offset),rows=len(best))
    return result


def choose_lte_bank(loader:BaseLoader, cap:Bank, rep:Reporter):
    """Collect LTE CA rows from EVERY bank that contains invariant-valid ones.

    This used to pick a single bank -- the one with the widest coverage -- and
    discard the rest.  That lost real data: on every grid device examined the
    capability bank carries a second, small LTE table of single-band entries
    (b1A2A, b8A2A, b19A2A, b26A2A ...) that does not appear in the main CA
    table at all, 9 to 18 rows per device.

    Extraction does not get to choose.  A bank whose rows satisfy the class
    weight invariant, MIMO encoding and cursor exhaustion has earned its place
    in the output; deciding that single-carrier entries are uninteresting is a
    consumer's judgement, not the parser's.  The primary bank is still
    identified and reported, because knowing which table a row came from is
    useful, but nothing is dropped.

    Returns (primary_bank, {profile: [Combo]}) where the profile map is the
    union across banks, plus a per-bank breakdown on the reporter.
    """
    pool = [b for b in loader.banks if b.images]
    found = []
    for b in pool:
        rows = scan_lte_rows_bank(b, loader.tables, rep)
        if rows:
            score = (max(map(len, rows.values())), sum(map(len, rows.values())), len(rows))
            found.append((score, b, rows))
    # Preserve physical provenance for the GUI; the CLI's merged projection
    # remains available below.
    loader.lte_rows_by_bank = {b.table_index: rows for _score, b, rows in found}
    if not found:
        rep.warn("lte_bank_missing", "no invariant-valid modern 32-byte LTE CA table found")
        return None, {}
    found.sort(key=lambda x: x[0], reverse=True)
    primary_score, primary, primary_rows = found[0]

    merged: dict[int, list] = defaultdict(list)
    for _score, b, rows in found:
        for prof, combos in rows.items():
            merged[prof].extend(combos)
    merged = {k: dedup_exact(v) for k, v in sorted(merged.items())}

    rep.info("lte_bank", "collected LTE CA rows from every bank with invariant-valid rows",
             primary_bank=hex(primary.bank_va), banks=len(found),
             per_bank=[{"bank_va": hex(b.bank_va),
                        "primary": b.bank_va == primary.bank_va,
                        "profiles": {str(k): len(v) for k, v in sorted(rows.items())}}
                       for _s, b, rows in found],
             merged_profiles={str(k): len(v) for k, v in merged.items()})
    return primary, merged


def extract_capability(loader:BaseLoader, profile_arg:str, rep:Reporter):
    cap=loader.capability_bank(); parser=GrammarParser(loader,rep); fr=FeatureResolver(parser)
    states=[]
    for im in cap.images:
        if profile_arg!="all" and im.profile!=int(profile_arg): continue
        off,rows,info=parser.find_candidate_array(im)
        fts=fr.find_tables(im)
        pairs=fr.pair_candidates(rows,fts)
        st=ProfileState(im,rows,info,fts,pairs)
        st.pair_stats=dict(getattr(fr,"last_pair_stats",{}) or {})
        states.append(st)
        rep.info("candidate_array","candidate array located by structural invariant",
                 bank_va=hex(cap.bank_va),profile=im.profile,**info,feature_tables=len(fts),passing_feature_pairs=len(pairs))
    if not states: raise UniversalError(f"requested profile {profile_arg} is not live in capability bank")
    states, unresolved = establish_feature_pairs(states,rep)

    per_profile={}; union=[]
    for s in states:
        combos=decode_profile_combos(parser,s)
        exact=dedup_exact(combos); per_profile[s.image.profile]=exact; union.extend(exact)
    union=dedup_exact(union)

    lte_bank,lte_profiles=loader.lte_tables(cap,rep)
    if profile_arg!="all" and lte_profiles:
        lte_profiles={k:v for k,v in lte_profiles.items() if k==int(profile_arg)}
    lte_union=dedup_exact(x for rows in lte_profiles.values() for x in rows)
    return cap,states,per_profile,union,lte_bank,lte_profiles,lte_union,unresolved


def gui_family_counts(combos):
    """Separate mixed FR1/FR2 NR-only rows for the GUI's NRDC column.

    This is a band-based presentation classification, not proof of a separate
    firmware RF_NRDC namespace. Keep the existing export classification intact.
    """
    endc, nr, lte = export.classify(combos, 1)
    nrdc = sum(any(c.band < 257 for c in row.nr)
               and any(c.band >= 257 for c in row.nr) for row in nr)
    return {"endc": len(endc), "nrca": len(nr) - nrdc,
            "nrdc": nrdc, "lte": len(lte)}


def _secondary_combos(profile) -> list[export.Combo]:
    """Convert the secondary decoder's cycle-free dictionaries to shared rows."""
    out = []
    for row in profile.combos:
        lte = [export.LteComponent(int(c["band"]), int(c["dl_class"]),
                                   int(c["ul_class"]), list(c.get("dl_mimo", ())))
               for c in row.get("lte", ())]
        nr = []
        for c in row.get("nr", ()):
            nr.append(export.NrComponent(
                int(c["band"]), int(c["dl_class"]), int(c["ul_class"]),
                [export.NrCC(int(cc["scs_khz"]), int(cc["dl_mimo"]),
                             cc.get("dl_bw_mhz"), cc.get("ul_mimo"),
                             cc.get("ul_bw_mhz")) for cc in c.get("ccs", ())]))
        out.append(export.Combo(lte, nr))
    return dedup_exact(out)


def tensor_related_lte(loader: BaseLoader, secondary_profile: int,
                       lte_profiles: dict[int, list[export.Combo]]):
    """Return the Bank-5 rows selected by the split-CDF profile map.

    Tensor keeps independent selector maps for each bank.  A secondary profile
    may legitimately pair with more than one Bank-5 profile; preserving that
    union is more faithful than silently choosing the numerically equal slot.
    """
    if not isinstance(loader, TensorCdfLoader) or not lte_profiles:
        return dedup_exact(x for rows in lte_profiles.values() for x in rows)
    try:
        target_off, target_size = loader.sections[6 + 8]
        source_off, source_size = loader.sections[6 + 5]
        if target_size < 2 * 128 or source_size < 2 * 128:
            raise ValueError("CDF selector maps are shorter than 128 entries")
        target = struct.unpack_from("<128H", loader.header, target_off)
        source = struct.unpack_from("<128H", loader.header, source_off)
    except (IndexError, struct.error, ValueError):
        return dedup_exact(x for rows in lte_profiles.values() for x in rows)
    related = {int(source[i]) for i in range(128)
               if int(target[i]) == int(secondary_profile)
               and int(source[i]) in lte_profiles}
    if not related:
        return dedup_exact(x for rows in lte_profiles.values() for x in rows)
    return dedup_exact(x for profile in sorted(related) for x in lte_profiles[profile])


def tensor_secondary_summaries(loader: BaseLoader, profile_arg: str, rep: Reporter,
                               *, lte_profiles: dict[int, list[export.Combo]] | None = None,
                               lte_count: int = 0):
    """Decode optional Tensor bank-8 profiles for GUI rows/reporting."""
    if not isinstance(loader, TensorCdfLoader):
        return []
    decoded = decode_tensor_secondary(loader, 8, rep)
    out = []
    for item in decoded:
        if profile_arg != "all" and item.profile != int(profile_arg):
            continue
        combos = _secondary_combos(item)
        counts = gui_family_counts(combos)
        related_lte = (tensor_related_lte(loader, item.profile, lte_profiles)
                       if lte_profiles is not None else ())
        out.append({"bank_index": item.bank_index, "bank_va": hex(item.bank_va),
                    "profile": item.profile, "decoded_rows": len(combos),
                    "expanded_rows": item.expanded_rows,
                    "excluded_single_fr2": item.excluded_single_fr2,
                    "candidate_count": item.candidate_count,
                    "roots": {k: hex(v) for k, v in item.roots.items()},
                    "kinds": {"endc": counts["endc"], "nrca": counts["nrca"],
                              "lte": counts["lte"]},
                    "lte_count": len(related_lte) if lte_profiles is not None else int(lte_count),
                    "gui_counts": counts})
    return out


def serialize_profile_summary(states,per_profile):
    out=[]
    for s in states:
        combos=per_profile[s.image.profile]; en,nr,lt=export.classify(combos,1)
        out.append({"profile":s.image.profile,"image":s.image.to_dict(),"candidate_array":s.candidate_info,
                    "feature_tables_found":len(s.feature_tables),
                    "feature_tables":[{"root_relative":hex(t.root_off-s.image.source_offset),"length":len(t)}
                                      for t in sorted(s.feature_tables, key=lambda t:-len(t))[:12]],
                    "feature_resolution":s.feature_detail,
                    "feature_pair_search":s.pair_stats,
                    "decoded_rows":len(combos),"kinds":{"endc":len(en),"nrca":len(nr),"lte":len(lt)},
                    "gui_counts": gui_family_counts(combos)})
    return out


def max_class_used(combos) -> int:
    """Highest NR/LTE *DL* class index present in the decoded namespace.

    UL fields are deliberately excluded: a UL byte is either a weighted class
    or an absent-marker such as 0x1c / 0x11, and markers are not class indices.
    Feeding one to the cap-prune letter renderer would refuse a perfectly good
    extraction for no reason.
    """
    m = -1
    for cb in combos:
        for c in cb.lte:
            m = max(m, c.dl_class)
        for c in cb.nr:
            m = max(m, c.dl_class)
    return m


def render_guard(union, rep: Reporter):
    """Refuse cap-prune rather than mis-render an extended class namespace.

    The shared renderer maps a class index to a single letter.  Tensor firmware
    carries a longer NR class table (A..Q then R2..R12), which no single letter
    can express.  If such a class actually appears in the decoded namespace the
    right answer is a precise refusal, not a wrong letter.
    """
    hi = max_class_used(union)
    if hi >= len(export.CLASS_LETTERS):
        raise UniversalError(
            "decoded namespace uses DL class index %d but the shared cap-prune renderer "
            "only names %d classes (A..%s); extraction succeeded and the structure is in "
            "report.json, but text export is refused rather than mis-rendered"
            % (hi, len(export.CLASS_LETTERS), export.CLASS_LETTERS[-1]))
    rep.info("render_guard", "cap-prune class namespace is representable",
             max_dl_class=hi, renderer_classes=len(export.CLASS_LETTERS))


def exclude_mimo_subsets(combos):
    """Keep the MIMO Pareto frontier within identical radio configurations.

    Compare physical carriers in source order. Unknown/absent values only
    compare with the same unknown/absent pattern; bandwidth, SCS, bands and
    classes must match. Incomparable vectors are retained.
    """
    groups = defaultdict(list)
    for combo in combos:
        values = tuple(v for c in combo.lte for v in c.dl_mimo)
        values += tuple(v for c in combo.nr for cc in c.ccs for v in (cc.dl_mimo, cc.ul_mimo))
        key = (tuple((c.band, c.dl_class, c.ul_class, len(c.dl_mimo)) for c in combo.lte),
               tuple((c.band, c.dl_class, c.ul_class,
                      tuple((cc.scs_khz, cc.dl_bw_mhz, cc.ul_bw_mhz) for cc in c.ccs)) for c in combo.nr),
               tuple(v is None for v in values))
        vector = tuple(v for v in values if v is not None)
        groups[key].append((vector, combo))
    keep = set()
    for group in groups.values():
        frontier = []
        for vector, combo in sorted(group, key=lambda item: sum(item[0]), reverse=True):
            if any(all(a >= b for a, b in zip(other, vector)) for other in frontier):
                continue
            frontier.append(vector)
            keep.add(id(combo))
    return [c for c in combos if id(c) in keep]


def export_selected_formats(combos, lte_combos, device: str, out_dir: Path,
                            stem: str, formats, *, exclude_subsets=False) -> dict:
    """Write only the GUI-selected renderings from validated normalized rows."""
    selected = frozenset(formats)
    supported = frozenset(("b0cd", "b826", "cap_prune", "mtk_nr", "mtk_lte"))
    unknown = selected.difference(supported)
    if unknown:
        raise UniversalError("unknown GUI export format(s): " + ", ".join(sorted(unknown)))
    if not selected:
        raise UniversalError("no GUI export formats selected")

    before = {"capability": len(combos), "lte": len(lte_combos)}
    if exclude_subsets:
        combos = exclude_mimo_subsets(combos)
        lte_combos = exclude_mimo_subsets(lte_combos)
    out_dir.mkdir(parents=True, exist_ok=True)
    endc, nr_all, _lte_from_capability = export.classify(combos, 1)
    nrdc = [row for row in nr_all
            if any(c.band < 257 for c in row.nr)
            and any(c.band >= 257 for c in row.nr)]
    nrca = [row for row in nr_all if row not in nrdc]
    meta = {"device": device,
            "counts": {"endc": len(endc), "nrca": len(nrca), "nrdc": len(nrdc),
                       "lte": len(lte_combos)},
            "selected_formats": sorted(selected), "files": {},
            "exclude_mimo_subsets": exclude_subsets,
            "mimo_filter": {"before": before,
                            "after": {"capability": len(combos), "lte": len(lte_combos)}}}
    if "cap_prune" in selected and lte_combos:
        name = f"{stem}_lte_ca_exact_mimo_cap_prune.txt"
        path = out_dir / name
        path.write_text(export.render_lte_capprune(lte_combos, with_mimo=True), encoding="utf-8")
        meta["files"][name] = str(path)
    if "b826" in selected:
        results = [export.build_b826(endc, 3), export.build_b826(nrca, 4)]
        # Source 5 is the v21 RF_NRDC namespace.  Keep the old two-block file
        # byte-for-byte identical for ordinary FR1 images; append the block
        # only when validated mixed FR1/FR2 rows are actually present.
        if nrdc:
            results.append(export.build_b826(nrdc, 5))
        name = f"{stem}_0xB826_v21_combined.txt"
        path = out_dir / name
        path.write_text(export.combined_b826_text(results, device), encoding="utf-8")
        meta["files"][name] = str(path)
        for result in results:
            meta[result.tag] = {"source": result.source, "input_rows": result.input_rows,
                                "records": result.records, "bytes": len(result.blob),
                                "sha256": result.sha256, "unsupported": result.unsupported,
                                "decodes": result.verify()}
    if "b0cd" in selected:
        name = f"{stem}_0xB0CD_v41.txt"
        path = out_dir / name
        meta["RF_LTE_B0CD"] = write_b0cd_v41(lte_combos, path, device)
        meta["files"][name] = str(path)
    for fmt, rows, suffix, is_nr in (("mtk_nr", combos, "mtk_nr_trace", True),
                                    ("mtk_lte", lte_combos, "mtk_lte_ca_comb_info", False)):
        if fmt in selected:
            path = out_dir / f"{stem}_{suffix}.txt"
            meta[fmt] = write_trace(rows, path, device, nr=is_nr)
            meta["files"][path.name] = str(path)
    return meta


def run_secondary_extraction(loader: BaseLoader, args, rep: Reporter) -> dict:
    """Export one validated Tensor bank-8 profile.

    Bank 8 is a sibling namespace, not a profile number in the Bank-6
    CandidateNode table.  Keeping this path explicit prevents a GUI selection
    of ``bank 8 / profile 1`` from accidentally re-running Bank-6 profile 1.
    The LTE projection remains the complete Bank-5 union, since B0CD is a
    device-level LTE capability rather than an FR2-only table.
    """
    decoded = decode_tensor_secondary(loader, 8, rep)
    wanted = decoded
    if args.profile != "all":
        wanted = [x for x in decoded if x.profile == int(args.profile)]
    if not wanted:
        raise UniversalError(f"requested secondary profile {args.profile} is not live in bank 8")
    combos = dedup_exact(x for item in wanted for x in (_secondary_combos(item)))

    cap = loader.capability_bank()
    lte_bank, lte_profiles = loader.lte_tables(cap, rep)
    lte_union = dedup_exact(x for rows in lte_profiles.values() for x in rows)
    lte_for_selection = dedup_exact(
        x for item in wanted for x in tensor_related_lte(loader, item.profile, lte_profiles))
    selected_formats = getattr(args, "export_formats", ("b0cd", "b826", "cap_prune"))
    if "cap_prune" in selected_formats:
        render_guard(lte_for_selection, rep)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = export_selected_formats(combos, lte_for_selection, args.device, args.out,
                                   args.stem, selected_formats,
                                   exclude_subsets=getattr(args, "exclude_mimo_subsets", False))
    summaries = []
    for item in wanted:
        item_combos = _secondary_combos(item)
        counts = gui_family_counts(item_combos)
        related_lte = tensor_related_lte(loader, item.profile, lte_profiles)
        summaries.append({"bank_index": item.bank_index, "bank_va": hex(item.bank_va),
                          "profile": item.profile, "decoded_rows": len(item_combos),
                          "expanded_rows": item.expanded_rows,
                          "excluded_single_fr2": item.excluded_single_fr2,
                          "candidate_count": item.candidate_count,
                          "roots": {k: hex(v) for k, v in item.roots.items()},
                          "lte_count": len(related_lte),
                          "kinds": {"endc": counts["endc"], "nrca": counts["nrca"],
                                    "lte": counts["lte"]}, "gui_counts": counts})
    gui_counts = gui_family_counts(combos)
    return {"loader": loader.name, "rom_tables": loader.tables.as_dict(),
            "banks": [b.to_dict() for b in loader.banks],
            "capability_bank": hex(cap.bank_va), "capability_bank_index": cap.table_index,
            "secondary_profiles": summaries, "profiles": summaries,
            "gui_counts": gui_counts,
            "union": {"exact_rows": len(combos),
                      "kinds": {"endc": gui_counts["endc"],
                                "nrca": gui_counts["nrca"],
                                "lte": gui_counts["lte"]},
                      "complete": True, "unresolved_profiles": []},
            "lte_bank": hex(lte_bank.bank_va) if lte_bank else None,
            "lte_bank_index": lte_bank.table_index if lte_bank else None,
            "lte_profiles": {str(k): len(v) for k, v in lte_profiles.items()},
            "lte_union_exact_rows": len(lte_for_selection),
            "lte_union_all_profiles_exact_rows": len(lte_union), "export": meta}


def run_extraction(loader: BaseLoader, args, rep: Reporter) -> dict:
    """One extraction path for every container; the loader supplies geometry."""
    if getattr(args, "bank_only", False):
        return run_bank_extraction(loader, args, rep)
    if getattr(args, "bank_index", None) == 8 and isinstance(loader, TensorCdfLoader):
        return run_secondary_extraction(loader, args, rep)
    cap, states, per_profile, union, lte_bank, lte_profiles, lte_union, unresolved = \
        extract_capability(loader, args.profile, rep)
    selected_formats = getattr(args, "export_formats", ("b0cd", "b826", "cap_prune"))
    # The class-letter renderer is used only by LTE cap-prune. A B0CD-only or
    # B826-only export must not be rejected because an unrelated NR class has
    # no cap-prune spelling.
    if "cap_prune" in selected_formats:
        render_guard(lte_union, rep)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = export_selected_formats(union, lte_union, args.device, args.out,
                                   args.stem, selected_formats,
                                   exclude_subsets=getattr(args, "exclude_mimo_subsets", False))
    detail = {
        "loader": loader.name,
        "rom_tables": loader.tables.as_dict(),
        "banks": [b.to_dict() for b in loader.banks],
        "capability_bank": hex(cap.bank_va),
        "capability_bank_index": cap.table_index,
        "profiles": serialize_profile_summary(states, per_profile),
        "gui_counts": gui_family_counts(union),
        "union": {"exact_rows": len(union),
                  "kinds": dict(zip(("endc", "nrca", "lte"),
                                    map(len, export.classify(union, 1)))),
                  "complete": not unresolved,
                  "unresolved_profiles": unresolved},
        "lte_bank": hex(lte_bank.bank_va) if lte_bank else None,
        "lte_bank_index": lte_bank.table_index if lte_bank else None,
        "lte_profiles": {str(k): len(v) for k, v in lte_profiles.items()},
        "lte_union_exact_rows": len(lte_union),
        "export": meta,
    }
    if isinstance(loader, FlatLoader):
        detail["flat_discovery"] = [
            {k: (hex(v) if k in ("array_off", "relocation", "array_va") else v)
             for k, v in f.items()} for f in loader.discovery]
        detail["rom_profile_table"] = (hex(loader.rom_profile_table_off)
                                       if loader.rom_profile_table_off else None)
    return detail


def run_bank_extraction(loader: BaseLoader, args, rep: Reporter) -> dict:
    """Export only the physically selected bank/profile for GUI rows."""
    cap = loader.capability_bank()
    bank = next((b for b in loader.banks if b.table_index == args.bank_index), None)
    if bank is None:
        raise UniversalError(f"bank {args.bank_index} does not exist")
    selected = [im.profile for im in bank.images
                if args.profile == "all" or im.profile == int(args.profile)]
    if not selected:
        raise UniversalError(f"profile {args.profile} is not live in bank {args.bank_index}")
    combos = []
    profiles = []
    if bank.table_index == cap.table_index:
        _, states, per_profile, combos, _, _, _, unresolved = extract_capability(loader, args.profile, rep)
        profiles = serialize_profile_summary(states, per_profile)
        if unresolved:
            raise UniversalError(f"unresolved profiles: {unresolved}")
    elif isinstance(loader, TensorCdfLoader) and bank.table_index == 8:
        decoded = [p for p in decode_tensor_secondary(loader, 8, rep) if p.profile in selected]
        if set(p.profile for p in decoded) != set(selected):
            raise UniversalError("selected secondary profiles did not pass structural validation")
        combos = dedup_exact(c for p in decoded for c in _secondary_combos(p))
    if isinstance(loader, FlatLoader):
        _, lte_profiles = loader.lte_tables(cap, rep)
    else:
        lte_profiles = getattr(loader, "lte_rows_by_bank", {}).get(bank.table_index)
        if lte_profiles is None:
            lte_profiles = scan_lte_rows_bank(bank, loader.tables, rep)
    lte_profiles = {p: rows for p, rows in lte_profiles.items() if p in selected}
    lte = dedup_exact(c for rows in lte_profiles.values() for c in rows)
    formats = set(args.export_formats)
    effective = formats & ({"b826", "mtk_nr"} if combos else set())
    effective |= formats & ({"b0cd", "cap_prune", "mtk_lte"} if lte else set())
    if "cap_prune" in effective:
        render_guard(lte, rep)
    meta = (export_selected_formats(combos, lte, args.device, args.out, args.stem, effective,
                                    exclude_subsets=getattr(args, "exclude_mimo_subsets", False))
            if effective else {"files": {}, "selected_formats": []})
    meta["skipped_formats"] = sorted(formats - effective)
    counts = {**gui_family_counts(combos), "lte": len(lte)}
    return {"loader": loader.name, "banks": [b.to_dict() for b in loader.banks],
            "selected_bank_index": bank.table_index, "bank_only": True,
            "capability_bank_index": cap.table_index, "capability_bank": hex(cap.bank_va),
            "profiles": profiles, "gui_counts": counts,
            "lte_bank_index": bank.table_index if lte else None,
            "lte_profiles": {str(p): len(rows) for p, rows in lte_profiles.items()},
            "lte_union_exact_rows": len(lte),
            "union": {"kinds": counts, "complete": True, "exact_rows": len(combos),
                      "unresolved_profiles": []}, "export": meta}


def select_loader(args, rom: bytes, hdr_or_drdi: bytes, rep: Reporter, *, drdi_data=None):
    """Choose a container loader and record *why*.

    Unknown-family diagnosis is only as good as this record, so every attempt
    and every rejection reason is kept, including the ones that were skipped.
    """
    attempts = []
    want = args.loader

    if drdi_data is not None or args.drdi_data or want == "tensor":
        if drdi_data is None and not args.drdi_data:
            raise UniversalError("tensor loader requires --drdi <md1drdi_hdr> --drdi-data <md1drdi_data>")
        data = drdi_data if drdi_data is not None else args.drdi_data.read_bytes()
        if not TensorCdfLoader.probe(hdr_or_drdi):
            raise UniversalError("--drdi is not a recognized split-CDF header "
                                 "(expected 0x30000 bytes, 641 slot offsets, 11 bounds, 640 SHA-384 digests)")
        attempts.append({"loader": "tensor", "accepted": True, "evidence": "split-CDF header geometry matched"})
        return TensorCdfLoader(rom, hdr_or_drdi, data, rep), attempts, data

    drdi = hdr_or_drdi
    if want in ("auto", "grid"):
        hits = GridLoader.descriptor_hits(rom, drdi)
        score = GridLoader._dense_score(hits)
        try:
            loader = GridLoader(rom, drdi, rep, descriptor_hits=hits)
            loader.capability_bank()
            attempts.append({"loader": "grid", "accepted": True,
                             "evidence": "descriptor dense-run score %d, capability bank proved" % score})
            return loader, attempts, None
        except UniversalError as e:
            attempts.append({"loader": "grid", "accepted": False,
                             "evidence": "dense-run score %d" % score, "reason": str(e)})
            if want == "grid":
                raise
    if want in ("auto", "flat"):
        runs = FlatLoader.probe(rom, drdi)
        try:
            loader = FlatLoader(rom, drdi, rep)
            attempts.append({"loader": "flat", "accepted": True,
                             "evidence": "%d runtime pointer runs, %d relocations proved by grammar"
                                         % (runs, len(loader.discovery))})
            return loader, attempts, None
        except UniversalError as e:
            attempts.append({"loader": "flat", "accepted": False,
                             "evidence": "%d runtime pointer runs" % runs, "reason": str(e)})
            if want == "flat":
                raise
    raise UniversalError("no container loader accepted this image; attempts: "
                         + json.dumps(attempts))


def make_report_base(args, rom, drdi, rep):
    return {"tool": "mtk_universal.py", "version": VERSION, "inputs": {
        "rom": str(args.rom), "rom_bytes": len(rom), "rom_sha256": sha256_bytes(rom),
        "drdi": str(args.drdi) if args.drdi else None,
        "drdi_bytes": len(drdi), "drdi_sha256": sha256_bytes(drdi)},
        "requested_profile": args.profile, "requested_loader": args.loader}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--rom", type=Path)
    source.add_argument("--image", type=Path, help="packaged image or directory; automatically unwrap supported layers")
    ap.add_argument("--drdi", type=Path)
    ap.add_argument("--drdi-data", type=Path, help="Tensor split CDF data file (with --drdi pointing to header)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="all", help="profile number or all")
    ap.add_argument("--loader", default="auto", choices=["auto", "grid", "tensor", "flat"])
    ap.add_argument("--device", default="MediaTek modem")
    ap.add_argument("--stem", default="mtk")
    args = ap.parse_args(argv)
    if args.image and (args.drdi or args.drdi_data):
        ap.error("--image cannot be combined with --drdi or --drdi-data")
    if args.rom and not args.drdi:
        ap.error("--rom requires --drdi (plus --drdi-data for split CDF)")

    rep = Reporter()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"tool": "mtk_universal.py", "version": VERSION,
              "inputs": {"image": str(args.image)} if args.image else
                        {"rom": str(args.rom), "drdi": str(args.drdi)},
              "requested_profile": args.profile, "requested_loader": args.loader}
    status = 2
    try:
        unwrapped_data = None
        if args.image:
            from mtk_containers import unwrap_path, UnwrapError
            try:
                parts = unwrap_path(args.image)
            except UnwrapError as exc:
                report["unwrapping"] = exc.report
                raise
            rom, hdr_or_drdi, unwrapped_data = parts.rom, parts.drdi, parts.drdi_data
            report["unwrapping"] = parts.report
            selected = parts.report["selected"]
            report["inputs"].update({"rom": selected["md1rom"]["sources"][0],
                                     "drdi": selected["md1drdi_hdr" if unwrapped_data is not None else "md1drdi"]["sources"][0],
                                     "rom_bytes": len(rom), "rom_sha256": sha256_bytes(rom),
                                     "drdi_bytes": len(hdr_or_drdi), "drdi_sha256": sha256_bytes(hdr_or_drdi)})
        else:
            rom, hdr_or_drdi = args.rom.read_bytes(), args.drdi.read_bytes()
            report = make_report_base(args, rom, hdr_or_drdi, rep)
        loader, attempts, data = select_loader(args, rom, hdr_or_drdi, rep, drdi_data=unwrapped_data)
        report["loader_selection"] = attempts
        if data is not None:
            report["inputs"].update({
                "drdi_header_bytes": len(hdr_or_drdi),
                "drdi_header_sha256": sha256_bytes(hdr_or_drdi),
                "drdi_data": (report["unwrapping"]["selected"]["md1drdi_data"]["sources"][0]
                              if args.image else str(args.drdi_data)), "drdi_data_bytes": len(data),
                "drdi_data_sha256": sha256_bytes(data)})
        report.update(run_extraction(loader, args, rep))
        status = 0
    except Exception as e:
        report["fatal"] = {"type": type(e).__name__, "message": str(e)}
        rep.fail("fatal", str(e), exception=type(e).__name__)
        if os.environ.get("MTK_UNIVERSAL_TRACEBACK"):
            traceback.print_exc()
    report["validation"] = rep.as_dict()
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if status == 0:
        print(json.dumps({"status": "ok", "loader": report.get("loader"),
                          "union": report.get("union"),
                          "lte_union_exact_rows": report.get("lte_union_exact_rows"),
                          "report": str(args.out / "report.json")}, indent=2))
    else:
        print(json.dumps({"status": "failed", "fatal": report.get("fatal"),
                          "report": str(args.out / "report.json")}, indent=2), file=sys.stderr)
    return status


if __name__=="__main__":
    raise SystemExit(main())
