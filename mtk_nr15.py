"""NR15 grids, ROM-resident roots, and the older two-byte MIMO grammar.

The NR15.R3 consumers use three-byte NR records, not the modern
u16-band record, and obtains bandwidth from separate per-band ROM tables.
All offsets below describe structures/instructions, never device file offsets.
"""
from dataclasses import dataclass
import itertools
import struct

import mtk_nr15_bw

# nl1_rfd_cap_fill_in_ca_comb_info chooses the pair-table mode from a hardware
# register and SBP feature 0x4E. Neither is static, so the projection fixes the
# 120-MHz branch (inter-band 4, intra-band 3): hardware state <=1 with SBP
# 0x4E in {0x0C, 0x0D, 0x14}. Redmi Note 12 Pro trace capture matches it exactly;
# the SBP default branch would select the 100-MHz modes 6/5 instead.
BW_MODE_INTER, BW_MODE_INTRA = 4, 3
# nl1_cap_arrange_by_sim_type: 15 kHz carriers are capped at enum 7 (50 MHz),
# and an SCS variant survives only if sum(DL MHz * 30/SCS) <= 160.
SCS15_MAX_CODE, DL_SCS_BUDGET = 7, 160


@dataclass(frozen=True)
class CopyRegion:
    source: int
    destination: int
    length: int
    instruction: int


def pcrel48(rom, off, reg):
    if off < 0 or off + 6 > len(rom):
        return None
    h, lo, hi = struct.unpack_from("<HHh", rom, off)
    if h != (0x6003 | (reg << 5)):
        return None
    return (off + 6 + ((hi << 16) | lo)) & 0xFFFFFFFF


def discover_copy_regions(rom):
    """Read the memcpy triples in INT_InitRegions_C's nanoMIPS sequence.

    ADDIUPC a3,length; SW/LW a3,stack; BEQZC; ADDIUPC a1,source;
    LW a2,stack; EXT a1,a1,0,28; ADDIUPC a0,destination; BALC.
    The firmware strips the source's 0x70000000 alias before copying.
    """
    regions = []
    start = 0
    while (off := rom.find(b"\xe3\x60", start)) >= 0:
        start = off + 2
        if off % 2 or off + 40 > len(rom):
            continue
        length, source, destination = (pcrel48(rom, off, 7),
                                       pcrel48(rom, off + 16, 5),
                                       pcrel48(rom, off + 30, 4))
        if None in (length, source, destination) or rom[off + 26:off + 30] != b"\xa5\x80\xc0\xf6":
            continue
        source, length = source & 0x0FFFFFFF, length & 0x0FFFFFFF
        if (length and source + length <= len(rom)
                and 0x90000000 <= destination < 0xE0000000):
            regions.append(CopyRegion(source, destination, length, off))
    return regions


def header_geometry(rom, drdi):
    """CHECK_HEADER v6 has explicit DRDI source offset and byte length."""
    size = 0x200
    off = len(rom) - size
    if (off < 0 or rom[off:off + 12] != b"CHECK_HEADER"
            or struct.unpack_from("<I", rom, off + 12)[0] != 6
            or struct.unpack_from("<I", rom, len(rom) - 4)[0] != size):
        raise ValueError("NR15 requires a validated CHECK_HEADER v6 trailer")
    source, length = struct.unpack_from("<II", rom, off + 0x16C)
    if not 0 < source <= off or length != len(drdi):
        raise ValueError("CHECK_HEADER DRDI source/length does not match the input")
    return off, source


def band_bw_pairs(data, off, bw_count):
    result = {}
    previous = 0
    while off + 2 <= len(data):
        band, code = data[off:off + 2]
        off += 2
        if band == 0:
            return result if code == bw_count and len(result) >= 10 else None
        if not previous < band <= 255 or code >= bw_count:
            return None
        result[band] = code
        previous = band
    return None


class Nr15Decoder:
    def __init__(self, api, loader):
        self.u, self.loader = api, loader
        self.rom, self.drdi, self.rep = loader.rom, loader.drdi, loader.rep
        self.regions = discover_copy_regions(self.rom)
        if not self.regions:
            raise api.UniversalError("NR15: no validated ROM-to-RAM copy regions found")
        # A single backing buffer lets descriptors share immutable ROM data
        # while every DRDI pointer remains confined to its selected profile.
        self.data = self.drdi + self.rom
        self.rom_start = len(self.drdi)
        self.words = (api.np.frombuffer(self.rom, dtype="<u4", count=len(self.rom)//4)
                      if api.np is not None else None)
        decoder = self

        class RegionImage(api.Image):
            def resolve(self, va, size=1):
                off = super().resolve(va, size)
                if off is not None:
                    return off
                off = decoder.resolve_rom(va, size)
                return None if off is None else decoder.rom_start + off

        for bank in loader.banks:
            bank.images = [RegionImage(**{**im.__dict__, "drdi": self.data}) for im in bank.images]
        self.arrays = {}
        self._descriptors = {}
        self._image = None
        choices = []
        for bank in loader.banks:
            if not bank.images:
                continue
            options = {}
            for im in bank.images:
                found = self.complete_arrays(im, self.candidate)
                if not found:
                    break
                options[im.profile] = dict(found)
            if len(options) == len(bank.images):
                for root, arrays in self.profile_tables(bank.images, options):
                    choices.append((bank, root, arrays))
        if len(choices) != 1:
            raise api.UniversalError("NR15: expected one complete, firmware-corroborated profile root table")
        self.cap, self.profile_table, arrays = choices[0]
        self.profile_count = len(self.cap.images)
        for im, array in zip(self.cap.images, arrays):
            self.arrays[im.profile] = array
        self.band_limits = {}
        for im in self.cap.images:
            # Three equal-length profile-root groups precede the primary
            # roots; the span is profile-count dependent, not fixed at 0x24.
            ptr = api.u32(self.rom, self.profile_table - 3*self.profile_count*4 + 4*im.profile)
            off = self.resolve_rom(ptr, 2)
            limits = None if off is None else band_bw_pairs(self.rom, off, len(loader.tables.bw))
            if limits is None:
                raise api.UniversalError("NR15: invalid profile-specific per-band bandwidth root")
            self.band_limits[im.profile] = limits
        self.rf_limits_off, self.rf_limits = self.discover_rf_limits()
        self.bb_default_off, self.bb_default = self.discover_default_bb()
        pairs = [o for o in api.find_all(self.rom, mtk_nr15_bw.table_signature())]
        self.pair_table_off = pairs[0] if len(pairs) == 1 else None
        self.rep.info("nr15_roots", "validated complete ROM candidate arrays and their profile root table",
                      bank_va=hex(self.cap.bank_va), profile_table=hex(self.profile_table),
                      default_bb_rom_offset=hex(self.bb_default_off),
                      bw_pair_tables_rom_offset=None if self.pair_table_off is None else hex(self.pair_table_off),
                      arrays={str(p): {"rom_offset": hex(o), "rows": len(rows)}
                              for p, (o, rows) in self.arrays.items()},
                      copy_regions=[{"source": hex(r.source), "destination": hex(r.destination),
                                     "bytes": r.length, "instruction": hex(r.instruction)} for r in self.regions])
        self.support = {}
        # Optional runtime policy, e.g. a PLMN NR bandwidth override:
        # {(band, scs_index): max_code or None (band/SCS removed)}.
        self.bw_overrides = {}

    def array_aliases(self, off):
        """Runtime aliases for initialized ROM objects, not DRDI bank pointers."""
        aliases = {0x90000000 + off}
        for region in self.regions:
            if region.source <= off < region.source + region.length:
                runtime = region.destination + off - region.source
                aliases.update((runtime, (runtime & 0x1FFFFFFF) | 0x80000000,
                                ((runtime & 0x1FFFFFFF) | 0x80000000) - 0x70000000))
        return aliases

    def pcrel_refs(self, off):
        """Corroborate ROM data through nanoMIPS ADDIUPC consumers."""
        if not hasattr(self, "_pcrel_index"):
            self._pcrel_index = {}
            np = self.u.np
            if np is not None:
                hs = np.frombuffer(self.rom,dtype="<u2",count=len(self.rom)//2)
                ix = np.flatnonzero((hs[:-2] & 0xFC1F) == 0x6003)
                target = ((ix.astype("int64")*2 + 6 + hs[ix+1].astype("int64")
                           + (hs[ix+2].view("<i2").astype("int64") << 16)) & 0xFFFFFFFF)
                for index, value in zip(ix,target):
                    self._pcrel_index.setdefault(int(value),[]).append(int(index)*2)
            else:
                for ref in range(0,len(self.rom)-5,2):
                    h,lo,hi = struct.unpack_from("<HHh",self.rom,ref)
                    if h & 0xFC1F == 0x6003:
                        value = (ref+6+lo+(hi << 16)) & 0xFFFFFFFF
                        self._pcrel_index.setdefault(value,[]).append(ref)
        # Direct ROM references normalize to file offsets; initialized RAM
        # references normalize to the destination aliases of their copy region.
        aliases = self.array_aliases(off) | {off}
        return sorted({ref for value in aliases for ref in self._pcrel_index.get(value,())})

    def profile_tables(self, images, options):
        """Select complete per-profile arrays via one ordered firmware root vector.

        Identical ROM copies can all parse in a reused DRDI runtime window.
        Structural validity does not select the copy: the root vector does.
        The BB-root group and ADDIUPC consumers corroborate the surrounding
        block; conflicting complete vectors remain an error.
        """
        if [im.profile for im in images] != list(range(len(images))):
            return []
        first_refs = set()
        for off in options[images[0].profile]:
            for value in self.array_aliases(off):
                first_refs.update(ref for ref in self.u.find_all(self.rom,struct.pack("<I",value))
                                  if ref % 4 == 0)
        found = []
        span = 4*len(images)
        for root in sorted(first_refs):
            if root < 3*span or root+2*span > len(self.rom):
                continue
            arrays = []
            for im in images:
                off = self.resolve_rom(self.u.u32(self.rom,root+4*im.profile),4)
                if off not in options[im.profile]:
                    break
                bb = self.resolve_rom(self.u.u32(self.rom,root-3*span+4*im.profile),2)
                if bb is None or band_bw_pairs(self.rom,bb,len(self.loader.tables.bw)) is None:
                    break
                arrays.append((off,options[im.profile][off]))
            if len(arrays) != len(images):
                continue
            # The selector and following supported-NR root group are consumed
            # by the linkage setter; BB roots are consumed by the BW setter.
            if not all(self.pcrel_refs(off) for off in (root,root+span,root-3*span)):
                continue
            found.append((root,arrays))
        return found

    def discover_rf_limits(self):
        """Find the enum-associated RF band list and its consuming pointer.

        NR15 variants have different intervening header words. A fixed +4
        incorrectly lands on the MT6877 header, not the band list.
        """
        start = self.loader.tables.bw_off + 4*(len(self.loader.tables.bw)+1)
        found = []
        for off in range(start+4,min(start+24,len(self.rom)-1),2):
            limits = band_bw_pairs(self.rom,off,len(self.loader.tables.bw))
            if limits is not None and self.pcrel_refs(off):
                found.append((off,limits))
        if len(found) != 1:
            raise self.u.UniversalError("NR15: RF per-band bandwidth list lacks a unique consuming pointer")
        return found[0]

    def discover_default_bb(self):
        """Find NL1_RFD_Drdi_Set_BW_Data's default-branch per-band BB table.

        The setter selects the profile BB roots only when a customization flag
        is 1, otherwise it loads two ROM constants (single-band bitmask DB,
        then per-band BB DB) with ADDIUPC right after the profile-root loads.
        On both audited NR15 images that flag is BSS with a reader and no
        direct store, and the MT6877 capture matches the default table.
        """
        span = 4*self.profile_count
        found = set()
        for ref in self.pcrel_refs(self.profile_table - 3*span):
            for off in range(ref + 6, min(ref + 64, len(self.rom) - 5), 2):
                for reg in range(32):
                    target = pcrel48(self.rom, off, reg)
                    if (target is not None and target < len(self.rom)
                            and band_bw_pairs(self.rom, target, len(self.loader.tables.bw)) is not None):
                        found.add(target)
        if len(found) != 1:
            raise self.u.UniversalError("NR15: default per-band BB bandwidth table is not uniquely referenced")
        off = found.pop()
        return off, band_bw_pairs(self.rom, off, len(self.loader.tables.bw))

    def resolve_rom(self, va, size=1):
        if not 0 < va <= 0xFFFFFFFF or size < 0:
            return None
        if 0x90000000 <= va and va - 0x90000000 + size <= len(self.rom):
            return va - 0x90000000
        raw = va + 0x70000000 if 0x20000000 <= va < 0x30000000 else va
        physical = raw & 0x1FFFFFFF
        for region in self.regions:
            delta = physical - (region.destination & 0x1FFFFFFF)
            if 0 <= delta and delta + size <= region.length:
                return region.source + delta
        return None

    def complete_arrays(self, im, parse):
        """Every member must decode and the maximal pointer run must end in zero."""
        u = self.u
        if self.words is not None:
            mask = (self.words >= im.bank_va) & (self.words < im.end_va)
            ix = u.np.flatnonzero(mask)
            runs = []
            if len(ix):
                start = previous = int(ix[0])
                for index in list(ix[1:]) + [None]:
                    index = None if index is None else int(index)
                    if index != previous + 1:
                        runs.append((start * 4, previous - start + 1))
                        if index is None:
                            break
                        start = index
                    previous = index
        else:
            runs = []
            off = 0
            while off + 4 <= len(self.rom):
                start = off
                while off + 4 <= len(self.rom) and im.bank_va <= u.u32(self.rom, off) < im.end_va:
                    off += 4
                if off > start:
                    runs.append((start, (off - start)//4))
                else:
                    off += 4
        found = []
        for off, count in runs:
            if count < 4 or off + count * 4 + 4 > len(self.rom) or u.u32(self.rom, off + count * 4):
                continue
            rows = []
            for k in range(count):
                row = parse(im, u.u32(self.rom, off + 4*k))
                if row is None:
                    break
                rows.append(row)
            if len(rows) == count:
                found.append((off, rows))
        return found

    def find_profile_table(self, arrays):
        roots = [o for o, _ in arrays]
        found = []
        # Only scan values which resolve to the first recovered array.
        for region in self.regions:
            if not region.source <= roots[0] < region.source + region.length:
                continue
            runtime = region.destination + roots[0] - region.source
            aliases = {runtime, (runtime & 0x1FFFFFFF) | 0x80000000,
                       ((runtime & 0x1FFFFFFF) | 0x80000000) - 0x70000000}
            for value in aliases:
                for off in self.u.find_all(self.rom, struct.pack("<I", value)):
                    if off % 4 or off < 0x24 or off + 4*len(roots) > len(self.rom):
                        continue
                    if all(self.resolve_rom(self.u.u32(self.rom, off + 4*p), 4) == root
                           for p, root in enumerate(roots)):
                        found.append(off)
        found = sorted(set(found))
        return found[0] if len(found) == 1 else None

    def local_arrays(self, im, parse):
        """DRDI-local, zero-terminated pointer arrays with an aligned ROM root."""
        roots = set()
        if self.words is not None:
            roots.update(int(v) for v in self.words[(self.words >= im.bank_va)
                                                    & (self.words < im.end_va)])
        else:
            roots.update(self.u.u32(self.rom, off) for off in range(0, len(self.rom)-3, 4)
                         if im.bank_va <= self.u.u32(self.rom, off) < im.end_va)
        result = []
        for root in sorted(roots):
            start = im.resolve(root, 4)
            if start is None or start < im.source_offset or start + 4 > im.end_source:
                continue
            # Do not silently accept a valid suffix of a malformed table.
            if start >= im.source_offset + 4 and im.contains_va(self.u.u32(im.drdi, start-4)):
                continue
            rows, off = [], start
            while off + 4 <= im.end_source:
                ptr = self.u.u32(im.drdi, off)
                if ptr == 0:
                    if rows:
                        result.append((start, rows))
                    break
                row = parse(im, ptr)
                if row is None:
                    break
                rows.append(row)
                off += 4
        return result

    def descriptor(self, im, ptr, nr):
        if self._image is not im:
            self._descriptors.clear()
            self._image = im
        key = ptr, nr
        if key not in self._descriptors:
            self._descriptors[key] = self._descriptor(im, ptr, nr)
        return self._descriptors[key]

    def _descriptor(self, im, ptr, nr):
        u, tables = self.u, self.loader.tables
        off = im.resolve(ptr, 16)
        if off is None:
            return None
        count, rp, vc, vp = u.D16.unpack_from(im.drdi, off)
        if not 1 <= count <= 6 or not 1 <= vc <= 32:
            return None
        ro, vo = im.resolve(rp, count * 3), im.resolve(vp, vc * 2)
        if ro is None or vo is None:
            return None
        weights = tables.nr_weights if nr else tables.lte_weights
        records, units = [], 0
        for k in range(count):
            band, ul, dl = im.drdi[ro + 3*k:ro + 3*k + 3]
            if not nr:
                if band >= len(tables.lte_band_map):
                    return None
                band = tables.lte_band_map[band]
            if not 1 <= band <= (255 if nr else u.MAX_LTE_BAND) or dl >= len(weights) or not weights[dl]:
                return None
            absent = ul in tables.nr_ul_absent if nr else ul == u.LTE_UL_ABSENT
            if not absent and (ul >= len(weights) or not weights[ul]):
                return None
            if not absent and weights[ul] > weights[dl]:
                return None
            records.append((band, ul, dl))
            units += weights[dl]
        if vc != units:
            return None
        mimo = [tuple(im.drdi[vo + 2*k:vo + 2*k + 2]) for k in range(vc)]
        if any(dl not in u.DL_MIMO or ul not in (0, 1, 2, 3) for ul, dl in mimo):
            return None
        return u.RawDescriptor(count, rp, vc, vp, records, units, mimo, nr)

    def candidate(self, im, va):
        off = im.resolve(va, 16)
        if off is None or not im.source_offset <= off < im.end_source:
            return None
        m0, m1, lp, np = self.u.D16.unpack_from(im.drdi, off)
        # Unlike permissive modern discovery, both non-null pointers must
        # close. No valid suffix of a foreign profile's array is accepted.
        ld = self.descriptor(im, lp, False) if lp else None
        nd = self.descriptor(im, np, True) if np else None
        if nd is None or (lp and ld is None):
            return None
        return self.u.RawCandidate(va, m0, m1, ld, nd)

    def lte_row(self, im, va):
        off = im.resolve(va, 20)
        if off is None or not im.source_offset <= off <= im.end_source - 20:
            return None
        flags, count, rp, units, mp = struct.unpack_from("<5I", im.drdi, off)
        if not 1 <= count <= 6 or not 1 <= units <= 6:
            return None
        ro, mo = im.resolve(rp, count*3), im.resolve(mp, count)
        if ro is None or mo is None:
            return None
        tables = self.loader.tables
        records, total = [], 0
        for k in range(count):
            band, ul, dl = im.drdi[ro + 3*k:ro + 3*k + 3]
            if band >= len(tables.lte_band_map) or dl >= 6 or (ul != 6 and ul >= 6):
                return None
            band = tables.lte_band_map[band]
            if not 1 <= band <= self.u.MAX_LTE_BAND:
                return None
            records.append((band, ul, dl))
            total += tables.lte_weights[dl]
        mimo = im.drdi[mo:mo + count]
        if total != units or any(value not in (2, 3, 4) for value in mimo):
            return None
        comps = []
        for (band, ul, dl), status in zip(records, mimo):
            weight = tables.lte_weights[dl]
            comps.append(self.u.export.LteComponent(band, dl, ul,
                         [{2: 2, 3: 4, 4: 8}[status]] * weight))
        return self.u.export.Combo(comps, [])

    def lte_single(self, im, va):
        off = im.resolve(va, 2)
        if off is None:
            return None
        band, mimo = im.drdi[off:off+2]
        if band >= len(self.loader.tables.lte_band_map) or mimo not in (2, 3):
            return None
        band = self.loader.tables.lte_band_map[band]
        if not 1 <= band <= self.u.MAX_LTE_BAND:
            return None
        # EL1D_RF_Ue_Cap_Comb_Info_Query's 1CC branch excludes SDL UL.
        ul = self.u.LTE_UL_ABSENT if band in (29, 32) else 0
        return self.u.export.Combo([self.u.export.LteComponent(band, 0, ul, [2 if mimo == 2 else 4])], [])

    def band_inventory(self, bank, profiles):
        sets = [set(v["bands"]) for v in profiles.values()]
        return {"bank_index": bank.table_index, "bank_va": hex(bank.bank_va), "profiles": profiles,
                "union": sorted(set().union(*sets)), "intersection": sorted(set.intersection(*sets)),
                "profiles_consistent": all(s == sets[0] for s in sets[1:])}

    def supported_bands(self):
        if "nr" in self.support:
            return self.support
        nr = {}
        for im in self.cap.images:
            root_ref = self.profile_table + 4*self.profile_count + 4*im.profile
            ptr = self.u.u32(self.rom, root_ref)
            off = self.resolve_rom(ptr, 4)
            if off is None:
                raise self.u.UniversalError("NR15: missing standalone NR band array")
            start, bands = off, []
            while off + 4 <= len(self.rom) and len(bands) <= 40:
                va = self.u.u32(self.rom, off)
                if not va:
                    break
                obj = im.resolve(va, 3)
                if obj is None:
                    raise self.u.UniversalError("NR15: unresolved standalone NR band object")
                band, ul, dl = im.drdi[obj:obj+3]
                if (not 1 <= band <= 255 or ul not in self.u.UL_MIMO or dl not in self.u.DL_MIMO
                        or (bands and band <= bands[-1])):
                    raise self.u.UniversalError("NR15: invalid standalone NR band list")
                bands.append(band)
                off += 4
            if not bands or len(bands) > 40 or off + 4 > len(self.rom) or self.u.u32(self.rom, off):
                raise self.u.UniversalError("NR15: standalone NR band list is not terminated")
            nr[str(im.profile)] = {"bands": bands, "rom_array_offset": hex(start),
                                   "rom_pointer_refs": [hex(root_ref)]}
        self.support["nr"] = {"rat": "NR", **self.band_inventory(self.cap, nr)}
        return self.support

    def lte_tables(self):
        if hasattr(self, "_lte_result"):
            return self._lte_result
        results, primary, best = {}, None, 0
        provenance, singles, roots = [], {}, {}
        for bank in self.loader.banks:
            for im in bank.images:
                arrays = self.local_arrays(im, self.lte_row)
                if not arrays:
                    continue
                rows = [row for _, array in arrays for row in array]
                single_arrays = self.local_arrays(im, self.lte_single)
                # A supported-band array is strictly ordered, unlike CA rows.
                single_arrays = [(off, array) for off, array in single_arrays
                                 if 1 <= len(array) <= 25
                                 and all(a.lte[0].band < b.lte[0].band for a, b in zip(array, array[1:]))]
                if len(single_arrays) != 1:
                    raise self.u.UniversalError("NR15: expected one complete LTE single-band array per profile")
                if len(arrays) != 2:
                    raise self.u.UniversalError("NR15: both LTE CA root arrays must close completely")
                so, single_rows = single_arrays[0]
                roots.setdefault(bank.table_index, {})[im.profile] = (so+im.relocation,
                    *(off+im.relocation for off, _ in arrays))
                singles[str(im.profile)] = {"bands": [row.lte[0].band for row in single_rows],
                                            "relative_off": hex(so-im.source_offset),
                                            "runtime_va": hex(so+im.relocation),
                                            "rom_pointer_refs": [hex(ref) for ref in
                                                self.u.find_all(self.rom, struct.pack("<I", so+im.relocation))
                                                if ref % 4 == 0]}
                rows += single_rows
                results.setdefault(im.profile, []).extend(rows)
                if len(rows) > best:
                    primary, best = bank, len(rows)
                provenance.append({"bank": bank.table_index, "profile": im.profile,
                                   "arrays": [{"drdi_offset": hex(o), "rows": len(a)} for o, a in arrays],
                                   "single_band_array": {"drdi_offset": hex(so), "rows": len(single_rows)}})
        if not results:
            raise self.u.UniversalError("NR15: no complete LTE row pointer arrays found")
        if len(roots) != 1 or set(roots[primary.table_index]) != {im.profile for im in primary.images}:
            raise self.u.UniversalError("NR15: ambiguous or incomplete LTE profile bank")
        ordered = [roots[primary.table_index][im.profile] for im in primary.images]
        pattern = b"".join(struct.pack("<I", row[column]) for column in range(3) for row in ordered)
        sites = [off for off in self.u.find_all(self.rom, pattern) if off % 4 == 0]
        if len(sites) != 1:
            raise self.u.UniversalError("NR15: LTE arrays lack an unambiguous ROM profile root block")
        self.lte_profile_table = sites[0]
        self.rep.info("nr15_lte", "decoded complete LTE pointer arrays", sources=provenance)
        self.support["lte"] = {"rat": "LTE", **self.band_inventory(primary, singles)}
        self._lte_result = primary, {p: self.u.dedup_exact(rows) for p, rows in results.items()}
        return self._lte_result

    def projection_info(self):
        return {"grammar": "NR records <BBB>, MIMO entries <BB>; bandwidth is separate",
                "complete_namespace": "stored primary candidate and LTE root arrays, not all runtime capabilities",
                "candidate_profile_table_rom_offset": hex(self.profile_table),
                "lte_profile_table_rom_offset": hex(self.lte_profile_table),
                "bandwidth_source": "minimum of RF and default-branch BB per-band limits",
                "bb_customization_activation": ("inactive: the setter's flag is zero-initialized with no direct "
                                                "store; profile BB tables are reported but not applied"),
                "default_bb_rom_offset": hex(self.bb_default_off),
                "rf_bandwidth_rom_offset": hex(self.rf_limits_off),
                "bw_pair_tables_rom_offset": None if self.pair_table_off is None else hex(self.pair_table_off),
                "bw_pair_policy": {"inter_band_mode": BW_MODE_INTER, "intra_band_mode": BW_MODE_INTRA,
                                   "meaning": "120 MHz two-carrier policy (hardware state <=1, SBP 0x4E not default)"},
                "scs_rules": {"scs15_max_mhz": self.loader.tables.bw[SCS15_MAX_CODE],
                              "dl_budget": f"sum(MHz * 30/SCS) <= {DL_SCS_BUDGET}"},
                "rf_bandwidth_mhz": {str(b): self.loader.tables.bw[c] for b, c in self.rf_limits.items()},
                "default_bb_bandwidth_mhz": {str(b): self.loader.tables.bw[c] for b, c in self.bb_default.items()},
                "profile_bb_bandwidth_mhz": {str(p): {str(b): self.loader.tables.bw[c]
                                                      for b, c in limits.items()}
                                             for p, limits in self.band_limits.items()},
                "max_stored_nr_cc": max(row.nr.units for _, rows in self.arrays.values() for row in rows),
                "validated_projection": ("one- and two-carrier NR (NR-CA and EN-DC), per-CC bandwidth pairs and "
                                         "15/30 kHz SCS; exact against a Redmi Note 12 Pro capture"),
                "limitations": ["Static profile projection, not a captured runtime capability.",
                                "Two-carrier policy fixed to the 120 MHz branch; the SBP default branch is 100 MHz.",
                                "PLMN NR bandwidth overrides, SIM band/SCS exclusions and SBP MIMO downgrades "
                                "are not applied.",
                                "Supplementary-uplink (SUL) combinations are not reconstructed.",
                                "More than two NR carriers is not validated and is rejected."]}

    def decode(self, im, candidates):
        u, tables = self.u, self.loader.tables
        out = []
        for cand in candidates:
            lte = []
            if cand.lte:
                cur = 0
                for band, ul, dl in cand.lte.records:
                    n = tables.lte_weights[dl]
                    lte.append(u.export.LteComponent(band, dl, ul,
                               [u.DL_MIMO[x[1]] for x in cand.lte.fsc[cur:cur+n]]))
                    cur += n
            out.extend(self.nr_variants(lte, cand.nr))
        return u.dedup_exact(out)

    def nr_limits(self, records):
        """Per-record ceilings exactly as nl1_rfd_cap_bw_info derives them.

        min(RF, BB) for the first two NR records. Like the firmware, a band
        missing from the BB table keeps the previous record's ceiling.
        """
        limits, ceiling = [], len(self.loader.tables.bw)
        for band, _, _ in records[:2]:
            if band not in self.rf_limits:
                raise self.u.UniversalError(f"NR15: n{band} has no validated RF bandwidth limit")
            if band in self.bb_default:
                ceiling = min(self.rf_limits[band], self.bb_default[band])
            if ceiling == len(self.loader.tables.bw):
                ceiling = self.rf_limits[band]
            limits.append(ceiling)
        if any(code > 10 for code in limits):
            raise self.u.UniversalError("NR15 carrier bandwidth above 100 MHz is not validated")
        return limits

    def nr_variants(self, lte, nd):
        """Bandwidth pairs, then per-CC SCS expansion and the DL budget gate.

        Mirrors nl1_rfd_cap_fill_in_ca_comb_info and nl1_cap_arrange_by_sim_type:
        the pair function runs on all-15-kHz copies, each pair then gains every
        other per-CC SCS assignment in firmware order. Runtime-only filters
        (PLMN bandwidth overrides, SIM band/SCS exclusions) are not applied.
        """
        u, tables = self.u, self.loader.tables
        records = nd.records
        if len(records) > 2 or nd.units > 2:
            raise u.UniversalError("NR15: more than two NR carriers is not validated")
        limits = self.nr_limits(records)
        if nd.units == 1:
            pairs = [tuple(limits)]
        else:
            if self.pair_table_off is None:
                raise u.UniversalError("NR15: firmware bandwidth-pair tables were not found uniquely")
            mode = BW_MODE_INTER if len(records) == 2 else BW_MODE_INTRA
            pairs = mtk_nr15_bw.bandwidth_pairs(limits, mode, tables.bw)
        shapes = []
        for pair in pairs:
            ccs, cur = [], 0
            for index, (band, ul, dl) in enumerate(records):
                uplinks = 0 if ul in tables.nr_ul_absent else tables.nr_weights[ul]
                for k in range(tables.nr_weights[dl]):
                    um, dm = nd.fsc[cur + k]
                    if (k >= uplinks) != (um == 3):
                        raise u.UniversalError("NR15 UL class/MIMO closure failed")
                    # Inter-band pairs are per record, intra-band pairs per CC.
                    code = pair[index] if len(records) == 2 else pair[k]
                    ccs.append((index, code, dm, None if k >= uplinks else um))
                cur += tables.nr_weights[dl]
            shapes.append(ccs)
        ordered = [(ccs, (0,)*nd.units) for ccs in shapes]
        ordered += [(ccs, scs) for ccs in shapes
                    for scs in itertools.product((0, 1), repeat=nd.units) if any(scs)]
        out = []
        for ccs, assignment in ordered:
            total, per_record = 0, [[] for _ in records]
            for (index, code, dm, um), scs in zip(ccs, assignment):
                override = self.bw_overrides.get((records[index][0], scs), code)
                if override is None:
                    break
                code = min(code, override)
                if scs == 0:
                    code = min(code, SCS15_MAX_CODE)
                bw = tables.bw[code]
                total += bw * (2 if scs == 0 else 1)
                per_record[index].append(u.export.NrCC(u.SCS[scs], u.DL_MIMO[dm], bw,
                                         None if um is None else u.UL_MIMO[um],
                                         None if um is None else bw))
            else:
                if total > DL_SCS_BUDGET:
                    continue
                nr = [u.export.NrComponent(band, dl, u.NR_UL_ABSENT_CANON if ul in tables.nr_ul_absent else ul, ccs)
                      for (band, ul, dl), ccs in zip(records, per_record)]
                out.append(u.export.Combo(lte, nr))
        return out

    def extract_capability(self, profile):
        u = self.u
        states, per_profile = [], {}
        for im in self.cap.images:
            if profile != "all" and im.profile != int(profile):
                continue
            off, rows = self.arrays[im.profile]
            info = {"file_offset": off, "relative_offset": None, "count": len(rows),
                    "raw_pointer_run_count": len(rows), "nr_descriptors": len(rows),
                    "lte_descriptors": sum(r.lte is not None for r in rows),
                    "invariant_pass": len(rows), "invariant_total": len(rows),
                    "invariant_rate": 1.0, "source": "nr15_rom_root"}
            state = u.ProfileState(im, rows, info, [], [])
            state.feature_detail = {"source": "nr15_mimo_stream_and_band_limits"}
            states.append(state)
            per_profile[im.profile] = self.decode(im, rows)
            self.rep.check("nr15_complete_candidate_array", True, profile=im.profile, rows=len(rows),
                           rom_offset=hex(off), terminated=True)
        if not states:
            raise u.UniversalError(f"requested profile {profile} is not live in capability bank")
        union = u.dedup_exact(c for rows in per_profile.values() for c in rows)
        lte_bank, lte_profiles = self.lte_tables()
        if profile != "all":
            lte_profiles = {p: rows for p, rows in lte_profiles.items() if p == int(profile)}
        lte_union = u.dedup_exact(c for rows in lte_profiles.values() for c in rows)
        return self.cap, states, per_profile, union, lte_bank, lte_profiles, lte_union, []
