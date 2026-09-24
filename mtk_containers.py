"""Read-only, bounded modem-container unwrapping, independent of combo grammar.

Public API: unwrap_path(path, limits=Limits()) or unwrap_bytes(data, name=...).
Returns ModemParts with rom, drdi, optional drdi_data and a JSON-safe report.
Recognizes MTK partition headers, HBLR, extent-based ext4, Android sparse,
single-stream gzip/xz, and directories of extracted parts. No mounts, external
programs, temporary files, or filename-specific device dispatch are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import lzma
from pathlib import Path
import re
import struct
import zlib

MTK_MAGIC = b"\x88\x16\x88\x58"
MTK_HEADER = struct.Struct("<II32s10I")
SPARSE_MAGIC = b"\x3a\xff\x26\xed"
ROLES = ("md1rom", "md1drdi", "md1drdi_hdr", "md1drdi_data")
ROLE_RE = re.compile(r"(?:^|[_\-.])(md1drdi_hdr|md1drdi_data|md1drdi|md1rom)(?=$|[.\-_])", re.I)


class UnwrapError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


@dataclass(frozen=True)
class Limits:
    max_layer_bytes: int = 1024 * 1024 * 1024
    max_total_bytes: int = 3 * 1024 * 1024 * 1024
    max_depth: int = 12
    max_entries: int = 4096

    def __post_init__(self):
        if min(self.max_layer_bytes, self.max_total_bytes, self.max_depth, self.max_entries) <= 0:
            raise ValueError("all unwrapping limits must be positive")


@dataclass
class ModemParts:
    rom: bytes
    drdi: bytes
    drdi_data: bytes | None
    report: dict


def _u16(data, off):
    return struct.unpack_from("<H", data, off)[0]


def _u32(data, off):
    return struct.unpack_from("<I", data, off)[0]


def _role(name):
    match = ROLE_RE.search(name.rsplit("/", 1)[-1])
    return match.group(1).lower() if match else None


def _kind(data):
    if data.startswith(SPARSE_MAGIC): return "android-sparse"
    if data.startswith(b"\x1f\x8b"): return "gzip"
    if data.startswith(b"\xfd7zXZ\x00"): return "xz"
    if data.startswith(b"HBLR"): return "hblr"
    if len(data) >= 1082 and data[1080:1082] == b"\x53\xef": return "ext4"
    if data.startswith(MTK_MAGIC): return "mtk"
    return None


class _Ext4:
    """Extent/inode traversal only; deliberately not a filesystem repair tool."""
    def __init__(self, data, owner):
        self.data, self.owner = data, owner
        if len(data) < 2048:
            raise UnwrapError("truncated ext4 superblock")
        incompat = _u32(data, 1120)
        # meta_bg, compression, journal-device, recovery, inline data,
        # encryption and casefold need semantics this reader does not provide.
        allowed = 0x2 | 0x40 | 0x80 | 0x200 | 0x2000  # filetype, extents, 64bit, flex_bg, csum_seed
        if incompat & ~allowed:
            raise UnwrapError(f"unsupported ext4 incompat features {incompat & ~allowed:#x}")
        exponent = _u32(data, 1048)
        if exponent > 5:
            raise UnwrapError("unsupported ext4 block size (maximum 32 KiB)")
        self.block = 1024 << exponent
        blocks = _u32(data, 1028) | ((_u32(data, 1360) << 32) if incompat & 0x80 else 0)
        if blocks * self.block > len(data) or not blocks:
            raise UnwrapError("ext4 declared filesystem extends beyond input")
        self.end = blocks * self.block
        self.inodes = _u32(data, 1024)
        self.ipg = _u32(data, 1064)
        self.isize = _u16(data, 1112)
        self.dsize = _u16(data, 1278) if incompat & 0x80 else 32
        if (not self.ipg or not self.inodes or not 128 <= self.isize <= self.block
                or self.isize % 4 or not 32 <= self.dsize <= self.block or self.dsize % 8):
            raise UnwrapError("invalid ext4 inode/group descriptor geometry")
        if incompat & 0x80 and self.dsize < 64:
            raise UnwrapError("64-bit ext4 needs 64-byte group descriptors")
        self.gdt = (_u32(data, 1044) + 1) * self.block
        self.filetype = bool(incompat & 2)

    def inode(self, number):
        if not 1 <= number <= self.inodes:
            raise UnwrapError(f"ext4 inode {number} is outside the inode table")
        group, index = divmod(number - 1, self.ipg)
        gd = self.gdt + group * self.dsize
        if gd + self.dsize > self.end:
            raise UnwrapError("ext4 group descriptor outside filesystem")
        block = _u32(self.data, gd + 8)
        if self.dsize >= 64: block |= _u32(self.data, gd + 40) << 32
        off = block * self.block + index * self.isize
        if not block or off + self.isize > self.end:
            raise UnwrapError("ext4 inode outside filesystem")
        return self.data[off:off + self.isize]

    def contents(self, inode):
        size = _u32(inode, 4) | (_u32(inode, 108) << 32)
        self.owner.charge(size)
        if not _u32(inode, 32) & 0x80000:
            if size == 0: return b""
            raise UnwrapError("ext4 inode needs extents; inline/indirect blocks are unsupported")
        extents, seen = [], set()

        def visit(node, expected=None):
            self.owner.entry()
            if len(node) < 12 or _u16(node, 0) != 0xF30A:
                raise UnwrapError("invalid ext4 extent header")
            n, capacity, depth = struct.unpack_from("<HHH", node, 2)
            if (n > capacity or 12 + capacity * 12 > len(node) or depth > 5
                    or (expected is not None and depth != expected)):
                raise UnwrapError("invalid ext4 extent count/depth")
            last_key = -1
            for i in range(n):
                off = 12 + i * 12
                logical = _u32(node, off)
                if logical <= last_key: raise UnwrapError("unordered ext4 extent keys")
                last_key = logical
                if depth:
                    physical = _u32(node, off + 4) | (_u16(node, off + 8) << 32)
                    if physical in seen: raise UnwrapError("cyclic/shared ext4 extent node")
                    seen.add(physical)
                    disk = physical * self.block
                    if not physical or disk + self.block > self.end:
                        raise UnwrapError("ext4 extent node outside filesystem")
                    visit(self.data[disk:disk + self.block], depth - 1)
                else:
                    raw = _u16(node, off + 4)
                    count = raw - 32768 if raw > 32768 else raw
                    physical = _u32(node, off + 8) | (_u16(node, off + 6) << 32)
                    if not count or not physical or (physical + count) * self.block > self.end:
                        raise UnwrapError("invalid/out-of-range ext4 extent")
                    extents.append((logical, physical, count, raw > 32768))

        visit(inode[40:100])
        end = 0
        for logical, _, count, _ in extents:
            if logical < end: raise UnwrapError("overlapping/unordered ext4 extents")
            end = logical + count
        out = bytearray(size)
        for logical, physical, count, unwritten in extents:
            dst = logical * self.block
            n = min(count * self.block, max(0, size - dst))
            if n and not unwritten:
                src = physical * self.block
                out[dst:dst + n] = self.data[src:src + n]
        return bytes(out)

    def directories(self, number=2, path="", depth=0, seen=None):
        self.owner.depth(depth)
        seen = set() if seen is None else seen
        if number in seen: raise UnwrapError("cyclic/shared ext4 directory")
        seen.add(number)
        inode = self.inode(number)
        if _u16(inode, 0) & 0xF000 != 0x4000:
            raise UnwrapError("ext4 directory entry does not reference a directory")
        directory = self.contents(inode)
        entries, children, names = [], [], set()
        pos = 0
        while pos < len(directory):
            self.owner.entry()
            if pos + 8 > len(directory): raise UnwrapError("truncated ext4 directory entry")
            ino, rec = struct.unpack_from("<IH", directory, pos)
            n = directory[pos + 6] if self.filetype else _u16(directory, pos + 6)
            if rec < 8 or rec % 4 or rec > self.block - pos % self.block or pos + rec > len(directory) or n > rec - 8:
                raise UnwrapError("invalid ext4 directory entry length")
            raw = directory[pos + 8:pos + 8 + n]
            pos += rec
            if not ino or raw in (b".", b".."): continue
            if not raw or b"/" in raw or b"\0" in raw or raw in names:
                raise UnwrapError("invalid/duplicate ext4 filename")
            names.add(raw)
            name = raw.decode("utf-8", "surrogateescape")
            child = self.inode(ino)
            mode = _u16(child, 0) & 0xF000
            if mode == 0x4000:
                children.append((ino, path + name + "/"))
            elif mode == 0x8000:
                entries.append((name, child))
            # Symlinks and special files are never followed.
        yield path, ((name, self.contents(child)) for name, child in entries)
        for ino, childpath in children:
            yield from self.directories(ino, childpath, depth + 1, seen)


class _Unwrapper:
    def __init__(self, limits):
        self.limits = limits
        self.total = self.entries = 0
        self.report = {"layers": [], "partial_sets": [], "ignored": []}
        self.bundles = []

    def charge(self, size):
        if size < 0 or size > self.limits.max_layer_bytes or self.total + size > self.limits.max_total_bytes:
            raise UnwrapError(f"unwrapping byte limit exceeded (requested {size} bytes)")
        self.total += size

    def entry(self):
        self.entries += 1
        if self.entries > self.limits.max_entries: raise UnwrapError("unwrapping entry limit exceeded")

    def depth(self, depth):
        if depth > self.limits.max_depth: raise UnwrapError("unwrapping depth limit exceeded")

    def collect(self, entries, source, depth):
        """A sibling collection is one namespace; never join parts across sets."""
        self.depth(depth)
        parts, origins = {}, {}
        for name, data in entries:
            self.entry()
            role = _role(name)
            path = source + "!/" + name
            # Named raw parts are terminal unless they have an outer signature.
            # In particular, do not carve MTK-looking literals inside md1rom.
            kind = _kind(data)
            layer_depth = depth + 1
            while kind in ("gzip", "xz", "android-sparse"):
                self.depth(layer_depth)
                self.report["layers"].append({"format": kind, "source": path, "bytes": len(data)})
                data = self.expand(data, kind)
                path += "!/" + kind
                kind = _kind(data)
                layer_depth += 1
                self.depth(layer_depth)
            if role and kind is None:
                if role in parts and parts[role] != data:
                    raise UnwrapError(f"conflicting {role} parts in {source}")
                parts[role] = data
                origins.setdefault(role, []).append(path)
            else:
                self.walk(data, path, layer_depth)
        if not parts: return
        split = "md1drdi_hdr" in parts and "md1drdi_data" in parts
        if "md1rom" in parts and ("md1drdi" in parts or split):
            if "md1drdi" in parts and ("md1drdi_hdr" in parts or "md1drdi_data" in parts):
                raise UnwrapError(f"both flat and split DRDI parts present in {source}")
            if split:
                selected = {k: parts[k] for k in ("md1rom", "md1drdi_hdr", "md1drdi_data")}
            else:
                selected = {k: parts[k] for k in ("md1rom", "md1drdi")}
            self.bundles.append((selected, origins, source))
        else:
            self.report["partial_sets"].append({"source": source, "parts": sorted(parts)})

    def expand(self, data, kind):
        if kind == "android-sparse": return self.sparse(data)
        maximum = min(self.limits.max_layer_bytes, self.limits.max_total_bytes - self.total)
        if kind == "gzip":
            dec = zlib.decompressobj(31)
            out = dec.decompress(data, maximum + 1)
        else:
            dec = lzma.LZMADecompressor(memlimit=self.limits.max_layer_bytes)
            out = dec.decompress(data, max_length=maximum + 1)
        self.charge(len(out))
        if not dec.eof: raise UnwrapError(f"truncated or oversized {kind} stream")
        if dec.unused_data: raise UnwrapError(f"trailing data/multiple streams in {kind} wrapper")
        return out

    def sparse(self, data):
        if len(data) < 28: raise UnwrapError("truncated Android sparse header")
        _, major, minor, fh, ch, block, blocks, chunks, checksum = struct.unpack_from("<I4H4I", data)
        if major != 1 or fh < 28 or ch < 12 or fh > len(data) or not block or block % 4:
            raise UnwrapError("unsupported/invalid Android sparse geometry")
        self.charge(blocks * block)
        out = bytearray(blocks * block)
        pos, cursor, crc = fh, 0, 0
        for _ in range(chunks):
            self.entry()
            if pos + ch > len(data): raise UnwrapError("truncated sparse chunk header")
            kind, _, nblocks, size = struct.unpack_from("<HHII", data, pos)
            if size < ch or pos + size > len(data): raise UnwrapError("invalid sparse chunk extent")
            length = nblocks * block
            if cursor + length > len(out): raise UnwrapError("sparse chunk exceeds declared output")
            payload = memoryview(data)[pos + ch:pos + size]
            if kind == 0xCAC1 and len(payload) == length:
                out[cursor:cursor + length] = payload
            elif kind == 0xCAC2 and len(payload) == 4:
                out[cursor:cursor + length] = bytes(payload) * (length // 4)
            elif kind == 0xCAC3 and not payload:
                pass  # Android specifies zero bytes for don't-care CRC calculation.
            elif kind == 0xCAC4 and len(payload) == 4 and nblocks == 0:
                if _u32(payload, 0) != crc: raise UnwrapError("Android sparse chunk CRC32 mismatch")
            else:
                raise UnwrapError(f"unsupported/malformed sparse chunk {kind:#x}")
            if length: crc = zlib.crc32(memoryview(out)[cursor:cursor + length], crc)
            cursor += length
            pos += size
        if cursor != len(out) or pos != len(data): raise UnwrapError("sparse extent/input exhaustion failed")
        if checksum and checksum != crc: raise UnwrapError("Android sparse image CRC32 mismatch")
        return bytes(out)

    def hblr(self, data):
        layer = self.report["layers"][-1]
        layer["members"] = []
        if len(data) < 64 or _u32(data, 4) != len(data): raise UnwrapError("HBLR declared size mismatch")
        count = _u32(data, 48)
        start = 64 + count * 48
        if not 1 <= count <= 128 or start > len(data): raise UnwrapError("invalid HBLR segment count")
        names, spans, records = set(), [], []
        for i in range(count):
            self.entry()
            off = 64 + i * 48
            if data[off:off + 4] != b"SEGM": raise UnwrapError("missing HBLR SEGM signature")
            name = data[off + 4:off + 36].split(b"\0", 1)[0].decode("ascii")
            src, logical, stored = struct.unpack_from("<III", data, off + 36)
            if not name or name in names: raise UnwrapError("invalid/duplicate HBLR segment name")
            # HBLR may round stored extents up to 16 bytes. The extra bytes
            # are padding (not necessarily zero), not compressed payload.
            if logical != stored and stored != (logical + 15) // 16 * 16:
                raise UnwrapError(f"unsupported HBLR segment size relationship: {name}")
            if src < start or src + stored > len(data): raise UnwrapError("HBLR segment outside container")
            names.add(name)
            spans.append((src, src + stored))
            records.append((name, src, logical))
            layer["members"].append({"name": name, "offset": src, "bytes": logical,
                                     "stored_bytes": stored, "padding_bytes": stored - logical})
        spans.sort()
        if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
            raise UnwrapError("overlapping HBLR segments")
        for name, off, size in records:
            self.charge(size)
            yield name, data[off:off + size]

    def mtk(self, data):
        layer = self.report["layers"][-1]
        layer["members"] = []
        pos = 0
        while True:
            pos = data.find(MTK_MAGIC, pos)
            if pos < 0: break
            self.entry()
            if pos + MTK_HEADER.size > len(data): break
            fields = MTK_HEADER.unpack_from(data, pos)
            if fields[5] != 0x58891689:
                pos += 4
                continue
            name = fields[2].split(b"\0", 1)[0].decode("ascii")
            size = fields[1] | (fields[11] << 32)
            off = fields[6]
            if not name or off < 512 or pos + off + size > len(data):
                raise UnwrapError(f"invalid/truncated MTK partition at {pos:#x}")
            self.charge(size)
            layer["members"].append({"name": name, "header_offset": pos,
                                     "offset": pos + off, "bytes": size})
            yield name, data[pos + off:pos + off + size]
            pos += off + size  # Inner containers are visited recursively, not carved twice.

    def walk(self, data, source, depth):
        self.depth(depth)
        kind = _kind(data)
        if kind is None and MTK_MAGIC in data: kind = "mtk"
        if kind is None:
            self.report["ignored"].append({"source": source, "bytes": len(data)})
            return
        self.report["layers"].append({"format": kind, "source": source, "bytes": len(data)})
        if kind in ("gzip", "xz", "android-sparse"):
            self.walk(self.expand(data, kind), source + "!/" + kind, depth + 1)
        elif kind == "hblr":
            self.collect(self.hblr(data), source, depth)
        elif kind == "mtk":
            self.collect(self.mtk(data), source, depth)
        else:
            fs = _Ext4(data, self)
            self.report["layers"][-1]["metadata_checksums_verified"] = False
            for path, entries in fs.directories(depth=depth):
                container_path = source + ("!/" + path.rstrip("/") if path else "")
                self.collect(entries, container_path, depth + path.count("/") + 1)

    def directory(self, path, depth=0):
        self.depth(depth)
        self.report["layers"].append({"format": "directory", "source": str(path)})
        children = []

        def entries():
            for child in sorted(path.iterdir()):
                self.entry()
                if child.is_symlink() or (hasattr(child, "is_junction") and child.is_junction()):
                    continue
                if child.is_dir(): children.append(child)
                elif child.is_file():
                    self.charge(child.stat().st_size)
                    yield child.name, child.read_bytes()
        self.collect(entries(), str(path), depth)
        for child in children: self.directory(child, depth + 1)

    def finish(self):
        unique = {}
        for parts, origins, source in self.bundles:
            key = tuple((name, hashlib.sha256(data).hexdigest()) for name, data in sorted(parts.items()))
            unique.setdefault(key, []).append((parts, origins, source))
        self.report["candidate_sets"] = [
            {"sources": [c[2] for c in copies], "sha256": dict(key)} for key, copies in unique.items()]
        self.report["processed_bytes"] = self.total
        if not unique:
            raise UnwrapError("no complete modem set found (need md1rom and md1drdi, or md1rom and both split-CDF parts)")
        if len(unique) != 1:
            raise UnwrapError(f"{len(unique)} different modem sets found; pass the intended image or parts directory explicitly")
        copies = next(iter(unique.values()))
        parts, origins, _ = copies[0]
        split = "md1drdi_hdr" in parts
        self.report.update(packaging="tensor-split" if split else "single-drdi",
                           selected={name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                            "sources": origins[name]} for name, data in parts.items()},
                           identical_sets=len(copies))
        return ModemParts(parts["md1rom"], parts["md1drdi_hdr" if split else "md1drdi"],
                          parts.get("md1drdi_data"), self.report)


def _run(action, limits):
    worker = _Unwrapper(limits)
    try:
        action(worker)
        return worker.finish()
    except (UnwrapError, OSError, ValueError, struct.error, zlib.error, lzma.LZMAError) as exc:
        worker.report["error"] = str(exc)
        raise UnwrapError(str(exc), worker.report) from exc


def unwrap_bytes(data: bytes, name="image", *, limits=Limits()) -> ModemParts:
    """Unwrap one in-memory container. The name is provenance, not format selection."""
    def action(worker):
        worker.charge(len(data))
        worker.report["input"] = {"source": str(name), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        worker.walk(data, str(name), 0)
    return _run(action, limits)


def unwrap_path(path, *, limits=Limits()) -> ModemParts:
    """Unwrap a file or directory; distinct complete modem sets are an error."""
    path = Path(path)
    def action(worker):
        if path.is_dir():
            worker.directory(path)
        else:
            worker.charge(path.stat().st_size)
            data = path.read_bytes()
            worker.report["input"] = {"source": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            worker.walk(data, str(path), 0)
    return _run(action, limits)
