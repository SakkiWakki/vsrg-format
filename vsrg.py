"""Minimal `.vsrg` V1 reader/writer for numpy consumers.

Drop this anywhere on the path (``analysis/core/vsrg.py`` fits the layout in
vsrg-analysis). Only depends on numpy.

`.grid` and `.notes` come back as structured numpy arrays that are *views*
into the file buffer, not copies. The format is little-endian and packed, so
``np.frombuffer`` maps it directly with no byte swapping and no decode pass:

    v = vsrg.load("chart.vsrg", mmap=True)
    v.notes["start"]          # strided view, no copy
    v.notes[v.is_hold]        # every LN

Writing round-trips the same structures:

    open("out.vsrg", "wb").write(vsrg.dumps(v))
"""

from __future__ import annotations

import mmap as _mmap
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"beatsoup"
END = b"\xde\xad\xbe\xa7"
VERSION = 1

#: bytes 1-2 version + bytes 3-64 reserved
_FIXED_PREFIX = 64
#: five u32 section sizes
_SIZE_TABLE = 20


class VsrgError(Exception):
    pass


# ── dtypes ──────────────────────────────────────────────────────────────────
# Packed (align=False is numpy's default for a field list), matching the
# spec's byte layout exactly. Note records land on a 2-mod-4 offset, which
# numpy handles fine; it just can't be reinterpreted as a C struct.


def grid_dtype(n_params: int) -> np.dtype:
    f = [("tick", "<u4"), ("function", "<u4")]
    if n_params:
        f.append(("params", "<u4", (n_params,)))
    return np.dtype(f)


def note_dtype(n_params: int) -> np.dtype:
    f = [("col", "u1"), ("type", "u1"), ("start", "<u4"), ("end", "<u4")]
    if n_params:
        f.append(("params", "<u4", (n_params,)))
    return np.dtype(f)


def _n_params(dt: np.dtype) -> int:
    names = dt.names or ()
    return dt["params"].shape[0] if "params" in names else 0


# ── the file ────────────────────────────────────────────────────────────────


@dataclass
class Vsrg:
    title: str = ""
    artist: str = ""
    tags: str = ""
    song_file: str = ""
    resources: list[str] = field(default_factory=list)

    column_count: int = 4
    note_type: int = 0

    grid: np.ndarray = field(default_factory=lambda: np.empty(0, grid_dtype(0)))
    notes: np.ndarray = field(default_factory=lambda: np.empty(0, note_dtype(0)))

    version: int = VERSION
    #: keeps the mmap/bytes backing alive while views into it exist
    _buf: object = None

    # Derived from the array dtypes rather than stored, so they cannot desync
    # from the data they describe.
    @property
    def grid_param_count(self) -> int:
        return _n_params(self.grid.dtype)

    @property
    def note_param_count(self) -> int:
        return _n_params(self.notes.dtype)

    # ── analysis helpers ────────────────────────────────────────────────
    @property
    def is_hold(self) -> np.ndarray:
        """Boolean mask; False for taps, True for anything spanning ticks."""
        return self.notes["end"] != self.notes["start"]

    @property
    def hand(self) -> np.ndarray:
        """0 = left, 1 = right. Splits the keyboard down the middle."""
        return (self.notes["col"] >= self.column_count // 2).astype(np.uint8)

    def chord_sizes(self) -> np.ndarray:
        """Per note, how many notes share its start tick (1 = solo)."""
        if not len(self.notes):
            return np.empty(0, np.intp)
        _, inv, counts = np.unique(
            self.notes["start"], return_inverse=True, return_counts=True
        )
        return counts[inv]

    def notes_per_column(self) -> np.ndarray:
        return np.bincount(self.notes["col"], minlength=self.column_count)

    def __repr__(self) -> str:
        return (
            f"<Vsrg {self.title!r} by {self.artist!r} "
            f"{self.column_count}K notes={len(self.notes)} "
            f"grid={len(self.grid)} resources={len(self.resources)}>"
        )


# ── reading ─────────────────────────────────────────────────────────────────


def _split_cstrings(block: bytes, what: str) -> list[str]:
    """Every string is NUL-terminated, so the trailing split is always empty."""
    if block and not block.endswith(b"\0"):
        raise VsrgError(f"{what}: last string is not NUL-terminated")
    parts = block.split(b"\0")[:-1] if block else []
    try:
        return [p.decode("utf-8") for p in parts]
    except UnicodeDecodeError as e:
        raise VsrgError(f"{what}: not valid UTF-8") from e


def loads(buf) -> Vsrg:
    mv = memoryview(buf).cast("B")
    n = len(mv)

    def need(off, count, what):
        if off + count > n:
            raise VsrgError(f"truncated: need {count} bytes for {what} at {off}")

    need(0, len(MAGIC), "magic")
    if bytes(mv[: len(MAGIC)]) != MAGIC:
        raise VsrgError("bad starting delimiter")
    p = len(MAGIC)
    meta_start = p

    need(p, _FIXED_PREFIX + _SIZE_TABLE, ".meta header")
    version = int.from_bytes(mv[p : p + 2], "little")
    if version != VERSION:
        # Sections are positional; an unknown version cannot be skipped.
        raise VsrgError(f"unsupported version {version}")
    p += _FIXED_PREFIX  # version + reserved

    sizes = np.frombuffer(mv, dtype="<u4", count=5, offset=p)
    meta_size, res_size, chart_size, grid_size, notes_size = (int(x) for x in sizes)
    p += _SIZE_TABLE

    # .meta's declared size bounds its own string block.
    meta_end = meta_start + meta_size
    if meta_end < p or meta_end > n:
        raise VsrgError(f"bad .meta size {meta_size}")
    strings = _split_cstrings(bytes(mv[p:meta_end]), ".meta")
    if len(strings) != 4:
        raise VsrgError(f".meta: expected 4 strings, got {len(strings)}")
    title, artist, tags, song_file = strings
    p = meta_end

    need(p, res_size, ".resources")
    resources = _split_cstrings(bytes(mv[p : p + res_size]), ".resources")
    p += res_size

    need(p, chart_size, ".chart")
    if chart_size < 4:
        raise VsrgError(f".chart too small ({chart_size} bytes)")
    column_count, note_type, grid_pc, note_pc = (int(x) for x in mv[p : p + 4])
    p += chart_size

    gdt, ndt = grid_dtype(grid_pc), note_dtype(note_pc)
    if grid_size % gdt.itemsize:
        raise VsrgError(f".grid size {grid_size} is not a multiple of {gdt.itemsize}")
    if notes_size % ndt.itemsize:
        raise VsrgError(f".notes size {notes_size} is not a multiple of {ndt.itemsize}")

    need(p, grid_size, ".grid")
    grid = np.frombuffer(mv, dtype=gdt, count=grid_size // gdt.itemsize, offset=p)
    p += grid_size

    need(p, notes_size, ".notes")
    notes = np.frombuffer(mv, dtype=ndt, count=notes_size // ndt.itemsize, offset=p)
    p += notes_size

    # Landing on the sentinel validates the whole size table at once.
    need(p, len(END), "ending delimiter")
    if bytes(mv[p : p + len(END)]) != END:
        raise VsrgError("bad ending delimiter (section sizes disagree with the file)")

    return Vsrg(
        title=title,
        artist=artist,
        tags=tags,
        song_file=song_file,
        resources=resources,
        column_count=column_count,
        note_type=note_type,
        grid=grid,
        notes=notes,
        version=version,
        _buf=buf,
    )


def load(path, mmap: bool = False) -> Vsrg:
    """Read a chart. With ``mmap=True`` the arrays are views into the file."""
    if not mmap:
        with open(path, "rb") as f:
            return loads(f.read())
    with open(path, "rb") as f:
        mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
    v = loads(mm)
    v._buf = mm  # keep mapping alive for the lifetime of the views
    return v


# ── writing ─────────────────────────────────────────────────────────────────


def _cstrings(strings) -> bytes:
    out = bytearray()
    for s in strings:
        b = s.encode("utf-8")
        if b"\0" in b:
            raise VsrgError(f"embedded NUL in {s!r}")
        out += b + b"\0"
    return bytes(out)


def dumps(v: Vsrg) -> bytes:
    meta = bytearray()
    meta += VERSION.to_bytes(2, "little")
    meta += b"\0" * (_FIXED_PREFIX - 2)  # reserved, bytes 3-64
    sizes_at = len(meta)
    meta += b"\0" * _SIZE_TABLE  # patched below
    meta += _cstrings([v.title, v.artist, v.tags, v.song_file])

    res = _cstrings(v.resources)
    chart = bytes(
        [v.column_count, v.note_type, v.grid_param_count, v.note_param_count]
    )
    grid = np.ascontiguousarray(v.grid).tobytes()
    notes = np.ascontiguousarray(v.notes).tobytes()

    table = np.array(
        [len(meta), len(res), len(chart), len(grid), len(notes)], dtype="<u4"
    )
    meta[sizes_at : sizes_at + _SIZE_TABLE] = table.tobytes()

    return b"".join([MAGIC, bytes(meta), res, chart, grid, notes, END])


def dump(v: Vsrg, path) -> None:
    with open(path, "wb") as f:
        f.write(dumps(v))


# ── cli ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        v = load(path, mmap=True)
        sizes = v.chord_sizes()
        print(v)
        print(f"  holds       {int(v.is_hold.sum())}")
        print(f"  per column  {v.notes_per_column().tolist()}")
        print(f"  max chord   {int(sizes.max()) if len(sizes) else 0}")
        print(f"  resources   {v.resources}")
