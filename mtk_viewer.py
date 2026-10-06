"""MTK snapshot adapter using the Qualcomm GUI table renderer/conventions."""
from __future__ import annotations

import csv
import hashlib
import re
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable


# Cap for user-dragged column widths (in characters) so a drag cannot run away.
MAX_COLS_CHARS = 300
# Minimum seconds between live re-renders while dragging a column edge.
RESIZE_RENDER_INTERVAL = 0.07

# Curated text colors dark enough to stay readable on white / #F7F9FA backgrounds.
PALETTE: tuple[str, ...] = (
    "#B71C1C", "#D32F2F", "#C2185B", "#AD1457", "#880E4F",
    "#7B1FA2", "#9C27B0", "#6A1B9A", "#512DA8", "#4527A0",
    "#3949AB", "#303F9F", "#283593", "#1A237E", "#1565C0",
    "#0D47A1", "#0277BD", "#01579B", "#00838F", "#006064",
    "#00796B", "#00695C", "#004D40", "#2E7D32", "#1B5E20",
    "#33691E", "#827717", "#8D6E63", "#795548", "#6D4C41",
    "#5D4037", "#4E342E", "#5F6368", "#455A64", "#37474F",
    "#263238", "#BF360C", "#D84315", "#E64A19", "#8E24AA",
)

BAND_COLUMN_HEADERS: frozenset[str] = frozenset({
    "LTE DL", "LTE UL", "NR DL", "NR UL",
    "FR1 DL", "FR2 DL", "FR1 UL", "FR2 UL",
})

_BAND_RE = re.compile(r"^([Bn]\d+)[A-Z]?$")
_PLAIN_BAND_RE = re.compile(r"^(\d+)([A-Z])?$")


def _column_band_prefix(header: str) -> str:
    return "B" if "LTE" in header else "n"


def _band_spans(cell: str, header: str) -> list[tuple[int, int, str]]:
    """Return (start, end, canonical band) spans for band tokens joined by ' + '."""
    if header not in BAND_COLUMN_HEADERS:
        return []
    prefix = _column_band_prefix(header)
    spans: list[tuple[int, int, str]] = []
    pos = 0
    for token in cell.split(" + "):
        match = _BAND_RE.match(token)
        if match:
            canonical = match.group(1)
        else:
            match = _PLAIN_BAND_RE.match(token)
            canonical = f"{prefix}{match.group(1)}" if match else None
        if canonical:
            spans.append((pos, pos + len(token), canonical))
        pos += len(token) + 3
    return spans


def _band_sort_key(cell: str) -> tuple[int, Any]:
    """Sort key for band columns: numeric per band, so 8A sorts before 11A.

    Fully band-parsable cells map to (0, ((number, letter), ...)) so tuple
    comparison orders positionally by band number with the letter as final
    tie-break. Cells with any unparsable token fall back to (1, cell) so they
    group after bands and order among themselves by string.
    """
    pairs: list[tuple[int, str]] = []
    for token in cell.split(" + "):
        match = _PLAIN_BAND_RE.match(token)
        if not match:
            return (1, cell)
        pairs.append((int(match.group(1)), match.group(2) or ""))
    return (0, tuple(pairs))


def _column_sort_key(col: str) -> Callable[[dict[str, Any]], Any]:
    """Row sort key for a column: band-aware for band columns, numeric-else-string otherwise."""
    band_col = col in BAND_COLUMN_HEADERS

    def _key(r: dict[str, Any]) -> Any:
        v = str(r.get(col, ""))
        if band_col:
            return _band_sort_key(v)
        try:
            return (0, int(v))
        except ValueError:
            return (1, v)

    return _key


def _band_color(canonical: str) -> str:
    """Deterministically map a canonical band (e.g. B3, n78) to a palette color."""
    digest = int.from_bytes(hashlib.md5(canonical.encode("utf-8")).digest(), "big")
    return PALETTE[digest % len(PALETTE)]


def _clamped_col_width(pointer_char: int, offset: int, min_chars: int) -> int:
    """Clamp a dragged column width between min_chars and MAX_COLS_CHARS."""
    return min(MAX_COLS_CHARS, max(min_chars, pointer_char - offset))


FAMILIES = ("LTE", "NR SA (1CC)", "NR-CA", "EN-DC", "NRDC")


def class_label(value):
    return chr(65 + value) if 0 <= value < 26 else f"[{value}]"


def family_for(combo):
    if combo.lte:
        return "EN-DC" if combo.nr else "LTE"
    if not combo.nr:
        return None
    if any(c.band < 257 for c in combo.nr) and any(c.band >= 257 for c in combo.nr):
        return "NRDC"
    return "NR SA (1CC)" if combo.nr_physical_ccs == 1 else "NR-CA"


def _identity(combo):
    return (tuple((c.band, c.dl_class, c.ul_class, tuple(c.dl_mimo)) for c in combo.lte),
            tuple((c.band, c.dl_class, c.ul_class, tuple((cc.scs_khz, cc.dl_mimo, cc.dl_bw_mhz,
                   cc.ul_mimo, cc.ul_bw_mhz) for cc in c.ccs)) for c in combo.nr))


def _ordered(components, ul=False):
    # Match the Qualcomm presentation: descending numeric band/class order.
    return sorted((c for c in components if not ul or c.has_ul),
                  key=lambda c: (c.band, c.ul_class if ul else c.dl_class), reverse=True)


def _bands(components, ul=False):
    return " + ".join(f"{c.band}{class_label(c.ul_class if ul else c.dl_class)}"
                      for c in components)


def _values(values):
    return " + ".join("?" if value is None else str(value) for value in values)


def _nr_values(components, field, ul=False):
    # Keep physical CC feature order within each band, with only actual UL CCs
    # in the UL columns. None for a DL feature is unknown, not a default.
    return _values(getattr(cc, field) for c in components for cc in c.ccs
                   if not ul or cc.ul_mimo is not None)


def _nr_columns(dl, ul, prefix=""):
    return {
        f"{prefix}MIMO DL": _nr_values(dl, "dl_mimo"),
        f"{prefix}SCS DL (kHz)": _nr_values(dl, "scs_khz"),
        f"{prefix}BW DL (MHz)": _nr_values(dl, "dl_bw_mhz"),
        f"{prefix}MIMO UL": _nr_values(ul, "ul_mimo", ul=True),
        f"{prefix}SCS UL (kHz)": _nr_values(ul, "scs_khz", ul=True),
        f"{prefix}BW UL (MHz)": _nr_values(ul, "ul_bw_mhz", ul=True),
    }


def build_tables(lte_rows, nr_rows):
    """Project cached MTK features using Qualcomm headers and cell conventions.

    No LTE BW/SCS/UL MIMO, QAM, BCS or UL TX Switch columns: these are not
    extracted from the normalized capability banks. Exact feature variants
    remain separate; band reordering is presentation-only.
    """
    tables = {family: [] for family in FAMILIES}
    seen = set()
    for rows in (lte_rows, nr_rows):
        for combo in rows:
            family, key = family_for(combo), _identity(combo)
            if family is None or key in seen:
                continue
            seen.add(key)
            lte_dl, lte_ul = _ordered(combo.lte), _ordered(combo.lte, ul=True)
            nr_dl, nr_ul = _ordered(combo.nr), _ordered(combo.nr, ul=True)
            lte_mimo = _values(m for c in lte_dl for m in c.dl_mimo)
            if family == "LTE":
                row = {"LTE DL": _bands(lte_dl), "MIMO DL": lte_mimo,
                       "LTE UL": _bands(lte_ul, ul=True)}
            elif family == "EN-DC":
                features = _nr_columns(nr_dl, nr_ul, "NR ")
                row = {
                    "LTE DL": _bands(lte_dl), "LTE MIMO DL": lte_mimo,
                    "NR DL": _bands(nr_dl), "NR MIMO DL": features["NR MIMO DL"],
                    "NR SCS DL (kHz)": features["NR SCS DL (kHz)"],
                    "NR BW DL (MHz)": features["NR BW DL (MHz)"],
                    "LTE UL": _bands(lte_ul, ul=True), "NR UL": _bands(nr_ul, ul=True),
                    "NR MIMO UL": features["NR MIMO UL"],
                    "NR SCS UL (kHz)": features["NR SCS UL (kHz)"],
                    "NR BW UL (MHz)": features["NR BW UL (MHz)"],
                }
            elif family == "NRDC":
                row = {}
                groups = {fr: ([c for c in nr_dl if (c.band < 257) == (fr == "FR1")],
                               [c for c in nr_ul if (c.band < 257) == (fr == "FR1")])
                          for fr in ("FR1", "FR2")}
                # Same FR1 DL / FR2 DL / FR1 UL / FR2 UL column ordering.
                for direction in ("DL", "UL"):
                    for fr, (dl, ul) in groups.items():
                        components = ul if direction == "UL" else dl
                        features = _nr_columns(dl, ul, fr + " ")
                        row[f"{fr} {direction}"] = _bands(components, ul=direction == "UL")
                        for feature in ("MIMO", "SCS", "BW"):
                            name = f"{fr} {feature} {direction}"
                            name += " (kHz)" if feature == "SCS" else " (MHz)" if feature == "BW" else ""
                            row[name] = features[name]
            else:
                features = _nr_columns(nr_dl, nr_ul)
                row = {"NR DL": _bands(nr_dl)}
                row.update((name, value) for name, value in features.items() if " DL" in name)
                row["NR UL"] = _bands(nr_ul, ul=True)
                row.update((name, value) for name, value in features.items() if " UL" in name)
            tables[family].append(row)
    return tables


class ComboViewerWindow(tk.Toplevel):
    """Interactive pop-up window to view cached MTK capability combinations."""

    TAB_DEFINITIONS = (
        ("LTE", "lte_ca"),
        ("NR SA", "nr_sa"),
        ("NRCA", "nr_ca"),
        ("ENDC", "endc"),
        ("NRDC", "nrdc"),
    )

    FAMILY_KEYS = {"LTE": "lte_ca", "NR SA (1CC)": "nr_sa",
                   "NR-CA": "nr_ca", "EN-DC": "endc", "NRDC": "nrdc"}

    def current_family(self) -> str | None:
        return next((family for family, key in self.FAMILY_KEYS.items()
                     if key == self.active_tab_key), None)

    def select_family(self, family: str | None) -> None:
        key = self.FAMILY_KEYS.get(family, family)
        if key not in self.tabs_data:
            # Qualcomm only creates tabs with data; never invent an empty CA tab.
            self._on_tab_changed()
            return
        keys = [k for _, k in self.TAB_DEFINITIONS if k in self.tabs_data]
        self.notebook.select(keys.index(key))
        self._on_tab_changed()

    def s(self, px: int) -> int:
        """Scale pixel value by display scale factor."""
        return max(1, round(px * self.scale))

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        record: Any = None,
        *,
        data: dict[str, Any],
        initial_tab: str | None = None,
        scale: float = 1.0,
        title_suffix: str = "",
    ) -> None:
        super().__init__(parent)
        self.parent = parent
        self.scale = scale
        self.record = record
        self.data = data

        # Window configuration
        card_id = getattr(record, "display_name", "MTK Modem")
        self.title(f"MTK Combo Viewer - {card_id}{title_suffix}")
        win_w = self.s(1120)
        win_h = self.s(700)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(self.s(850), self.s(450))

        # Build table data
        tables = build_tables(data["lte"], data["nr"])
        self.raw_tables = {self.FAMILY_KEYS[family]: rows for family, rows in tables.items()}

        # Tab data tracking: tab_key -> {"columns": [...], "rows": [...], "text": Text, "sort_state": (col, reverse)}
        self.tabs_data: dict[str, dict[str, Any]] = {}
        self.active_tab_key: str | None = None
        self._resize_state: dict[str, Any] | None = None

        # Filter state
        self.search_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="")

        self._build_ui()
        self._populate_tabs()
        self.select_family(initial_tab)

        # Keyboard shortcuts
        self.bind("<Control-f>", lambda _: self.search_entry.focus_set())
        self.bind("<Escape>", lambda _: self.clear_search())
        self.bind("<Control-c>", lambda _: self.copy_selected())

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=self.s(10))
        outer.pack(fill="both", expand=True)

        # Header Info Banner
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, self.s(8)))

        info_parts = []
        if self.record:
            info_parts.append(f"Bank: {self.record.capability_bank_index} / Profile: {self.record.profile}")
            info_parts.append(f"Loader: {self.data['loader']}")
            companion = self.data.get("lte_bank_index")
            if companion is not None and companion != self.record.capability_bank_index:
                info_parts.append(f"LTE companion: bank {companion} / profile {self.record.profile}")
            info_parts.append(f"Path: {self.record.source}")
        info_text = "  |  ".join(info_parts) if info_parts else "MTK Capability Combination Viewer"
        info_label = ttk.Label(header, text=info_text, font=("TkDefaultFont", 9, "bold"))
        info_label.pack(side="left", fill="x", expand=True)

        # Search Bar Frame
        search_frame = ttk.Frame(outer)
        search_frame.pack(fill="x", pady=(0, self.s(8)))

        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, self.s(6)))
        self.search_entry = ttk.Entry(
            search_frame,
            textvariable=self.search_var,
            width=36,
        )
        self.search_entry.pack(side="left", padx=(0, self.s(4)))
        self.search_entry.focus_set()

        clear_btn = ttk.Button(
            search_frame,
            text="✕",
            width=3,
            command=self.clear_search,
        )
        clear_btn.pack(side="left", padx=(0, self.s(12)))

        self.show_scs_var = tk.BooleanVar(value=False)
        self.show_scs_check = ttk.Checkbutton(
            search_frame,
            text="Show SCS",
            variable=self.show_scs_var,
            command=self._on_scs_toggle,
        )
        self.show_scs_check.pack(side="left", padx=(0, self.s(14)))

        self.count_label = ttk.Label(
            search_frame,
            textvariable=self.count_var,
            foreground="#555555",
        )
        self.count_label.pack(side="left", padx=(0, self.s(12)))

        # Live search filtering trigger
        self.search_var.trace_add("write", lambda *_: self.apply_filter())

        # Notebook tabs
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _update_visible_columns(self, tbl_key: str) -> None:
        """Update visible columns for a tab based on SCS toggle, then re-render."""
        info = self.tabs_data.get(tbl_key)
        if not info:
            return
        self._layout_columns(tbl_key)
        self._render_table(tbl_key)

    def _layout_columns(self, tbl_key: str) -> None:
        """Compute visible columns, monospace character widths, and offsets for a tab."""
        info = self.tabs_data[tbl_key]
        columns = info["columns"]
        rows = info["rows"]
        show_scs = self.show_scs_var.get()

        visible = [c for c in columns if show_scs or "SCS" not in c]
        char_w = max(1, self.cell_font.measure("0"))
        pad_chars = max(2, round(self.s(16) / char_w))
        min_chars = max(4, round(self.s(45) / char_w))

        widths = []
        overrides = info.get("width_overrides", {})
        for col in visible:
            override = overrides.get(col)
            if override is not None:
                widths.append(max(1, int(override)))
                continue
            # Reserve room for the sort indicator (" ▲"/" ▼") like the old heading measure
            header_len = len(col) + 2
            content_len = max((len(str(r.get(col, ""))) for r in rows), default=0)
            widths.append(max(min_chars, header_len, content_len) + pad_chars)

        offsets = []
        pos = 0
        for width in widths:
            offsets.append(pos)
            pos += width

        info["visible_columns"] = visible
        info["widths"] = widths
        info["offsets"] = offsets
        info["min_chars"] = min_chars

    def _on_scs_toggle(self) -> None:
        """Handle Show SCS toggle across all tabs."""
        for tbl_key in self.tabs_data:
            self._update_visible_columns(tbl_key)

    def _populate_tabs(self) -> None:
        """Create tabs for available tables and populate them with combination data."""
        # Monospace font for exact column alignment (must use the real fixed family)
        fixed = tkfont.nametofont("TkFixedFont")
        self.cell_font = tkfont.Font(
            root=self,
            family=fixed.actual("family"),
            size=max(8, self.s(9)),
        )
        self.header_cell_font = tkfont.Font(
            root=self,
            family=self.cell_font.actual("family"),
            size=self.cell_font.actual("size"),
            weight="bold",
        )

        created_tabs = 0
        for tab_label, tbl_key in self.TAB_DEFINITIONS:
            rows = self.raw_tables.get(tbl_key, [])
            if not rows:
                continue

            columns = list(rows[0].keys())

            tab_frame = ttk.Frame(self.notebook, padding=self.s(2))
            tab_frame.rowconfigure(0, weight=1)
            tab_frame.columnconfigure(0, weight=1)

            text = tk.Text(
                tab_frame,
                font=self.cell_font,
                state="disabled",
                wrap="none",
                cursor="arrow",
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                selectbackground="#0078D7",
                selectforeground="#FFFFFF",
                spacing1=self.s(2),
                spacing3=self.s(2),
            )
            text.tag_configure("header", background="#E8ECF0", font=self.header_cell_font)
            text.tag_configure("oddrow", background="#F7F9FA")
            text.tag_configure("evenrow", background="#FFFFFF")
            text.tag_raise("sel")

            y_scroll = ttk.Scrollbar(tab_frame, orient="vertical", command=text.yview)
            x_scroll = ttk.Scrollbar(tab_frame, orient="horizontal", command=text.xview)
            text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

            text.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")

            # Click handling: header sorting, column resize, row selection (block Text class bindings)
            text.bind("<Button-1>", lambda e, tk_key=tbl_key: self._on_table_click(e, tk_key))
            text.bind("<B1-Motion>", lambda e, tk_key=tbl_key: self._on_table_drag_motion(e, tk_key))
            text.bind("<ButtonRelease-1>", lambda e, tk_key=tbl_key: self._on_table_release(e, tk_key))
            text.bind("<Motion>", lambda e, tk_key=tbl_key: self._on_table_hover(e, tk_key))
            text.bind("<Double-Button-1>", lambda _e: "break")
            text.bind("<Triple-Button-1>", lambda _e: "break")
            text.bind("<B1-Leave>", lambda _e: "break")
            text.bind("<Button-3>", lambda e, tk_key=tbl_key: self._show_context_menu(e, tk_key))

            self.tabs_data[tbl_key] = {
                "label": tab_label,
                "columns": columns,
                "rows": rows,
                "filtered_rows": list(rows),
                "text": text,
                "visible_columns": [],
                "widths": [],
                "offsets": [],
                "width_overrides": {},
                "band_tags": set(),
                "sort_col": None,
                "sort_reverse": False,
                "selected_lines": set(),
                "sel_anchor": None,
            }

            # Apply initial column visibility (hide SCS by default) and render
            self._update_visible_columns(tbl_key)

            self.notebook.add(tab_frame, text=f"{tab_label} ({len(rows)})")
            created_tabs += 1

        if created_tabs == 0:
            empty_frame = ttk.Frame(self.notebook, padding=self.s(20))
            ttk.Label(
                empty_frame,
                text="No combinations found for this capability profile.",
                font=("TkDefaultFont", 11),
            ).pack(expand=True)
            self.notebook.add(empty_frame, text="Empty")
            self.count_var.set("0 combos")

    def _render_table(self, tbl_key: str, keep_selection: bool = False) -> None:
        """Render header and filtered rows of a tab into its Text widget."""
        info = self.tabs_data[tbl_key]
        text: tk.Text = info["text"]
        columns = info["visible_columns"]
        widths = info["widths"]
        rows = info["filtered_rows"]
        band_tags: set[str] = info["band_tags"]

        saved_lines = set(info["selected_lines"]) if keep_selection else None
        saved_anchor = info["sel_anchor"] if keep_selection else None
        if not keep_selection:
            info["selected_lines"] = set()
            info["sel_anchor"] = None

        arrow = " ▼" if info["sort_reverse"] else " ▲"
        header_cells = []
        for col, width in zip(columns, widths):
            label = f"{col}{arrow}" if col == info["sort_col"] else col
            header_cells.append(label.center(width))

        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("end", "".join(header_cells) + "\n")
        text.tag_add("header", "1.0", "2.0")

        for idx, row in enumerate(rows):
            line_no = idx + 2
            cells = []
            for col, width in zip(columns, widths):
                cells.append(str(row.get(col, "")).center(width))
            text.insert("end", "".join(cells) + "\n")
            zebra = "evenrow" if idx % 2 == 0 else "oddrow"
            text.tag_add(zebra, f"{line_no}.0", f"{line_no + 1}.0")

            # Color band tokens wherever they appear (any column)
            base = 0
            for col, width in zip(columns, widths):
                cell = str(row.get(col, ""))
                marg = width - len(cell)
                left_pad = marg // 2 + (marg & width & 1) if marg > 0 else 0
                for start, end, canonical in _band_spans(cell, col):
                    tag = f"band_{canonical}"
                    if tag not in band_tags:
                        text.tag_configure(tag, foreground=_band_color(canonical))
                        band_tags.add(tag)
                    text.tag_add(
                        tag,
                        f"{line_no}.{base + left_pad + start}",
                        f"{line_no}.{base + left_pad + end}",
                    )
                base += width

        if keep_selection:
            info["selected_lines"] = saved_lines
            info["sel_anchor"] = saved_anchor
            self._apply_selection(tbl_key)

        text.tag_raise("sel")
        text.configure(state="disabled")

    def _on_tab_changed(self, _event: Any = None) -> None:
        selected_id = self.notebook.select()
        if not selected_id:
            return
        selected_idx = self.notebook.index(selected_id)
        active_keys = [k for _, k in self.TAB_DEFINITIONS if k in self.tabs_data]
        if 0 <= selected_idx < len(active_keys):
            self.active_tab_key = active_keys[selected_idx]
            self.apply_filter()

    def clear_search(self) -> None:
        self.search_var.set("")
        self.search_entry.focus_set()

    def apply_filter(self) -> None:
        """Filter the active tab based on the search query."""
        if not self.active_tab_key or self.active_tab_key not in self.tabs_data:
            return

        info = self.tabs_data[self.active_tab_key]
        raw_rows = info["rows"]
        query = self.search_var.get().strip().casefold()

        if not query:
            filtered = list(raw_rows)
        else:
            query_nospace = query.replace(" ", "")
            filtered = []
            for row in raw_rows:
                row_text = " ".join(str(v) for v in row.values()).casefold()
                row_nospace = row_text.replace(" ", "")
                if query in row_text or query_nospace in row_nospace:
                    filtered.append(row)

        info["filtered_rows"] = filtered

        # If a sort was active, re-apply sort
        sort_col = info["sort_col"]
        if sort_col:
            reverse = info["sort_reverse"]
            filtered.sort(key=_column_sort_key(sort_col), reverse=reverse)

        self._render_table(self.active_tab_key)

        total = len(raw_rows)
        shown = len(filtered)
        if query:
            self.count_var.set(f"Showing {shown:,} of {total:,} combos")
        else:
            self.count_var.set(f"Total: {total:,} combos")

    def sort_column(self, tbl_key: str, col: str) -> None:
        """Sort tab rows by the clicked column."""
        info = self.tabs_data.get(tbl_key)
        if not info:
            return

        if info["sort_col"] == col:
            info["sort_reverse"] = not info["sort_reverse"]
        else:
            info["sort_col"] = col
            info["sort_reverse"] = False

        reverse = info["sort_reverse"]

        filtered = info["filtered_rows"]
        filtered.sort(key=_column_sort_key(col), reverse=reverse)
        self._render_table(tbl_key)

    def _header_column_at(self, info: dict[str, Any], char: int) -> int | None:
        """Map a character offset on the header line to a visible column index."""
        for idx, (start, width) in enumerate(zip(info["offsets"], info["widths"])):
            if start <= char < start + width:
                return idx
        return None

    def _apply_selection(self, tbl_key: str) -> None:
        """Mirror the tracked selected body lines onto the Text 'sel' tag."""
        info = self.tabs_data[tbl_key]
        text: tk.Text = info["text"]
        text.tag_remove("sel", "1.0", "end")
        for line in info["selected_lines"]:
            text.tag_add("sel", f"{line}.0", f"{line + 1}.0")

    def _resize_handle_at(self, info: dict[str, Any], char: int) -> int | None:
        """Return the index of the column whose right header edge is under the pointer."""
        tol = max(2, self.s(2))
        best = None
        best_dist = tol + 1
        for idx, (start, width) in enumerate(zip(info["offsets"], info["widths"])):
            dist = abs(char - (start + width))
            if dist <= tol and dist < best_dist:
                best = idx
                best_dist = dist
        return best

    def _begin_column_resize(self, tbl_key: str, col_idx: int) -> None:
        """Start dragging the right edge of a header column."""
        info = self.tabs_data[tbl_key]
        self._resize_state = {
            "tbl_key": tbl_key,
            "col_idx": col_idx,
            "col_name": info["visible_columns"][col_idx],
            "start_width": info["widths"][col_idx],
            "last_render": 0.0,
        }
        info["text"].configure(cursor="sb_h_double_arrow")

    def _render_resized(self, tbl_key: str) -> None:
        """Re-layout and re-render a tab after a width change, keeping selection."""
        self._layout_columns(tbl_key)
        self._render_table(tbl_key, keep_selection=True)

    def _on_table_drag_motion(self, event: Any, tbl_key: str) -> str:
        """Resize the dragged column on B1-Motion with a throttled live re-render."""
        state = self._resize_state
        if state is None or state["tbl_key"] != tbl_key:
            return "break"
        info = self.tabs_data[tbl_key]
        pointer_char = int(info["text"].index(f"@{event.x},{event.y}").split(".")[1])
        info["width_overrides"][state["col_name"]] = _clamped_col_width(
            pointer_char, info["offsets"][state["col_idx"]], info["min_chars"],
        )
        now = time.monotonic()
        if now - state["last_render"] >= RESIZE_RENDER_INTERVAL:
            state["last_render"] = now
            self._render_resized(tbl_key)
        return "break"

    def _on_table_release(self, event: Any, tbl_key: str) -> str:
        """Finish a column resize with a final re-render."""
        state = self._resize_state
        if state is None or state["tbl_key"] != tbl_key:
            return "break"
        self._resize_state = None
        info = self.tabs_data[tbl_key]
        info["text"].configure(cursor="arrow")
        if info["width_overrides"].get(state["col_name"]) != state["start_width"]:
            self._render_resized(tbl_key)
        return "break"

    def _on_table_hover(self, event: Any, tbl_key: str) -> str:
        """Show a resize cursor when hovering a header column boundary."""
        if self._resize_state is not None:
            return "break"
        info = self.tabs_data.get(tbl_key)
        if not info:
            return "break"
        text: tk.Text = info["text"]
        line_s, char_s = text.index(f"@{event.x},{event.y}").split(".")
        cursor = "arrow"
        if line_s == "1" and self._resize_handle_at(info, int(char_s)) is not None:
            cursor = "sb_h_double_arrow"
        if str(text.cget("cursor")) != cursor:
            text.configure(cursor=cursor)
        return "break"

    def _on_table_click(self, event: Any, tbl_key: str) -> str:
        """Handle clicks on a tab's Text widget (header sorting and row selection)."""
        info = self.tabs_data[tbl_key]
        text: tk.Text = info["text"]
        line_s, char_s = text.index(f"@{event.x},{event.y}").split(".")
        line = int(line_s)

        if line == 1:
            col_idx = self._resize_handle_at(info, int(char_s))
            if col_idx is not None:
                self._begin_column_resize(tbl_key, col_idx)
                return "break"
            col_idx = self._header_column_at(info, int(char_s))
            if col_idx is not None:
                self.sort_column(tbl_key, info["visible_columns"][col_idx])
            return "break"

        if not 2 <= line <= 1 + len(info["filtered_rows"]):
            return "break"

        if event.state & 0x0004:  # Control: toggle row under cursor
            if line in info["selected_lines"]:
                info["selected_lines"].discard(line)
            else:
                info["selected_lines"].add(line)
            info["sel_anchor"] = line
        elif event.state & 0x0001:  # Shift: extend from anchor row
            anchor = info["sel_anchor"] or (min(info["selected_lines"]) if info["selected_lines"] else line)
            info["selected_lines"].update(range(min(anchor, line), max(anchor, line) + 1))
            info["sel_anchor"] = line
        else:
            info["selected_lines"] = {line}
            info["sel_anchor"] = line
        self._apply_selection(tbl_key)
        return "break"

    def _show_context_menu(self, event: Any, tbl_key: str) -> None:
        info = self.tabs_data[tbl_key]
        text: tk.Text = info["text"]
        line = int(text.index(f"@{event.x},{event.y}").split(".")[0])
        if 2 <= line <= 1 + len(info["filtered_rows"]) and line not in info["selected_lines"]:
            info["selected_lines"] = {line}
            info["sel_anchor"] = line
            self._apply_selection(tbl_key)

        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Copy Selected Row(s)", command=self.copy_selected)
        menu.add_command(label="Copy Carrier Combo Only", command=self.copy_combo_only)
        menu.add_separator()
        menu.add_command(label="Copy All Filtered Rows", command=self.copy_all_visible)
        menu.tk_popup(event.x_root, event.y_root)

    def copy_selected(self) -> None:
        if not self.active_tab_key or self.active_tab_key not in self.tabs_data:
            return
        info = self.tabs_data[self.active_tab_key]
        if not info["selected_lines"]:
            return

        columns = info["columns"]
        lines = ["\t".join(columns)]
        for line in sorted(info["selected_lines"]):
            row = info["filtered_rows"][line - 2]
            lines.append("\t".join(str(row.get(c, "")) for c in columns))

        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)

    def copy_combo_only(self) -> None:
        """Copy just the primary DL combo string of selected rows."""
        if not self.active_tab_key or self.active_tab_key not in self.tabs_data:
            return
        info = self.tabs_data[self.active_tab_key]
        if not info["selected_lines"]:
            return

        # Find DL column name
        columns = info["columns"]
        dl_col_idx = 0
        for idx, col in enumerate(columns):
            if "DL" in col:
                dl_col_idx = idx
                break

        combos = []
        for line in sorted(info["selected_lines"]):
            row = info["filtered_rows"][line - 2]
            value = row.get(columns[dl_col_idx], "")
            combos.append(str(value))

        text = "\n".join(combos)
        self.clipboard_clear()
        self.clipboard_append(text)

    def copy_all_visible(self) -> None:
        if not self.active_tab_key or self.active_tab_key not in self.tabs_data:
            return
        info = self.tabs_data[self.active_tab_key]
        columns = info["columns"]
        rows = info["filtered_rows"]

        lines = ["\t".join(columns)]
        for r in rows:
            lines.append("\t".join(str(r.get(c, "")) for c in columns))

        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)

    def export_current_tab_csv(self) -> None:
        """Export the visible rows of the current tab to a CSV file."""
        if not self.active_tab_key or self.active_tab_key not in self.tabs_data:
            return
        info = self.tabs_data[self.active_tab_key]
        label = info["label"]
        rows = info["filtered_rows"]
        if not rows:
            messagebox.showinfo("Export CSV", "No rows to export.")
            return

        default_name = f"{label.lower()}_combos.csv"
        if self.record and getattr(self.record, "identity", None):
            default_name = f"rf_config_{self.record.identity}_{label.lower()}.csv"

        path_str = filedialog.asksaveasfilename(
            parent=self,
            title=f"Export {label} Combinations to CSV",
            initialfile=default_name,
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path_str:
            return

        target_path = Path(path_str)
        try:
            with target_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=info["columns"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            messagebox.showinfo("Export CSV", f"Successfully exported {len(rows):,} rows to:\n{target_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not write CSV:\n{e}")
