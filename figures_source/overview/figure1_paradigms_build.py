"""Build a richly-illustrated Figure 1 SVG.

Three stacked panels:
  Top  : Discrete diffusion prior (masked → blob → complete → score → evaluated stack)
  (A)  : Feedback for selection with frozen-backbone search (search tree, generator fixed)
  (B)  : Feedback for adaptation with test-time fine-tuning (prior shifts after update)
"""

from textwrap import dedent

# Token palette (matching the reference image)
TOKEN_COLORS = {
    "B": "#A8C5E0",   # light
    "P": "#5E92C4",   # medium
    "G": "#C9DCEE",   # very light
    "R": "#2A6FB8",   # deep
    "C": "#7FAAD0",   # medium-light
}

TOKEN_W = 22
TOKEN_GAP = 3


def token_square(x, y, color_key, masked=False):
    """A single token square. If masked, draws an outlined '?' square."""
    if masked:
        border = "#B0B0B0"
        return (
            f'<rect x="{x}" y="{y}" width="{TOKEN_W}" height="{TOKEN_W}" rx="3" ry="3" '
            f'fill="#FFFFFF" stroke="{border}" stroke-width="1.2"/>'
            f'<text x="{x + TOKEN_W/2}" y="{y + TOKEN_W/2 + 4.5}" text-anchor="middle" '
            f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" fill="#777777">?</text>'
        )
    fill = TOKEN_COLORS[color_key]
    return (
        f'<rect x="{x}" y="{y}" width="{TOKEN_W}" height="{TOKEN_W}" rx="3" ry="3" '
        f'fill="{fill}" stroke="#666666" stroke-width="0.7"/>'
    )


def token_row(x, y, pattern):
    """A row of tokens. pattern is a list of (color_key, masked) tuples."""
    parts = []
    cur_x = x
    for color_key, masked in pattern:
        parts.append(token_square(cur_x, y, color_key, masked))
        cur_x += TOKEN_W + TOKEN_GAP
    return "\n".join(parts), cur_x  # return total used width


def score_box(x, y, value, kind="oracle"):
    """A small score-pill: green-ish for oracle, grey for likelihood."""
    if kind == "oracle":
        fill, border, text_color = "#E8EFF7", "#9FB6CD", "#2A4A6F"
    else:
        fill, border, text_color = "#F0F0F0", "#A0A0A0", "#333333"
    w, h = 56, 22
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" ry="3" '
        f'fill="{fill}" stroke="{border}" stroke-width="0.9"/>'
        f'<text x="{x + w/2}" y="{y + h/2 + 4}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" fill="{text_color}">{value}</text>'
    )


def density_blob(cx, cy, scale=1.0, palette="blue"):
    """Multi-layer peanut/ellipse density blob."""
    if palette == "blue":
        layers = [
            (1.00, "#E2EEF9"),
            (0.82, "#BFDBF0"),
            (0.62, "#8EBFE2"),
            (0.42, "#5599D0"),
            (0.25, "#2A6FB8"),
        ]
    elif palette == "pink":
        layers = [
            (1.00, "#FBE3E8"),
            (0.82, "#F2C2CC"),
            (0.62, "#E891A4"),
            (0.42, "#D3617A"),
            (0.25, "#B23A55"),
        ]
    out = []
    # Peanut shape: two slightly-offset ellipses forming each contour layer
    for s, color in layers:
        rx_outer = 75 * s * scale
        ry_outer = 38 * s * scale
        out.append(
            f'<ellipse cx="{cx - 28 * s * scale}" cy="{cy}" '
            f'rx="{rx_outer}" ry="{ry_outer}" fill="{color}"/>'
        )
        out.append(
            f'<ellipse cx="{cx + 28 * s * scale}" cy="{cy}" '
            f'rx="{rx_outer}" ry="{ry_outer}" fill="{color}"/>'
        )
    return "\n".join(out)


def density_blob_shifted(cx, cy, scale=1.0, palette="deep_blue"):
    """A 'shifted' blob with a different shape: bimodal peaks more separated."""
    if palette == "deep_blue":
        layers = [
            (1.00, "#B5CEE5"),
            (0.82, "#7FA8CE"),
            (0.62, "#4F82B5"),
            (0.42, "#2A609A"),
            (0.25, "#0F4178"),
        ]
    else:
        layers = [
            (1.00, "#E2EEF9"),
            (0.82, "#BFDBF0"),
            (0.62, "#8EBFE2"),
            (0.42, "#5599D0"),
            (0.25, "#2A6FB8"),
        ]
    out = []
    for s, color in layers:
        rx_outer = 65 * s * scale
        ry_outer = 36 * s * scale
        out.append(
            f'<ellipse cx="{cx - 42 * s * scale}" cy="{cy + 4 * s * scale}" '
            f'rx="{rx_outer}" ry="{ry_outer}" fill="{color}"/>'
        )
        out.append(
            f'<ellipse cx="{cx + 42 * s * scale}" cy="{cy - 4 * s * scale}" '
            f'rx="{rx_outer * 0.95}" ry="{ry_outer * 0.95}" fill="{color}"/>'
        )
    return "\n".join(out)


def stacked_samples(x, y, rows, dx=-6, dy=6):
    """A short stack of evaluated-sample rows. The first (foreground) row is drawn last."""
    out = []
    n_back = 2  # number of background offset copies
    # background copies (just outlined boxes)
    full_row_w = (TOKEN_W + TOKEN_GAP) * 5 - TOKEN_GAP + 60 + 8  # approximate
    for k in range(n_back, 0, -1):
        ox, oy = x + dx * k, y + dy * k
        out.append(
            f'<rect x="{ox - 6}" y="{oy - 6}" width="{full_row_w + 12}" '
            f'height="{(len(rows) * 28) + 12}" rx="4" ry="4" '
            f'fill="#FFFFFF" stroke="#999999" stroke-width="0.8"/>'
        )
    # foreground frame
    out.append(
        f'<rect x="{x - 6}" y="{y - 6}" width="{full_row_w + 12}" '
        f'height="{(len(rows) * 28) + 12}" rx="4" ry="4" '
        f'fill="#FFFFFF" stroke="#222222" stroke-width="0.9"/>'
    )
    # rows
    cur_y = y
    for tokens, oracle, ll in rows:
        row_svg, end_x = token_row(x, cur_y, tokens)
        out.append(row_svg)
        if oracle is not None:
            out.append(score_box(end_x + 6, cur_y - 1, oracle, "oracle"))
        if ll is not None:
            out.append(score_box(end_x + 6 + 60, cur_y - 1, ll, "ll"))
        cur_y += 28
    # ellipsis
    out.append(
        f'<text x="{x + (full_row_w / 2) - 6}" y="{cur_y + 6}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="14" fill="#888888">...</text>'
    )
    return "\n".join(out), full_row_w


def arrow(x1, y1, x2, y2, marker="ah"):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#222222" stroke-width="1.0" marker-end="url(#{marker})"/>'
    )


# ============================================================
# Build the figure
# ============================================================

W, H = 1500, 1080

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
    '''<defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#222222"/>
        </marker>
        <marker id="ah-pink" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#C2185B"/>
        </marker>
        <marker id="ah-blue" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#3A7FC8"/>
        </marker>
      </defs>''',
    f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
]

# ============================================================
# PANEL 0: Discrete diffusion prior
# ============================================================

PANEL0_Y = 30
PANEL0_H = 290
parts.append(
    f'<text x="40" y="{PANEL0_Y + 30}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="18" font-weight="700" fill="#111111">Discrete diffusion prior</text>'
)

# x_T (noise) — fully masked
xT_x = 80
xT_y = PANEL0_Y + 110
parts.append(
    f'<text x="{xT_x + 60}" y="{xT_y - 20}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" fill="#111111">'
    f'noise (fully masked)</text>'
)
masked_pattern = [("B", True), ("P", True), ("G", True), ("R", True), ("C", True)]
row_svg, _ = token_row(xT_x, xT_y, masked_pattern)
parts.append(row_svg)

# arrow
parts.append(arrow(xT_x + 130, xT_y + TOKEN_W / 2, xT_x + 175, xT_y + TOKEN_W / 2))

# Pretrained prior pθ(x) — blob
blob_cx = 380
blob_cy = xT_y + TOKEN_W / 2
parts.append(
    f'<text x="{blob_cx}" y="{blob_cy - 70}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" fill="#111111">'
    f'Pretrained prior <tspan font-family="Times New Roman, Times, serif" '
    f'font-style="italic">p&#952;(x)</tspan></text>'
)
parts.append(density_blob(blob_cx, blob_cy, scale=1.0, palette="blue"))

# arrow
parts.append(arrow(blob_cx + 110, blob_cy, blob_cx + 155, blob_cy))

# x_0 (complete)
x0_x = 555
x0_y = xT_y
parts.append(
    f'<text x="{x0_x + 60}" y="{x0_y - 18}" text-anchor="middle" '
    f'font-style="italic" font-size="14" fill="#111111">'
    f'x₀ (complete)</text>'
)
complete_pattern = [("B", False), ("P", False), ("G", False), ("R", False), ("C", False)]
row_svg, _ = token_row(x0_x, x0_y, complete_pattern)
parts.append(row_svg)

# arrow
parts.append(arrow(x0_x + 130, x0_y + TOKEN_W / 2, x0_x + 175, x0_y + TOKEN_W / 2))

# Scoring box
scoring_x = 745
scoring_y = PANEL0_Y + 70
scoring_w = 200
scoring_h = 130
parts.append(
    f'<rect x="{scoring_x}" y="{scoring_y}" width="{scoring_w}" height="{scoring_h}" '
    f'rx="6" ry="6" fill="#FFFFFF" stroke="#777777" stroke-width="0.9"/>'
)
parts.append(
    f'<text x="{scoring_x + scoring_w/2}" y="{scoring_y + 22}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" font-weight="600" fill="#111111">'
    f'Scoring</text>'
)
# oracle r(x)
parts.append(
    f'<text x="{scoring_x + 18}" y="{scoring_y + 56}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" fill="#222222">oracle '
    f'<tspan font-style="italic">r(x)</tspan></text>'
)
parts.append(score_box(scoring_x + 110, scoring_y + 42, "—", "oracle"))
# divider
parts.append(
    f'<line x1="{scoring_x + 18}" y1="{scoring_y + 75}" '
    f'x2="{scoring_x + scoring_w - 18}" y2="{scoring_y + 75}" '
    f'stroke="#BBBBBB" stroke-width="0.8" stroke-dasharray="3,3"/>'
)
# log p_theta(x)
parts.append(
    f'<text x="{scoring_x + 18}" y="{scoring_y + 105}" '
    f'font-style="italic" font-size="12" fill="#222222">'
    f'log p&#952;(x)</text>'
)
parts.append(score_box(scoring_x + 110, scoring_y + 91, "—", "ll"))

# arrow
parts.append(arrow(scoring_x + scoring_w + 5, blob_cy, scoring_x + scoring_w + 50, blob_cy))

# Evaluated samples stack
es_x = 1010
es_y = PANEL0_Y + 75
parts.append(
    f'<text x="{es_x + 130}" y="{es_y - 20}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" font-weight="600" fill="#111111">'
    f'Evaluated samples</text>'
)
samples_rows = [
    ([("B", False), ("P", False), ("G", False), ("R", False), ("C", False)], "—", "—"),
    ([("B", False), ("G", False), ("P", False), ("R", False), ("C", False)], "—", "—"),
]
stack_svg, stack_w = stacked_samples(es_x, es_y + 8, samples_rows)
parts.append(stack_svg)


# ============================================================
# PANEL A: Frozen-backbone search
# ============================================================

PANEL_A_Y = 350
PANEL_A_H = 320
parts.append(
    f'<text x="40" y="{PANEL_A_Y + 30}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="17" font-weight="700" fill="#111111">'
    f'Feedback for selection with frozen-backbone search</text>'
)
parts.append(
    f'<text x="40" y="{PANEL_A_Y + 50}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="11" font-style="italic" fill="#666666">'
    f'intermediate scores via surrogate or rollout</text>'
)

# Density blob on left, generator fixed
A_blob_cx = 145
A_blob_cy = PANEL_A_Y + 165
parts.append(
    f'<text x="{A_blob_cx}" y="{A_blob_cy - 70}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" fill="#111111">'
    f'Pretrained prior <tspan font-family="Times New Roman, Times, serif" '
    f'font-style="italic">p&#952;(x)</tspan></text>'
)
parts.append(density_blob(A_blob_cx, A_blob_cy, scale=0.9, palette="blue"))
parts.append(
    f'<text x="{A_blob_cx}" y="{A_blob_cy + 80}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" font-style="italic" fill="#666666">'
    f'generator fixed</text>'
)

# Search tree: 3 chains, each filling more tokens over t1, t2, t3
# Column x positions for the three timesteps
COL_T1 = 360
COL_T2 = 600
COL_T3 = 840
COL_FINAL = 1100
COL_SEL = 1320

# Column headers
for col_x, label in [(COL_T1, "partial state t₁"), (COL_T2, "partial state t₂"),
                     (COL_T3, "partial state t₃")]:
    parts.append(
        f'<text x="{col_x + 60}" y="{PANEL_A_Y + 80}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" font-weight="600" fill="#111111">'
        f'{label}</text>'
    )

parts.append(
    f'<text x="{COL_FINAL + 60}" y="{PANEL_A_Y + 80}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" font-weight="600" fill="#111111">'
    f'Complete sequences</text>'
)
parts.append(
    f'<text x="{COL_SEL}" y="{PANEL_A_Y + 80}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" font-weight="600" fill="#111111">'
    f'Oracle <tspan font-style="italic">r(x)</tspan></text>'
)

# Three rows for the search chains
ROW_Y = [PANEL_A_Y + 110, PANEL_A_Y + 165, PANEL_A_Y + 220]

# Branch out from prior to t1
for i, ry in enumerate(ROW_Y):
    parts.append(
        f'<path d="M {A_blob_cx + 80} {A_blob_cy} '
        f'C {A_blob_cx + 150} {A_blob_cy}, '
        f'{COL_T1 - 30} {ry + 11}, '
        f'{COL_T1 - 5} {ry + 11}" '
        f'fill="none" stroke="#222222" stroke-width="0.9" marker-end="url(#ah)"/>'
    )

# t1 column: each row has 1 filled, 4 masked, plus a small score box
patterns_t1 = [
    [("B", False), ("P", True), ("G", True), ("R", True), ("C", True)],
    [("B", False), ("G", True), ("P", True), ("R", True), ("C", True)],
    [("P", False), ("B", True), ("G", True), ("R", True), ("C", True)],
]
scores_t1 = ["—", "—", "—"]
for ry, pat, sc in zip(ROW_Y, patterns_t1, scores_t1):
    row_svg, end_x = token_row(COL_T1, ry, pat)
    parts.append(row_svg)
    # small grey score pill
    parts.append(
        f'<rect x="{end_x + 4}" y="{ry - 1}" width="44" height="22" rx="3" ry="3" '
        f'fill="#EEF3F9" stroke="#5C8DBC" stroke-width="0.9" stroke-dasharray="3,2"/>'
        f'<text x="{end_x + 26}" y="{ry + 14}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#2A4A6F">{sc}</text>'
    )

# t1 → t2 arrows (top two rows continue, bottom row continues with prune later)
for i, ry in enumerate(ROW_Y):
    parts.append(
        f'<path d="M {COL_T1 + 175} {ry + 11} '
        f'L {COL_T2 - 5} {ry + 11}" '
        f'fill="none" stroke="#222222" stroke-width="0.9" marker-end="url(#ah)"/>'
    )

# t2 column: 2 filled, 3 masked
patterns_t2 = [
    [("B", False), ("P", False), ("G", True), ("R", True), ("C", True)],
    [("B", False), ("G", False), ("P", True), ("R", True), ("C", True)],
    [("P", False), ("B", False), ("G", True), ("R", True), ("C", True)],  # this one will be pruned
]
scores_t2 = ["—", "—", None]  # last one pruned
for i, (ry, pat, sc) in enumerate(zip(ROW_Y, patterns_t2, scores_t2)):
    if i == 2:
        # this branch is pruned at t2 — show an X instead
        cx = COL_T2 + 65
        cy = ry + 11
        parts.append(
            f'<g stroke="#888888" stroke-width="2.4" fill="none" stroke-linecap="round">'
            f'<line x1="{cx - 10}" y1="{cy - 10}" x2="{cx + 10}" y2="{cy + 10}"/>'
            f'<line x1="{cx + 10}" y1="{cy - 10}" x2="{cx - 10}" y2="{cy + 10}"/>'
            f'</g>'
        )
        # also a faint partial state behind the X
        row_svg, end_x = token_row(COL_T2, ry, [("P", False), ("B", False), ("G", True), ("R", True), ("C", True)])
        # not adding to keep clean
        continue
    row_svg, end_x = token_row(COL_T2, ry, pat)
    parts.append(row_svg)
    parts.append(
        f'<rect x="{end_x + 4}" y="{ry - 1}" width="44" height="22" rx="3" ry="3" '
        f'fill="#EEF3F9" stroke="#5C8DBC" stroke-width="0.9" stroke-dasharray="3,2"/>'
        f'<text x="{end_x + 26}" y="{ry + 14}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#2A4A6F">{sc}</text>'
    )

# t2 → t3 arrows (only first two rows; row 3 pruned)
for i in [0, 1]:
    ry = ROW_Y[i]
    parts.append(
        f'<path d="M {COL_T2 + 175} {ry + 11} '
        f'L {COL_T3 - 5} {ry + 11}" '
        f'fill="none" stroke="#222222" stroke-width="0.9" marker-end="url(#ah)"/>'
    )

# t3 column: 4 filled, 1 masked
patterns_t3 = [
    [("B", False), ("P", False), ("G", False), ("R", False), ("C", True)],
    [("B", False), ("G", False), ("P", False), ("R", False), ("C", True)],
]
scores_t3 = ["—", "—"]
for i, ry, pat, sc in zip(range(2), ROW_Y[:2], patterns_t3, scores_t3):
    row_svg, end_x = token_row(COL_T3, ry, pat)
    parts.append(row_svg)
    parts.append(
        f'<rect x="{end_x + 4}" y="{ry - 1}" width="44" height="22" rx="3" ry="3" '
        f'fill="#EEF3F9" stroke="#5C8DBC" stroke-width="0.9" stroke-dasharray="3,2"/>'
        f'<text x="{end_x + 26}" y="{ry + 14}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#2A4A6F">{sc}</text>'
    )

# Show row-3 pruned at t3 too (alternative: show one path got pruned at t3, second one at t2)
# Make it more visually balanced — show the pruning happens between t3 and complete
ry3 = ROW_Y[2]
cx = COL_T3 + 65
cy = ry3 + 11
# leave row 3 area with another X to show the pruning continues
parts.append(
    f'<g stroke="#888888" stroke-width="2.4" fill="none" stroke-linecap="round">'
    f'<line x1="{cx - 10}" y1="{cy - 10}" x2="{cx + 10}" y2="{cy + 10}"/>'
    f'<line x1="{cx + 10}" y1="{cy - 10}" x2="{cx - 10}" y2="{cy + 10}"/>'
    f'</g>'
)

# t3 → final complete arrows (top two rows)
for i in [0, 1]:
    ry = ROW_Y[i]
    parts.append(
        f'<path d="M {COL_T3 + 175} {ry + 11} '
        f'L {COL_FINAL - 5} {ry + 11}" '
        f'fill="none" stroke="#222222" stroke-width="0.9" marker-end="url(#ah)"/>'
    )

# Final complete sequences (5 filled)
patterns_final = [
    [("B", False), ("P", False), ("G", False), ("R", False), ("C", False)],
    [("B", False), ("G", False), ("P", False), ("R", False), ("C", False)],
]
oracle_finals = ["—", "—"]
for i, (ry, pat, sc) in enumerate(zip(ROW_Y[:2], patterns_final, oracle_finals)):
    row_svg, end_x = token_row(COL_FINAL, ry, pat)
    parts.append(row_svg)
    parts.append(score_box(end_x + 8, ry - 1, sc, "oracle"))

# "Select" bracket on the right
sel_x = COL_SEL - 25
sel_top = ROW_Y[0] - 5
sel_bot = ROW_Y[1] + TOKEN_W + 5
parts.append(
    f'<path d="M {sel_x + 10} {sel_top} '
    f'Q {sel_x} {sel_top} {sel_x} {sel_top + 10} '
    f'L {sel_x} {sel_bot - 10} '
    f'Q {sel_x} {sel_bot} {sel_x + 10} {sel_bot}" '
    f'fill="none" stroke="#222222" stroke-width="1.0"/>'
)
parts.append(
    f'<text x="{sel_x + 18}" y="{(sel_top + sel_bot) / 2 + 5}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="14" font-weight="600" '
    f'fill="#111111">Select</text>'
)



# --- Legend under the search tree ---
LEG_Y = PANEL_A_Y + 268
LEG_X = COL_T1 - 30
# dashed pale-blue pill = surrogate (top row)
parts.append(
    f'<rect x="{LEG_X}" y="{LEG_Y}" width="34" height="18" rx="3" ry="3" '
    f'fill="#EEF3F9" stroke="#5C8DBC" stroke-width="0.9" stroke-dasharray="3,2"/>'
)
parts.append(
    f'<text x="{LEG_X + 42}" y="{LEG_Y + 13}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10.5" fill="#333333">'
    f'surrogate score: estimate of r(x) on partial state</text>'
)
# solid pale-blue pill = oracle measured (second row, directly below)
parts.append(
    f'<rect x="{LEG_X}" y="{LEG_Y + 24}" width="34" height="18" rx="3" ry="3" '
    f'fill="#E8EFF7" stroke="#5C8DBC" stroke-width="0.9"/>'
)
parts.append(
    f'<text x="{LEG_X + 42}" y="{LEG_Y + 37}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10.5" fill="#333333">'
    f'oracle score: r(x) on completed sample</text>'
)

# ============================================================
# PANEL B: Test-time fine-tuning
# ============================================================

PANEL_B_Y = 700
PANEL_B_H = 350
parts.append(
    f'<text x="40" y="{PANEL_B_Y + 30}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="17" font-weight="700" fill="#111111">'
    f'Feedback for adaptation with test-time fine-tuning</text>'
)

# Prior before update (blue blob)
B_blob1_cx = 165
B_blob1_cy = PANEL_B_Y + 175
parts.append(
    f'<text x="{B_blob1_cx}" y="{B_blob1_cy - 75}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" fill="#111111">'
    f'Prior before update <tspan font-family="Times New Roman, Times, serif" '
    f'font-style="italic">p&#952;(x)</tspan></text>'
)
parts.append(density_blob(B_blob1_cx, B_blob1_cy, scale=1.0, palette="blue"))

# arrow
parts.append(arrow(B_blob1_cx + 110, B_blob1_cy, B_blob1_cx + 165, B_blob1_cy))

# Evaluated samples (x, r(x))
B_es_x = 380
B_es_y = PANEL_B_Y + 100
parts.append(
    f'<text x="{B_es_x + 140}" y="{B_es_y - 20}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" font-weight="600" fill="#111111">'
    f'Evaluated samples (x, r(x))</text>'
)
B_samples = [
    ([("B", False), ("P", False), ("G", False), ("R", False), ("C", False)], "—", None),
    ([("B", False), ("G", False), ("P", False), ("R", False), ("C", False)], "—", None),
    ([("R", False), ("P", False), ("G", False), ("B", False), ("C", False)], "—", None),
]
B_stack_svg, B_stack_w = stacked_samples(B_es_x, B_es_y + 8, B_samples)
parts.append(B_stack_svg)

# Update box (pink) — declare its position first so the connecting arrow can target it
upd_x = 850
upd_y = PANEL_B_Y + 140
upd_w = 160
upd_h = 80

# arrow to update box
arr_y = B_blob1_cy
parts.append(arrow(B_es_x + B_stack_w + 8, arr_y, upd_x - 5, arr_y))

# Draw the Update box itself
parts.append(
    f'<rect x="{upd_x}" y="{upd_y}" width="{upd_w}" height="{upd_h}" rx="8" ry="8" '
    f'fill="#D9E5F2" stroke="#222222" stroke-width="0.9"/>'
)
parts.append(
    f'<text x="{upd_x + upd_w/2}" y="{upd_y + 38}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="16" font-weight="700" fill="#111111">'
    f'Update</text>'
)
parts.append(
    f'<text x="{upd_x + upd_w/2}" y="{upd_y + 60}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" fill="#222222">'
    f'adapt <tspan font-style="italic">p&#952;</tspan></text>'
)

# arrow update → after-update prior
parts.append(arrow(upd_x + upd_w + 5, upd_y + upd_h / 2, upd_x + upd_w + 50, upd_y + upd_h / 2))

# Prior after update (pink, shifted shape)
B_blob2_cx = 1230
B_blob2_cy = B_blob1_cy
parts.append(
    f'<text x="{B_blob2_cx}" y="{B_blob2_cy - 75}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" fill="#111111">'
    f'Prior after update <tspan font-family="Times New Roman, Times, serif" '
    f'font-style="italic">p&#952;\'(x)</tspan></text>'
)
parts.append(density_blob_shifted(B_blob2_cx, B_blob2_cy, scale=1.05, palette="deep_blue"))

# Feedback loop: from prior-after-update back to evaluated samples (the loop)
loop_y = PANEL_B_Y + PANEL_B_H - 25
parts.append(
    f'<path d="M {B_blob2_cx} {B_blob2_cy + 60} '
    f'L {B_blob2_cx} {loop_y} '
    f'L {B_es_x + B_stack_w / 2 - 10} {loop_y} '
    f'L {B_es_x + B_stack_w / 2 - 10} {B_es_y + 100}" '
    f'fill="none" stroke="#222222" stroke-width="1.0" marker-end="url(#ah)"/>'
)
parts.append(
    f'<text x="{(B_blob2_cx + B_es_x + B_stack_w / 2) / 2}" y="{loop_y - 8}" '
    f'text-anchor="middle" font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" '
    f'font-style="italic" fill="#111111">future samples drawn from the updated prior</text>'
)

parts.append("</svg>")

with open("/home/claude/figure1.svg", "w") as f:
    f.write("\n".join(parts))

print("svg written")
