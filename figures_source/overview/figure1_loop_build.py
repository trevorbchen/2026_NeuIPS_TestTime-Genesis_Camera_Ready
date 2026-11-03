"""Build Figure 1 as a single online loop (Section 2's two-step structure).

Layout
------
  Top row              Step 1 (shared): sample, score, evaluated samples
  Step-2 header        text band that splits the figure horizontally
  Left column          Frozen-backbone search: Step 2 = select-only, no return
  Right column         Test-time fine-tuning: Step 2 = reshape + update
  Return arrow         from the fine-tune branch's pθ' back up to pθ at the top
                       (this is the only edge that closes the loop)

Color conventions (pale-blue family + greys; deep vs light marks the priors)
  Tokens               5 shades of blue, light to deep
  pθ blob              light-blue palette
  pθ' blob             deep-blue palette
  Oracle r(x)          solid pale-blue pill
  Surrogate r̂(x)       dashed pale-blue pill
  Likelihood log pθ    solid pale-grey pill
  Update / reshape     pale-blue accent boxes (matches the Fig-3 register)
"""

# ---------- token / score helpers ----------

TOKEN_COLORS = {
    "B": "#A8C5E0",
    "P": "#5E92C4",
    "G": "#C9DCEE",
    "R": "#2A6FB8",
    "C": "#7FAAD0",
}
TOKEN_W = 22
TOKEN_GAP = 3


def token_square(x, y, color_key, masked=False):
    if masked:
        return (
            f'<rect x="{x}" y="{y}" width="{TOKEN_W}" height="{TOKEN_W}" rx="3" ry="3" '
            f'fill="#FFFFFF" stroke="#B0B0B0" stroke-width="0.7"/>'
            f'<text x="{x + TOKEN_W/2}" y="{y + TOKEN_W/2 + 4.5}" text-anchor="middle" '
            f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
            f'font-size="11" fill="#888888">?</text>'
        )
    fill = TOKEN_COLORS[color_key]
    return (
        f'<rect x="{x}" y="{y}" width="{TOKEN_W}" height="{TOKEN_W}" rx="3" ry="3" '
        f'fill="{fill}" stroke="#666666" stroke-width="0.7"/>'
    )


def token_row(x, y, pattern):
    parts = []
    cur = x
    for k, m in pattern:
        parts.append(token_square(cur, y, k, m))
        cur += TOKEN_W + TOKEN_GAP
    return "\n".join(parts), cur


def oracle_pill(x, y, text="—", w=44, h=22):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" ry="3" '
        f'fill="#E8EFF7" stroke="#5C8DBC" stroke-width="0.9"/>'
        f'<text x="{x + w/2}" y="{y + h/2 + 4}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
        f'font-size="10" fill="#2A4A6F">{text}</text>'
    )


def surrogate_pill(x, y, text="—", w=44, h=22):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" ry="3" '
        f'fill="#EEF3F9" stroke="#5C8DBC" stroke-width="0.9" stroke-dasharray="3,2"/>'
        f'<text x="{x + w/2}" y="{y + h/2 + 4}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
        f'font-size="10" fill="#2A4A6F">{text}</text>'
    )


def likelihood_pill(x, y, text="—", w=56, h=22):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" ry="3" '
        f'fill="#F0F0F0" stroke="#A0A0A0" stroke-width="0.8"/>'
        f'<text x="{x + w/2}" y="{y + h/2 + 4}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
        f'font-size="10" fill="#333333">{text}</text>'
    )


def density_blob(cx, cy, scale=1.0, palette="light"):
    if palette == "deep":
        layers = [(1.00, "#B5CEE5"), (0.82, "#7FA8CE"), (0.62, "#4F82B5"),
                  (0.42, "#2A609A"), (0.25, "#0F4178")]
    else:
        layers = [(1.00, "#E2EEF9"), (0.82, "#BFDBF0"), (0.62, "#8EBFE2"),
                  (0.42, "#5599D0"), (0.25, "#2A6FB8")]
    out = []
    for s, color in layers:
        rx = 75 * s * scale
        ry = 38 * s * scale
        out.append(f'<ellipse cx="{cx - 28*s*scale}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{color}"/>')
        out.append(f'<ellipse cx="{cx + 28*s*scale}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{color}"/>')
    return "\n".join(out)


def density_blob_shifted(cx, cy, scale=1.0, palette="deep"):
    if palette == "deep":
        layers = [(1.00, "#B5CEE5"), (0.82, "#7FA8CE"), (0.62, "#4F82B5"),
                  (0.42, "#2A609A"), (0.25, "#0F4178")]
    else:
        layers = [(1.00, "#E2EEF9"), (0.82, "#BFDBF0"), (0.62, "#8EBFE2"),
                  (0.42, "#5599D0"), (0.25, "#2A6FB8")]
    out = []
    for s, color in layers:
        rx = 65 * s * scale
        ry = 36 * s * scale
        out.append(f'<ellipse cx="{cx - 42*s*scale}" cy="{cy + 4*s*scale}" rx="{rx}" ry="{ry}" fill="{color}"/>')
        out.append(f'<ellipse cx="{cx + 42*s*scale}" cy="{cy - 4*s*scale}" rx="{rx*0.95}" ry="{ry*0.95}" fill="{color}"/>')
    return "\n".join(out)


def stacked_samples(x, y, rows, dx=-6, dy=6):
    n_rows = len(rows)
    full_row_w = (TOKEN_W + TOKEN_GAP) * 5 - TOKEN_GAP + 60
    out = []
    for k in (2, 1):
        ox, oy = x + dx * k, y + dy * k
        out.append(
            f'<rect x="{ox - 6}" y="{oy - 6}" width="{full_row_w + 12}" '
            f'height="{n_rows * 28 + 12}" rx="4" ry="4" '
            f'fill="#FFFFFF" stroke="#999999" stroke-width="0.8"/>'
        )
    out.append(
        f'<rect x="{x - 6}" y="{y - 6}" width="{full_row_w + 12}" '
        f'height="{n_rows * 28 + 12}" rx="4" ry="4" '
        f'fill="#FFFFFF" stroke="#222222" stroke-width="0.9"/>'
    )
    cur_y = y
    for tokens, oracle_text in rows:
        row_svg, end_x = token_row(x, cur_y, tokens)
        out.append(row_svg)
        out.append(oracle_pill(end_x + 6, cur_y - 1, oracle_text))
        cur_y += 28
    out.append(
        f'<text x="{x + full_row_w / 2 - 6}" y="{cur_y + 6}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
        f'font-size="14" fill="#888888">...</text>'
    )
    return "\n".join(out), full_row_w


def arrow(x1, y1, x2, y2, marker="ah", stroke="#222222", width=1.0, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra} marker-end="url(#{marker})"/>'
    )


# ============================================================
# Build
# ============================================================

W, H = 1620, 660

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
    '''<defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#222222"/>
        </marker>
      </defs>''',
    f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
]

# ----------------------------------------------------------------
# Step 1 — shared horizontal row
# ----------------------------------------------------------------

STEP1_Y_TITLE = 60
STEP1_Y_AXIS = 155

parts.append(
    f'<text x="40" y="{STEP1_Y_TITLE}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="17" font-weight="700" fill="#111111">'
    f'Step 1: sample candidates from p&#952;, score them</text>'
)
parts.append(
    f'<text x="40" y="{STEP1_Y_TITLE + 18}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="11" font-style="italic" fill="#666666">'
    f'shared by both branches below</text>'
)

# pθ blob anchor
PTHETA_CX, PTHETA_CY = 145, STEP1_Y_AXIS
parts.append(density_blob(PTHETA_CX, PTHETA_CY, scale=0.85, palette="light"))
# label below the blob so the return arrow can land cleanly on top
parts.append(
    f'<text x="{PTHETA_CX - 18}" y="{PTHETA_CY + 70}" text-anchor="end" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" fill="#111111">'
    f'pretrained prior</text>'
)
parts.append(
    f'<text x="{PTHETA_CX - 12}" y="{PTHETA_CY + 70}" text-anchor="start" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" font-style="italic" fill="#111111">'
    f'p&#952;</text>'
)
parts.append(arrow(PTHETA_CX + 80, PTHETA_CY, PTHETA_CX + 145, PTHETA_CY))

# sample box
SAMPLE_X = 305
parts.append(
    f'<rect x="{SAMPLE_X}" y="{STEP1_Y_AXIS - 22}" width="160" height="44" rx="2" '
    f'fill="#FFFFFF" stroke="#222222" stroke-width="0.9"/>'
)
parts.append(
    f'<text x="{SAMPLE_X + 80}" y="{STEP1_Y_AXIS - 4}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" fill="#111111">'
    f'sample</text>'
)
parts.append(
    f'<text x="{SAMPLE_X + 80}" y="{STEP1_Y_AXIS + 14}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" '
    f'font-style="italic" fill="#111111">x ~ p&#952;</text>'
)
parts.append(arrow(SAMPLE_X + 162, STEP1_Y_AXIS, SAMPLE_X + 207, STEP1_Y_AXIS))

# score box
SCORE_X = SAMPLE_X + 210
parts.append(
    f'<rect x="{SCORE_X}" y="{STEP1_Y_AXIS - 50}" width="200" height="100" rx="3" '
    f'fill="#FFFFFF" stroke="#999999" stroke-width="0.9"/>'
)
parts.append(
    f'<text x="{SCORE_X + 100}" y="{STEP1_Y_AXIS - 32}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" '
    f'font-weight="600" fill="#111111">score</text>'
)
parts.append(
    f'<text x="{SCORE_X + 18}" y="{STEP1_Y_AXIS - 8}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" fill="#222222">'
    f'oracle <tspan font-style="italic">r(x)</tspan></text>'
)
parts.append(oracle_pill(SCORE_X + 130, STEP1_Y_AXIS - 22))
parts.append(
    f'<line x1="{SCORE_X + 18}" y1="{STEP1_Y_AXIS + 6}" '
    f'x2="{SCORE_X + 182}" y2="{STEP1_Y_AXIS + 6}" '
    f'stroke="#BBBBBB" stroke-width="0.8" stroke-dasharray="3,3"/>'
)
parts.append(
    f'<text x="{SCORE_X + 18}" y="{STEP1_Y_AXIS + 30}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" '
    f'font-style="italic" fill="#222222">log p&#952;(x)</text>'
)
parts.append(likelihood_pill(SCORE_X + 130, STEP1_Y_AXIS + 16))
parts.append(arrow(SCORE_X + 202, STEP1_Y_AXIS, SCORE_X + 247, STEP1_Y_AXIS))

# evaluated samples stack
ES_X = SCORE_X + 250
ES_Y = STEP1_Y_AXIS - 35
es_rows = [
    ([("B", False), ("P", False), ("G", False), ("R", False), ("C", False)], "—"),
    ([("B", False), ("G", False), ("P", False), ("R", False), ("C", False)], "—"),
]
parts.append(
    f'<text x="{ES_X + 100}" y="{ES_Y - 14}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="12" '
    f'font-weight="600" fill="#111111">evaluated samples</text>'
)
es_svg, es_w = stacked_samples(ES_X, ES_Y, es_rows)
parts.append(es_svg)

# Funnel down to a junction node, then diverge from there
FORK_X = W // 2
FORK_Y = 230
# arrow from evaluated-samples stack down into the junction
parts.append(
    f'<path d="M {ES_X + es_w / 2} {ES_Y + 60} '
    f'C {ES_X + es_w / 2} {FORK_Y - 30}, {FORK_X} {FORK_Y - 30}, {FORK_X} {FORK_Y - 6}" '
    f'fill="none" stroke="#222222" stroke-width="1.0" marker-end="url(#ah)"/>'
)
# junction dot
parts.append(
    f'<circle cx="{FORK_X}" cy="{FORK_Y}" r="4" fill="#222222"/>'
)

# ----------------------------------------------------------------
# Step 2 — header and split
# ----------------------------------------------------------------

STEP2_Y_TITLE = FORK_Y + 5
parts.append(
    f'<text x="{FORK_X + 12}" y="{STEP2_Y_TITLE}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="15" font-weight="700" fill="#111111">'
    f'Step 2: use the feedback</text>'
)
parts.append(
    f'<text x="{FORK_X + 12}" y="{STEP2_Y_TITLE + 16}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="11" font-style="italic" fill="#666666">'
    f'two ways the loop closes (or doesn\'t)</text>'
)

LEFT_BR_X = 360
RIGHT_BR_X = 950
BRANCH_TOP_Y = 320
parts.append(
    f'<path d="M {FORK_X} {FORK_Y + 4} '
    f'C {FORK_X} {BRANCH_TOP_Y - 20}, {LEFT_BR_X} {BRANCH_TOP_Y - 20}, {LEFT_BR_X} {BRANCH_TOP_Y}" '
    f'fill="none" stroke="#222222" stroke-width="1.0" marker-end="url(#ah)"/>'
)
parts.append(
    f'<path d="M {FORK_X} {FORK_Y + 4} '
    f'C {FORK_X} {BRANCH_TOP_Y - 20}, {RIGHT_BR_X} {BRANCH_TOP_Y - 20}, {RIGHT_BR_X} {BRANCH_TOP_Y}" '
    f'fill="none" stroke="#222222" stroke-width="1.0" marker-end="url(#ah)"/>'
)

# ----------------------------------------------------------------
# LEFT branch — frozen-backbone search
# ----------------------------------------------------------------

LEFT_TITLE_Y = BRANCH_TOP_Y + 30
parts.append(
    f'<text x="40" y="{LEFT_TITLE_Y}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="15" font-weight="700" fill="#111111">'
    f'Frozen-backbone search</text>'
)
parts.append(
    f'<text x="40" y="{LEFT_TITLE_Y + 18}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="11" font-style="italic" fill="#666666">'
    f'Step 2 = select; p&#952; unchanged; the loop ends here</text>'
)

SEARCH_LEFT_X = 60
COL_T1 = SEARCH_LEFT_X + 200
COL_T2 = COL_T1 + 200
COL_FINAL = COL_T2 + 200
SEARCH_TOP = LEFT_TITLE_Y + 50
ROW_Y = [SEARCH_TOP + 30, SEARCH_TOP + 85, SEARCH_TOP + 140]

# Frozen pθ anchor on the left side of the search tree
parts.append(density_blob(SEARCH_LEFT_X + 75, ROW_Y[1] + 11, scale=0.7, palette="light"))
parts.append(
    f'<text x="{SEARCH_LEFT_X + 75}" y="{ROW_Y[2] + 60}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" '
    f'font-style="italic" fill="#666666">generator fixed</text>'
)

# Column headers
for cx, label in [(COL_T1, "partial state t\u2081"),
                  (COL_T2, "partial state t\u2082"),
                  (COL_FINAL, "complete sequence")]:
    parts.append(
        f'<text x="{cx + 60}" y="{SEARCH_TOP - 8}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" '
        f'font-weight="600" fill="#111111">{label}</text>'
    )

# Branches from prior to t1
for ry in ROW_Y:
    parts.append(
        f'<path d="M {SEARCH_LEFT_X + 130} {ROW_Y[1] + 11} '
        f'C {SEARCH_LEFT_X + 170} {ROW_Y[1] + 11}, '
        f'{COL_T1 - 30} {ry + 11}, {COL_T1 - 5} {ry + 11}" '
        f'fill="none" stroke="#222222" stroke-width="0.9" marker-end="url(#ah)"/>'
    )

# t1
patterns_t1 = [
    [("B", False), ("P", True), ("G", True), ("R", True), ("C", True)],
    [("B", False), ("G", True), ("P", True), ("R", True), ("C", True)],
    [("P", False), ("B", True), ("G", True), ("R", True), ("C", True)],
]
for ry, pat in zip(ROW_Y, patterns_t1):
    row_svg, end_x = token_row(COL_T1, ry, pat)
    parts.append(row_svg)
    parts.append(surrogate_pill(end_x + 4, ry - 1))

# t1 → t2
for i in (0, 1):
    parts.append(
        f'<line x1="{COL_T1 + 175}" y1="{ROW_Y[i] + 11}" '
        f'x2="{COL_T2 - 5}" y2="{ROW_Y[i] + 11}" '
        f'stroke="#222222" stroke-width="0.9" marker-end="url(#ah)"/>'
    )
# bottom row pruned
cx_x = COL_T2 + 60
cy_y = ROW_Y[2] + 11
parts.append(
    f'<g stroke="#888888" stroke-width="2.4" fill="none" stroke-linecap="round">'
    f'<line x1="{cx_x - 9}" y1="{cy_y - 9}" x2="{cx_x + 9}" y2="{cy_y + 9}"/>'
    f'<line x1="{cx_x + 9}" y1="{cy_y - 9}" x2="{cx_x - 9}" y2="{cy_y + 9}"/>'
    f'</g>'
)

# t2
patterns_t2 = [
    [("B", False), ("P", False), ("G", True), ("R", True), ("C", True)],
    [("B", False), ("G", False), ("P", True), ("R", True), ("C", True)],
]
for ry, pat in zip(ROW_Y[:2], patterns_t2):
    row_svg, end_x = token_row(COL_T2, ry, pat)
    parts.append(row_svg)
    parts.append(surrogate_pill(end_x + 4, ry - 1))

# t2 → complete
for i in (0, 1):
    parts.append(
        f'<line x1="{COL_T2 + 175}" y1="{ROW_Y[i] + 11}" '
        f'x2="{COL_FINAL - 5}" y2="{ROW_Y[i] + 11}" '
        f'stroke="#222222" stroke-width="0.9" marker-end="url(#ah)"/>'
    )

# Complete sequences with oracle pills
patterns_final = [
    [("B", False), ("P", False), ("G", False), ("R", False), ("C", False)],
    [("B", False), ("G", False), ("P", False), ("R", False), ("C", False)],
]
for ry, pat in zip(ROW_Y[:2], patterns_final):
    row_svg, end_x = token_row(COL_FINAL, ry, pat)
    parts.append(row_svg)
    parts.append(oracle_pill(end_x + 6, ry - 1))

# "select" bracket
sel_x = COL_FINAL + 128
sel_top = ROW_Y[0] - 5
sel_bot = ROW_Y[1] + TOKEN_W + 5
parts.append(
    f'<path d="M {sel_x + 8} {sel_top} '
    f'Q {sel_x} {sel_top} {sel_x} {sel_top + 8} '
    f'L {sel_x} {sel_bot - 8} '
    f'Q {sel_x} {sel_bot} {sel_x + 8} {sel_bot}" '
    f'fill="none" stroke="#222222" stroke-width="1.0"/>'
)
parts.append(
    f'<text x="{sel_x + 16}" y="{(sel_top + sel_bot)/2 + 5}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" '
    f'font-weight="600" fill="#111111">select</text>'
)


# Legend (stacked rows)
LEG_Y = ROW_Y[2] + 38
LEG_X = COL_T1 - 30
parts.append(
    f'<rect x="{LEG_X}" y="{LEG_Y}" width="34" height="16" rx="3" ry="3" '
    f'fill="#EEF3F9" stroke="#5C8DBC" stroke-width="0.8" stroke-dasharray="3,2"/>'
)
parts.append(
    f'<text x="{LEG_X + 42}" y="{LEG_Y + 12}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#444444">'
    f'surrogate estimate of r(x) on partial state</text>'
)
parts.append(
    f'<rect x="{LEG_X}" y="{LEG_Y + 22}" width="34" height="16" rx="3" ry="3" '
    f'fill="#E8EFF7" stroke="#5C8DBC" stroke-width="0.8"/>'
)
parts.append(
    f'<text x="{LEG_X + 42}" y="{LEG_Y + 34}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#444444">'
    f'oracle r(x) measured on completion</text>'
)

# ----------------------------------------------------------------
# RIGHT branch — test-time fine-tuning
# ----------------------------------------------------------------

RIGHT_TITLE_Y = LEFT_TITLE_Y
parts.append(
    f'<text x="850" y="{RIGHT_TITLE_Y}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="15" font-weight="700" fill="#111111">'
    f'Test-time fine-tuning</text>'
)
parts.append(
    f'<text x="850" y="{RIGHT_TITLE_Y + 18}" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
    f'font-size="11" font-style="italic" fill="#666666">'
    f'Step 2 = reshape and update; the loop closes back to p&#952;</text>'
)

RIGHT_AXIS_Y = (ROW_Y[0] + ROW_Y[1]) // 2 + 11
ENTRY_X = 870

# arrow → reshape
RESHAPE_X = ENTRY_X
# arrow drops down from the Step-2 diverging path onto the top of reshape
parts.append(arrow(RESHAPE_X + 80, BRANCH_TOP_Y, RESHAPE_X + 80, RIGHT_AXIS_Y - 36))

# reshape box
parts.append(
    f'<rect x="{RESHAPE_X}" y="{RIGHT_AXIS_Y - 35}" width="160" height="70" rx="6" '
    f'fill="#D9E5F2" stroke="#222222" stroke-width="0.9"/>'
)
parts.append(
    f'<text x="{RESHAPE_X + 80}" y="{RIGHT_AXIS_Y - 14}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" '
    f'font-weight="700" fill="#111111">reshape</text>'
)
parts.append(
    f'<text x="{RESHAPE_X + 80}" y="{RIGHT_AXIS_Y + 4}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#222222">'
    f'CVaR + neg-score</text>'
)
parts.append(
    f'<text x="{RESHAPE_X + 80}" y="{RIGHT_AXIS_Y + 19}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#222222">'
    f'+ validity penalty</text>'
)

# arrow → fine-tune
FT_X = RESHAPE_X + 180
parts.append(arrow(FT_X - 13, RIGHT_AXIS_Y, FT_X, RIGHT_AXIS_Y))

# fine-tune box
parts.append(
    f'<rect x="{FT_X}" y="{RIGHT_AXIS_Y - 35}" width="150" height="70" rx="6" '
    f'fill="#D9E5F2" stroke="#222222" stroke-width="0.9"/>'
)
parts.append(
    f'<text x="{FT_X + 75}" y="{RIGHT_AXIS_Y - 14}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" '
    f'font-weight="700" fill="#111111">fine-tune</text>'
)
parts.append(
    f'<text x="{FT_X + 75}" y="{RIGHT_AXIS_Y + 4}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" fill="#222222">'
    f'gradient on L_ft</text>'
)
parts.append(
    f'<text x="{FT_X + 75}" y="{RIGHT_AXIS_Y + 19}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" '
    f'font-style="italic" fill="#222222">updates &#952;</text>'
)

# arrow → pθ'
PRIME_CX = FT_X + 240
PRIME_CY = RIGHT_AXIS_Y
parts.append(arrow(FT_X + 152, RIGHT_AXIS_Y, PRIME_CX - 65, RIGHT_AXIS_Y))

# pθ' deep-blue blob
parts.append(density_blob_shifted(PRIME_CX, PRIME_CY, scale=0.85, palette="deep"))
parts.append(
    f'<text x="{PRIME_CX - 14}" y="{PRIME_CY - 65}" text-anchor="end" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" fill="#111111">'
    f'updated prior</text>'
)
parts.append(
    f'<text x="{PRIME_CX - 8}" y="{PRIME_CY - 65}" text-anchor="start" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="13" font-style="italic" fill="#111111">'
    f'p&#952;\u2032</text>'
)
parts.append(
    f'<text x="{PRIME_CX}" y="{PRIME_CY + 75}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="10" '
    f'font-style="italic" fill="#666666">replaces p&#952; for the next round</text>'
)

# ----------------------------------------------------------------
# Return arrow — pθ' → pθ (the only edge that closes the loop)
# ----------------------------------------------------------------

RAIL_Y = 25
parts.append(
    f'<path d="M {PRIME_CX + 70} {PRIME_CY} '
    f'L {W - 30} {PRIME_CY} '
    f'Q {W - 12} {PRIME_CY} {W - 12} {PRIME_CY - 20} '
    f'L {W - 12} {RAIL_Y + 20} '
    f'Q {W - 12} {RAIL_Y} {W - 30} {RAIL_Y} '
    f'L {PTHETA_CX + 20} {RAIL_Y} '
    f'Q {PTHETA_CX} {RAIL_Y} {PTHETA_CX} {RAIL_Y + 20} '
    f'L {PTHETA_CX} {PTHETA_CY - 36}" '
    f'fill="none" stroke="#222222" stroke-width="1.4" '
    f'stroke-dasharray="6,4" marker-end="url(#ah)"/>'
)
parts.append(
    f'<text x="{(PRIME_CX + PTHETA_CX) / 2}" y="{RAIL_Y - 6}" text-anchor="middle" '
    f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11" '
    f'font-style="italic" fill="#444444">'
    f'next round starts from the updated prior</text>'
)

parts.append("</svg>")

with open("/home/claude/figure1.svg", "w") as f:
    f.write("\n".join(parts))

print("svg written")
