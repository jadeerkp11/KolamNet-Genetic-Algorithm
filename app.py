import base64
import io
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from kolamnet import (
    evolve_ga,
    hill_climb,
    random_search,
    render_pattern,
    image_to_target,
    suggest_photo_settings,
)


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Kolam Studio",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------
# Rendering helpers (flat 2D kolam)
# ------------------------------------------------------------
def render_pattern_no_dots(pattern, ax):
    """Render the kolam, then strip the background grid dots (pulli) from the axes."""
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    render_pattern(pattern, ax=ax)

    no_marker = (None, "", "None", " ", "none")
    no_line = (None, "", "None", " ", "none")
    for artist in list(ax.get_children()):
        # Dots drawn with ax.scatter(...)
        if isinstance(artist, PathCollection):
            artist.remove()
        # Dots drawn with ax.plot(..., "o") / marker-only lines
        elif isinstance(artist, Line2D):
            if artist.get_marker() not in no_marker and artist.get_linestyle() in no_line:
                artist.remove()
        # Dots drawn as small circle patches
        elif isinstance(artist, Circle):
            artist.remove()


def pattern_png(pattern):
    """Render a tile pattern as a downloadable 2D PNG image."""
    fig, ax = plt.subplots(figsize=(7, 7))
    render_pattern_no_dots(pattern, ax)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def matrix_csv(pattern):
    """Convert a tile matrix to downloadable CSV text."""
    buffer = io.StringIO()
    np.savetxt(buffer, pattern, delimiter=",", fmt="%d")
    return buffer.getvalue()


def display_pattern(pattern, title="Your kolam design"):
    fig, ax = plt.subplots(figsize=(6, 6))
    render_pattern_no_dots(pattern, ax)
    ax.set_title(title, fontsize=14, pad=14)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    plt.close(fig)


def next_seed(new_variation, fixed_seed):
    """A fresh random seed per click (so every click gives a new pattern),
    or the fixed seed when the user wants a reproducible result."""
    return random.randrange(1_000_000) if new_variation else int(fixed_seed)


def auto_find_area():
    """Button callback: run the (heuristic) pattern-area finder on the uploaded photo."""
    img = st.session_state.get("ref_img")
    if img is None:
        return
    found = suggest_photo_settings(img)
    l, t, r, b = found["crop_box"]
    st.session_state["crop_x"] = (int(round(l * 100)), max(int(round(r * 100)), int(round(l * 100)) + 5))
    st.session_state["crop_y"] = (int(round(t * 100)), max(int(round(b * 100)), int(round(t * 100)) + 5))
    st.session_state["opt_invert"] = found["invert"]
    st.session_state["opt_blobs"] = True
    st.session_state["opt_auto_thr"] = True


def crop_overlay(img, x_range, y_range):
    """Draw the chosen pattern area on top of the photo."""
    out = img.copy()
    W, H = out.size
    box = (x_range[0] / 100 * W, y_range[0] / 100 * H, x_range[1] / 100 * W, y_range[1] / 100 * H)
    draw = ImageDraw.Draw(out)
    for off in range(4):
        draw.rectangle(
            (box[0] - off, box[1] - off, box[2] + off, box[3] + off), outline=(255, 60, 120)
        )
    return out


# ------------------------------------------------------------
# Procedural kolam artwork used as the page background
# ------------------------------------------------------------
def kolam_background_b64():
    """Build a crisp, resolution-independent lotus kolam as an SVG (base64)."""
    cx = cy = 400

    def petal(angle, r0, r1, w):
        k1 = r0 + (r1 - r0) * 0.35
        k2 = r0 + (r1 - r0) * 0.80
        d = (
            f"M{r0},0 C{k1:.1f},{-w:.1f} {k2:.1f},{-w * 0.7:.1f} {r1},0 "
            f"C{k2:.1f},{w * 0.7:.1f} {k1:.1f},{w:.1f} {r0},0 Z"
        )
        return f'<path d="{d}" transform="translate({cx} {cy}) rotate({angle})"/>'

    def polar(angle_deg, radius):
        a = np.radians(angle_deg)
        return cx + radius * np.cos(a), cy + radius * np.sin(a)

    lines = []

    # Outer ring of 16 lotus petals (long and short alternating)
    for i in range(16):
        long_petal = i % 2 == 0
        angle = i * 22.5
        r1 = 335 if long_petal else 292
        w = 40 if long_petal else 29
        lines.append(petal(angle, 205, r1, w))
        lines.append(petal(angle, 222, r1 - 24, w * 0.5))

    # Eight mid petals around the central diamond
    for i in range(8):
        angle = i * 45
        lines.append(petal(angle, 120, 228, 36))
        lines.append(petal(angle, 136, 200, 17))

    # Diamond, inscribed square and small inner diamond
    h = 120
    lines.append(
        f'<polygon points="{cx},{cy - h} {cx + h},{cy} {cx},{cy + h} {cx - h},{cy}"/>'
    )
    lines.append(f'<rect x="{cx - 60}" y="{cy - 60}" width="120" height="120"/>')
    h2 = 42
    lines.append(
        f'<polygon points="{cx},{cy - h2} {cx + h2},{cy} {cx},{cy + h2} {cx - h2},{cy}"/>'
    )

    # Central flower
    for i in range(8):
        lines.append(petal(i * 45, 10, 54, 14))
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="57"/>')

    # Pulli (dot) ring and thin outer rings
    dots = []
    for i in range(48):
        x, y = polar(i * 7.5, 366)
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4"/>')
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="374" stroke-width="1"/>')

    # Marigold-orange accents at the petal tips
    accents = []
    for i in range(8):
        x, y = polar(i * 45, 241)
        accents.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.5"/>')
    for i in range(16):
        r_tip = 349 if i % 2 == 0 else 306
        x, y = polar(i * 22.5, r_tip)
        accents.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
    accents.append(f'<circle cx="{cx}" cy="{cy}" r="9"/>')

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800">'
        '<defs><filter id="glow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="2.6" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter></defs>"
        '<g filter="url(#glow)">'
        '<g fill="none" stroke="#e9e4ff" stroke-width="2.2" stroke-linejoin="round">'
        + "".join(lines)
        + "</g>"
        '<g fill="#cfc6ff">' + "".join(dots) + "</g>"
        '<g fill="#ffb347">' + "".join(accents) + "</g>"
        "</g></svg>"
    )
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


# ------------------------------------------------------------
# Visual styling
# ------------------------------------------------------------
APP_CSS = """
<style>
:root {
    --bg: #0b0f24;
    --panel: rgba(255,255,255,0.055);
    --panel-border: rgba(160,170,255,0.20);
    --text: #e8ebff;
    --muted: #9aa3d6;
    --accent: #7c5cff;
    --accent-2: #19d3da;
    --accent-3: #ff5fa2;
}
.stApp {
    background:
        radial-gradient(900px 500px at 12% -5%, rgba(124,92,255,0.28), transparent 60%),
        radial-gradient(800px 500px at 100% 10%, rgba(25,211,218,0.18), transparent 60%),
        radial-gradient(700px 600px at 50% 120%, rgba(255,95,162,0.16), transparent 60%),
        var(--bg);
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1200px; padding-top: 1.6rem; padding-bottom: 3rem; }

h1, h2, h3, h4 { color: #ffffff !important; letter-spacing: -0.01em; }
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* ---------- Hero with rotating cube and perspective floor ---------- */
.hero {
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    padding: 38px 44px; margin-bottom: 26px; min-height: 270px;
    border-radius: 26px;
    background: linear-gradient(135deg, rgba(124,92,255,0.30), rgba(25,211,218,0.10) 60%, rgba(255,95,162,0.16));
    border: 1px solid var(--panel-border);
    box-shadow: 0 30px 60px -25px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.18);
}
.hero-floor {
    position: absolute; left: -20%; right: -20%; bottom: -120px; height: 300px;
    background-image:
        linear-gradient(rgba(124,92,255,0.45) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,92,255,0.45) 1px, transparent 1px);
    background-size: 44px 44px;
    transform: perspective(420px) rotateX(68deg);
    mask-image: linear-gradient(to top, rgba(0,0,0,0.9), transparent 85%);
    -webkit-mask-image: linear-gradient(to top, rgba(0,0,0,0.9), transparent 85%);
    animation: floor-slide 5s linear infinite;
}
@keyframes floor-slide { from { background-position: 0 0; } to { background-position: 0 44px; } }
.hero-text { position: relative; z-index: 2; max-width: 600px; }
.hero-kicker {
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    background: rgba(255,255,255,0.08); border: 1px solid var(--panel-border);
    color: #bfc7ff; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
}
.hero h1 {
    font-size: 3rem; line-height: 1.05; margin: 14px 0 12px; font-weight: 800;
    background: linear-gradient(90deg, #ffffff 0%, #b7a8ff 45%, #5ee6ea 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important;
}
.hero p { color: #c4caf5; font-size: 1.1rem; margin: 0; }

.scene {
    position: relative; z-index: 2; width: 150px; height: 150px;
    perspective: 700px; margin-right: 40px; flex-shrink: 0;
}
.cube {
    position: absolute; inset: 0; transform-style: preserve-3d;
    animation: spin 14s linear infinite;
}
@keyframes spin {
    from { transform: rotateX(-24deg) rotateY(0deg); }
    to   { transform: rotateX(-24deg) rotateY(360deg); }
}
.face {
    position: absolute; width: 150px; height: 150px;
    display: flex; align-items: center; justify-content: center;
    font-size: 3rem; color: rgba(255,255,255,0.92);
    border: 1.5px solid rgba(255,255,255,0.35);
    background:
        radial-gradient(circle at 25% 25%, rgba(255,255,255,0.22), transparent 55%),
        linear-gradient(135deg, rgba(124,92,255,0.65), rgba(25,211,218,0.45));
    box-shadow: inset 0 0 30px rgba(255,255,255,0.12);
    backdrop-filter: blur(2px);
}
.face.front  { transform: rotateY(0deg)   translateZ(75px); }
.face.back   { transform: rotateY(180deg) translateZ(75px); }
.face.right  { transform: rotateY(90deg)  translateZ(75px); }
.face.left   { transform: rotateY(-90deg) translateZ(75px); }
.face.top    { transform: rotateX(90deg)  translateZ(75px); }
.face.bottom { transform: rotateX(-90deg) translateZ(75px); }
.cube-shadow {
    position: absolute; left: 50%; bottom: -46px; width: 130px; height: 24px;
    transform: translateX(-50%); border-radius: 50%;
    background: radial-gradient(ellipse, rgba(0,0,0,0.55), transparent 70%);
    animation: pulse 14s linear infinite;
}
@keyframes pulse { 0%,100% { opacity: 0.8; } 50% { opacity: 0.45; } }

/* ---------- Section headings with 3D step badges ---------- */
.step-title { display: flex; align-items: center; gap: 14px; margin: 30px 0 12px; }
.step-badge {
    width: 42px; height: 42px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; color: white; font-size: 1.1rem;
    background: linear-gradient(145deg, #8f73ff, #5a3de0);
    box-shadow: 0 5px 0 #3a25a8, 0 12px 20px rgba(90,61,224,0.45),
                inset 0 1px 0 rgba(255,255,255,0.35);
    transform: rotate(-4deg);
}
.step-title h2 { margin: 0 !important; padding: 0 !important; font-size: 1.6rem; }

/* ---------- Glass cards (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--panel-border) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 18px 40px -22px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.10);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(124,92,255,0.55) !important;
    box-shadow: 0 26px 50px -22px rgba(124,92,255,0.45), inset 0 1px 0 rgba(255,255,255,0.14);
}

/* ---------- Buttons: chunky 3D keys ---------- */
div.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #8f73ff 0%, #6a47f0 100%);
    color: #fff; border: none; border-radius: 14px;
    font-weight: 700; min-height: 3rem;
    box-shadow: 0 6px 0 #3a25a8, 0 14px 22px rgba(90,61,224,0.40),
                inset 0 1px 0 rgba(255,255,255,0.35);
    transition: transform 0.08s ease, box-shadow 0.08s ease, filter 0.2s ease;
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.1); color: #fff; }
div.stButton > button[kind="primary"]:active {
    transform: translateY(5px);
    box-shadow: 0 1px 0 #3a25a8, 0 5px 10px rgba(90,61,224,0.40);
}
div.stButton > button[kind="primary"]:disabled {
    background: #2a3057; color: #7e86b8; box-shadow: 0 6px 0 #1b1f3d;
}
div.stButton > button[kind="secondary"],
div[data-testid="stDownloadButton"] button {
    background: rgba(255,255,255,0.06); color: #dfe3ff;
    border: 1px solid var(--panel-border); border-radius: 12px;
    box-shadow: 0 4px 0 rgba(0,0,0,0.35);
    transition: transform 0.08s ease, box-shadow 0.08s ease, border-color 0.2s ease;
}
div.stButton > button[kind="secondary"]:hover,
div[data-testid="stDownloadButton"] button:hover {
    border-color: var(--accent-2); color: #fff;
}
div[data-testid="stDownloadButton"] button:active {
    transform: translateY(3px); box-shadow: 0 1px 0 rgba(0,0,0,0.35);
}

/* ---------- Metrics as floating tiles ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(124,92,255,0.22), rgba(25,211,218,0.08));
    border: 1px solid var(--panel-border);
    border-radius: 16px; padding: 14px 16px;
    box-shadow: 0 8px 0 rgba(0,0,0,0.25), 0 16px 26px -14px rgba(0,0,0,0.7),
                inset 0 1px 0 rgba(255,255,255,0.14);
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: #ffffff; }
div[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #8f73ff, #19d3da) !important; }

/* ---------- Misc ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1433 0%, #0b0f24 100%);
    border-right: 1px solid var(--panel-border);
}
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.04);
    border: 1.5px dashed rgba(160,170,255,0.45); border-radius: 16px;
}
[data-testid="stExpander"] {
    background: var(--panel); border: 1px solid var(--panel-border) !important;
    border-radius: 16px;
}
.small-note { color: var(--muted); font-size: 0.9rem; }
.chip {
    display: inline-block; margin: 2px 6px 2px 0; padding: 4px 12px; border-radius: 999px;
    background: rgba(124,92,255,0.18); border: 1px solid rgba(124,92,255,0.45);
    color: #cfc6ff; font-size: 0.82rem; font-weight: 600;
}

/* ---------- Kolam background (slowly rotating, behind everything) ---------- */
.stApp::before {
    content: ""; position: fixed; top: 50%; left: 50%;
    width: min(125vh, 1150px); aspect-ratio: 1 / 1;
    background: url("data:image/svg+xml;base64,__KOLAM__") center / contain no-repeat;
    opacity: 0.38; pointer-events: none; z-index: 0;
    transform: translate(-50%, -50%);
    animation: kolam-rotate 260s linear infinite;
}
@keyframes kolam-rotate {
    from { transform: translate(-50%, -50%) rotate(0deg); }
    to   { transform: translate(-50%, -50%) rotate(360deg); }
}
@media (prefers-reduced-motion: reduce) {
    .stApp::before, .hero-floor, .cube { animation: none; }
}
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; background: transparent; }
[data-testid="stMain"] { background: transparent; }

/* sidebar is no longer used */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

/* ---------- Slim horizontal control bar ---------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(#ctrl-marker) {
    border-radius: 999px !important;
    padding: 0.3rem 1.1rem !important;
    margin-bottom: 14px;
    background: rgba(12,16,44,0.72);
    box-shadow: 0 10px 0 -6px rgba(0,0,0,0.35), 0 18px 30px -18px rgba(0,0,0,0.8),
                inset 0 1px 0 rgba(255,255,255,0.12);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(#ctrl-marker) [data-testid="stHorizontalBlock"] {
    align-items: center; gap: 0.8rem;
}
[data-testid="stElementContainer"]:has(#ctrl-marker),
.element-container:has(#ctrl-marker) { display: none; }
.ctrl-title { font-weight: 700; color: #fff; white-space: nowrap; font-size: 0.98rem; }
.ctrl-hint { color: var(--muted); font-size: 0.85rem; text-align: right; }

/* ---------- Dark select boxes, popovers and inputs ---------- */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 12px !important; min-height: 2.5rem;
}
div[data-baseweb="select"] * { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--muted) !important; }
div[data-baseweb="popover"] > div,
ul[role="listbox"], div[data-baseweb="menu"] {
    background: #151a3d !important; color: var(--text) !important;
}
li[role="option"] { color: var(--text) !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background: rgba(124,92,255,0.30) !important;
}
[data-testid="stPopover"] button {
    background: rgba(255,255,255,0.07); color: #dfe3ff;
    border: 1px solid var(--panel-border); border-radius: 12px;
    box-shadow: 0 4px 0 rgba(0,0,0,0.35); min-height: 2.5rem;
}
[data-testid="stPopover"] button:hover { border-color: var(--accent-2); color: #fff; }
[data-testid="stPopoverBody"] {
    background: #12163a; border: 1px solid var(--panel-border);
    border-radius: 16px; min-width: 340px;
}
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.07); color: var(--text);
}
</style>
"""

st.markdown(APP_CSS.replace("__KOLAM__", kolam_background_b64()), unsafe_allow_html=True)


def step_title(number, text):
    st.markdown(
        f'<div class="step-title"><div class="step-badge">{number}</div>'
        f"<h2>{text}</h2></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Slim horizontal control bar (replaces the old sidebar)
# ------------------------------------------------------------
with st.container(border=True):
    st.markdown('<span id="ctrl-marker"></span>', unsafe_allow_html=True)
    bar_title, bar_grid, bar_adv, bar_hint = st.columns([1.5, 1.5, 1.4, 3.6], gap="small")

    with bar_title:
        st.markdown('<div class="ctrl-title">🪷 Design controls</div>', unsafe_allow_html=True)

    with bar_grid:
        grid_size = st.selectbox(
            "Pattern grid",
            [8, 12, 16],
            index=1,
            format_func=lambda n: f"Grid {n} × {n}",
            label_visibility="collapsed",
        )

    with bar_adv:
        advanced = st.popover("⚙️ Advanced") if hasattr(st, "popover") else st.expander("⚙️ Advanced")
        with advanced:
            st.markdown("**Algorithm settings**")
            population = st.slider("GA population", 40, 200, 100, 20)
            generations = st.slider("Generations", 20, 250, 100, 10)
            mutation_rate = st.select_slider(
                "Mutation rate",
                options=[0.005, 0.01, 0.03, 0.05, 0.1],
                value=0.03,
            )
            new_variation = st.checkbox(
                "New variation on every click",
                value=True,
                help="On: each click uses a fresh random seed, so you get a new pattern "
                "every time. Off: the seed below is used, so results repeat exactly.",
            )
            random_seed = st.number_input(
                "Fixed seed (used when the box above is off)",
                min_value=0, max_value=999999, value=42, step=1,
            )
            creativity = st.slider(
                "Creativity for photo mode",
                0.0, 1.0, 0.25, 0.05,
                help="0 = copy the photo as closely as possible. Higher values blend in "
                "the symmetry / closed-loop design goals, giving new patterns inspired by the photo.",
            )

    with bar_hint:
        st.markdown(
            '<div class="ctrl-hint">Pick a grid size · fine-tune the search under Advanced</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# Hero header
# ------------------------------------------------------------
st.markdown(
    """
<div class="hero">
<div class="hero-floor"></div>
<div class="hero-text">
<span class="hero-kicker">Tradition · Geometry · Depth</span>
<h1>Kolam Studio</h1>
<p>Design a kolam and get a clean flat drawing you can download and follow.</p>
</div>
<div class="scene">
<div class="cube">
<div class="face front">🌸</div>
<div class="face back">✦</div>
<div class="face right">❀</div>
<div class="face left">✿</div>
<div class="face top">◈</div>
<div class="face bottom">❖</div>
</div>
<div class="cube-shadow"></div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "Upload a kolam for inspiration, or create a fresh pattern. "
    "You can save the finished design as a flat drawing guide."
)

# ------------------------------------------------------------
# Step 1: Upload and reference extraction
# ------------------------------------------------------------
step_title(1, "Start with your inspiration")
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Choose a kolam photograph",
            type=["png", "jpg", "jpeg"],
            help="A top-down photo with dark lines and a plain, light background works best.",
        )
        if uploaded_file is not None:
            reference_image = Image.open(uploaded_file).convert("RGB")
            st.session_state["ref_img"] = reference_image

            st.markdown("**Choose where the kolam is in the photo**")
            st.caption(
                "The converter does not recognise doors, flowers or the floor on its own. "
                "Frame only the kolam (the pink box) so it ignores everything else."
            )
            st.button(
                "✨ Auto-find the pattern area",
                on_click=auto_find_area,
                use_container_width=True,
                help="Heuristic: looks for thin line strokes and ignores big solid shapes.",
            )
            crop_x = st.slider("Left ↔ right (%)", 0, 100, (0, 100), key="crop_x")
            crop_y = st.slider("Top ↔ bottom (%)", 0, 100, (0, 100), key="crop_y")
            st.image(
                crop_overlay(reference_image, crop_x, crop_y),
                caption="Pink box = area used as the reference",
                use_container_width=True,
            )
            o1, o2 = st.columns(2)
            invert = o1.checkbox(
                "White lines on dark floor", value=False, key="opt_invert",
                help="Tick for rice-flour kolams drawn in white on a dark surface.",
            )
            ignore_blobs = o2.checkbox(
                "Ignore thick shapes", value=False, key="opt_blobs",
                help="Removes big solid areas (door, large flowers) and keeps thin strokes.",
            )
            auto_thr = st.checkbox(
                "Automatic line threshold", value=False, key="opt_auto_thr"
            )
            threshold = st.slider(
                "Image line threshold", 20, 220, 125, 5, disabled=auto_thr
            )
            if st.button("Prepare this reference", type="primary", use_container_width=True):
                crop_box = (crop_x[0] / 100, crop_y[0] / 100, crop_x[1] / 100, crop_y[1] / 100)
                if crop_box[2] - crop_box[0] < 0.03 or crop_box[3] - crop_box[1] < 0.03:
                    st.error("The selected area is too small. Widen the pink box.")
                else:
                    target, preview = image_to_target(
                        reference_image, n=grid_size, threshold=threshold,
                        crop_box=crop_box, invert=invert,
                        auto_threshold=auto_thr, ignore_blobs=ignore_blobs,
                    )
                    st.session_state["kolam_target"] = target
                    st.session_state["kolam_preview"] = preview
                    st.session_state["kolam_target_grid_size"] = grid_size
                    st.session_state.pop("kolam_result", None)
                    st.success("Your reference has been converted into a pattern grid.")
        else:
            st.markdown(
                "**No reference photo?** That's fine — you can create a new kolam below."
            )

with right:
    with st.container(border=True):
        st.markdown("### A good reference photo")
        st.markdown(
            '<span class="chip">Top-down angle</span>'
            '<span class="chip">Full pattern in frame</span>'
            '<span class="chip">Plain background</span>'
            '<span class="chip">No shadows</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="small-note" style="margin-top:12px">The photo converter is an '
            "image-processing prototype. It does not understand objects such as doors, "
            "flowers or floors; it only sees dark/light strokes inside the area you frame. "
            "It approximates them using the project's tile representation and is not a "
            "trained image-recognition model.</p>",
            unsafe_allow_html=True,
        )

if "kolam_target" in st.session_state:
    if st.session_state.get("kolam_target_grid_size") != grid_size:
        st.info("The grid size changed. Prepare the uploaded reference again to match it.")
    else:
        st.markdown("### Reference prepared")
        st.image(
            st.session_state["kolam_preview"],
            caption="Processed view used by the optimizer",
            width=320,
        )

# ------------------------------------------------------------
# Step 2: Create
# ------------------------------------------------------------
step_title(2, "Create your kolam")
col_ref, col_new = st.columns(2, gap="large")

reference_ready = (
    "kolam_target" in st.session_state
    and st.session_state.get("kolam_target_grid_size") == grid_size
)

with col_ref:
    with st.container(border=True):
        st.markdown("### 🖼️ Refine my reference")
        st.write("Optimize a pattern based on the photo you uploaded.")
        if st.button(
            "Create from my photo",
            type="primary",
            use_container_width=True,
            disabled=not reference_ready,
        ):
            with st.spinner("Working on your pattern…"):
                used_seed = next_seed(new_variation, random_seed)
                result = evolve_ga(
                    n=grid_size,
                    task="recreation",
                    target=st.session_state["kolam_target"],
                    population_size=population,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    seed=used_seed,
                    creativity=creativity,
                )
            st.session_state["kolam_result"] = result
            st.session_state["kolam_result_type"] = "Reference recreation"
        if not reference_ready:
            st.caption("Upload and prepare a reference photo to enable this option.")

with col_new:
    with st.container(border=True):
        st.markdown("### ✨ Create something new")
        st.write("Let the optimizer explore a fresh, balanced and symmetric design.")
        if st.button("Create a new kolam", type="primary", use_container_width=True):
            with st.spinner("Creating a new pattern…"):
                used_seed = next_seed(new_variation, random_seed)
                result = evolve_ga(
                    n=grid_size,
                    task="design",
                    population_size=population,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    seed=used_seed,
                )
            st.session_state["kolam_result"] = result
            st.session_state["kolam_result_type"] = "New design"

# ------------------------------------------------------------
# Step 3: Result (flat 2D kolam) and downloads
# ------------------------------------------------------------
if "kolam_result" in st.session_state:
    result = st.session_state["kolam_result"]
    pattern = result["pattern"]

    step_title(3, "Your kolam design")
    st.caption(
        f"{st.session_state.get('kolam_result_type', 'Kolam result')} · "
        f"Grid {grid_size} × {grid_size}"
    )

    result_col, detail_col = st.columns([1.25, 0.75], gap="large")

    with result_col:
        display_pattern(pattern, "Kolam Studio design")

    with detail_col:
        with st.container(border=True):
            st.markdown("### Design details")
            st.metric("Overall fitness", f"{result['fitness']:.3f}")
            st.caption(
                f"Seed used: {result.get('seed', '—')} · to repeat this exact pattern, "
                "turn off “New variation on every click” in Advanced and enter this seed."
            )
            components = result.get("components", {})
            m1, m2 = st.columns(2)
            m1.metric("Closed components", f"{components.get('closure', 0):.2f}")
            m2.metric("Loop score", f"{components.get('loops', 0):.2f}")
            m3, m4 = st.columns(2)
            m3.metric("Ink balance", f"{components.get('ink', 0):.2f}")
            m4.metric("Symmetry", f"{components.get('symmetry', 0):.2f}")

        with st.container(border=True):
            st.markdown("### Save your design")
            st.download_button(
                "Download flat kolam (PNG)",
                data=pattern_png(pattern),
                file_name="kolam_studio_design.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                "Download pattern grid (CSV)",
                data=matrix_csv(pattern),
                file_name="kolam_studio_tiles.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.info(
        "Use the flat image as a drawing guide. The fitness value "
        "measures the optimizer's design objective; it is not a percentage accuracy "
        "score or a guarantee that every stroke forms one continuous traditional kolam."
    )

# ------------------------------------------------------------
# Algorithm comparison (kept, tucked away)
# ------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Project evaluation — compare algorithms", expanded=False):
    st.write(
        "Run Genetic Algorithm, Hill Climbing and Random Search with the same "
        "evaluation budget. Results are optimization fitness scores, not model accuracy."
    )
    comparison_task = st.selectbox(
        "Evaluation task",
        ["Generative design", "Reference recreation"],
        key="comparison_task",
    )
    comparison_runs = st.slider("Independent runs", 1, 10, 3, key="comparison_runs")
    target_for_comparison = None
    if comparison_task == "Reference recreation":
        if reference_ready:
            target_for_comparison = st.session_state["kolam_target"]
        else:
            st.warning("Prepare a reference photo above before running recreation tests.")

    if st.button(
        "Run algorithm comparison",
        disabled=(comparison_task == "Reference recreation" and target_for_comparison is None),
    ):
        rows = []
        progress = st.progress(0)
        budget = population * generations
        for run_index in range(comparison_runs):
            run_seed = int(random_seed) + run_index
            if comparison_task == "Generative design":
                ga_result = evolve_ga(
                    n=grid_size, task="design", population_size=population,
                    generations=generations, mutation_rate=mutation_rate, seed=run_seed,
                )
                hc_result = hill_climb(
                    n=grid_size, task="design", evaluations=budget, seed=run_seed
                )
                rs_result = random_search(
                    n=grid_size, task="design", evaluations=budget, seed=run_seed
                )
            else:
                ga_result = evolve_ga(
                    n=grid_size, task="recreation", target=target_for_comparison,
                    population_size=population, generations=generations,
                    mutation_rate=mutation_rate, seed=run_seed,
                )
                hc_result = hill_climb(
                    n=grid_size, task="recreation", target=target_for_comparison,
                    evaluations=budget, seed=run_seed,
                )
                rs_result = random_search(
                    n=grid_size, task="recreation", target=target_for_comparison,
                    evaluations=budget, seed=run_seed,
                )

            for algorithm_name, algorithm_result in [
                ("Genetic Algorithm", ga_result),
                ("Hill Climbing", hc_result),
                ("Random Search", rs_result),
            ]:
                rows.append(
                    {
                        "Run": run_index + 1,
                        "Algorithm": algorithm_name,
                        "Fitness": algorithm_result["fitness"],
                    }
                )
            progress.progress((run_index + 1) / comparison_runs)

        results_df = pd.DataFrame(rows)
        summary = (
            results_df.groupby("Algorithm")["Fitness"]
            .agg(["mean", "std"])
            .fillna(0)
            .sort_values("mean", ascending=False)
        )
        st.markdown("### Individual run results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.markdown("### Average performance")
        st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)
        winner = summary["mean"].idxmax()
        st.success(f"Highest mean fitness in these runs: {winner}.")
        st.bar_chart(summary["mean"])
        st.download_button(
            "Download comparison results (CSV)",
            data=results_df.to_csv(index=False),
            file_name="kolam_algorithm_comparison.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Kolam Studio · A project prototype for exploring kolam pattern optimization")
