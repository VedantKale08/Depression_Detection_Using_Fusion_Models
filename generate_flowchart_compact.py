import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
import matplotlib.path as mpath

FIG_W, FIG_H = 11.5, 7.5
DPI = 200

# Color palette suitable for academic papers
C_INPUT      = "#E3F2FD"   # light blue for data/extraction
C_PROCESS    = "#FFF9C4"   # light yellow for processing
C_MODEL      = "#FFE082"   # amber for deep learning
C_OUTPUT     = "#FFB300"   # dark amber for output
C_BORDER     = "#263238"   
C_ARROW      = "#37474F"   

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.05)
ax.axis("off")
fig.patch.set_facecolor("white")

def draw_cylinder(cx, cy, w, h, color):
    rx = w / 2
    ry = h * 0.25
    x, y = cx - rx, cy - h/2
    # body
    ax.add_patch(FancyBboxPatch((x, y + ry), w, h - 2*ry, boxstyle="square,pad=0", facecolor=color, edgecolor="none", zorder=2))
    # top/bottom ellipses
    ax.add_patch(mpatches.Ellipse((cx, y + ry), w, 2*ry, facecolor=color, edgecolor=C_BORDER, lw=1.5, zorder=4))
    ax.plot([x, x], [y + ry, y + h - ry], color=C_BORDER, lw=1.5, zorder=3)
    ax.plot([x+w, x+w], [y + ry, y + h - ry], color=C_BORDER, lw=1.5, zorder=3)
    ax.add_patch(mpatches.Ellipse((cx, y + h - ry), w, 2*ry, facecolor=color, edgecolor=C_BORDER, lw=1.5, zorder=3))

def draw_rect(cx, cy, w, h, color, radius=0.03):
    x, y = cx - w/2, cy - h/2
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=color, edgecolor=C_BORDER, lw=1.5, zorder=2))
    
def draw_parallelogram(cx, cy, w, h, color):
    skew = h * 0.35
    x, y = cx - w/2, cy - h/2
    path = mpath.Path([
        (x + skew, y),
        (x + w, y),
        (x + w - skew, y + h),
        (x, y + h),
        (x + skew, y)
    ])
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor=C_BORDER, lw=1.5, zorder=2))

def draw_arrow(x1, y1, x2, y2, path=None):
    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=C_ARROW, lw=1.8, zorder=1)
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), 
                    arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.8, mutation_scale=16), zorder=1)
    else:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                    arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.8, mutation_scale=16), zorder=1)

boxes = {
    "in": {"cx": 0.5, "cy": 0.95, "w": 0.20, "h": 0.10, "type": "cylinder", "title": "Input Video", "sub": ".mp4 Clinical Interview", "c": C_INPUT},
    
    "face": {"cx": 0.18, "cy": 0.75, "w": 0.28, "h": 0.11, "type": "rect", "title": "Face Feat. Extractor", "sub": "OpenFace → CLNF (136-dim)", "c": C_INPUT},
    "audio": {"cx": 0.50, "cy": 0.75, "w": 0.28, "h": 0.11, "type": "rect", "title": "Audio Feat. Extractor", "sub": "FFmpeg+Librosa → COVAREP (74-dim)", "c": C_INPUT},
    "emo_ext": {"cx": 0.82, "cy": 0.75, "w": 0.28, "h": 0.11, "type": "rect", "title": "Emotion Vector Gen.", "sub": "Audio CNN + RoBERTa (Text)", "c": C_INPUT},
    
    "merge": {"cx": 0.34, "cy": 0.55, "w": 0.32, "h": 0.10, "type": "rect", "title": "Temporal Alignment", "sub": "Merge CLNF+COVAREP → ℝ^(T×210)", "c": C_PROCESS},
    "emo_vec": {"cx": 0.82, "cy": 0.55, "w": 0.28, "h": 0.10, "type": "rect", "title": "Aux. Affective Context", "sub": "Emotion Vector e ∈ ℝ¹⁴", "c": C_PROCESS},
    
    "norm": {"cx": 0.34, "cy": 0.35, "w": 0.32, "h": 0.10, "type": "rect", "title": "Windowing & Norm.", "sub": "30s Chunks → X_norm = (X-μ)/σ", "c": C_PROCESS},
    
    "lstm": {"cx": 0.34, "cy": 0.15, "w": 0.32, "h": 0.12, "type": "rect", "title": "Bi-LSTM + Temporal Attn.", "sub": "H = BiLSTM(X_norm)\nContext c = Σ α_t·h_t ∈ ℝ¹²⁸", "c": C_MODEL},
    
    "fusion": {"cx": 0.70, "cy": 0.25, "w": 0.32, "h": 0.11, "type": "rect", "title": "Late Fusion & FC Head", "sub": "f = [c ; e] ∈ ℝ¹⁴² → FC → σ(ŷ)", "c": C_MODEL},
    
    "out": {"cx": 0.70, "cy": 0.06, "w": 0.26, "h": 0.10, "type": "paral", "title": "Risk Prediction", "sub": "LOW | MED | HIGH", "c": C_OUTPUT}
}

for k, b in boxes.items():
    if b["type"] == "cylinder":
        draw_cylinder(b["cx"], b["cy"], b["w"], b["h"], b["c"])
    elif b["type"] == "rect":
        draw_rect(b["cx"], b["cy"], b["w"], b["h"], b["c"])
    elif b["type"] == "paral":
        draw_parallelogram(b["cx"], b["cy"], b["w"], b["h"], b["c"])
        
    y_offset = 0.015 if b["h"] < 0.11 else 0.018
    if "\n" in b["sub"]: y_offset = 0.02
    
    ax.text(b["cx"], b["cy"] + y_offset, b["title"], ha="center", va="center", fontsize=11, fontweight="bold", color="#111")
    ax.text(b["cx"], b["cy"] - y_offset, b["sub"], ha="center", va="center", fontsize=9, color="#333", style="italic")

# Input -> Extractors
draw_arrow(0.5, 0.90, 0.18, 0.81, path=[(0.5, 0.90), (0.18, 0.85), (0.18, 0.81)])
draw_arrow(0.5, 0.90, 0.50, 0.81)
draw_arrow(0.5, 0.90, 0.82, 0.81, path=[(0.5, 0.90), (0.82, 0.85), (0.82, 0.81)])

# Extractors -> Merge
draw_arrow(0.18, 0.69, 0.28, 0.60, path=[(0.18, 0.69), (0.18, 0.65), (0.28, 0.60)])
draw_arrow(0.50, 0.69, 0.40, 0.60, path=[(0.50, 0.69), (0.50, 0.65), (0.40, 0.60)])

# Extractor -> Emo Vec
draw_arrow(0.82, 0.69, 0.82, 0.60)

# Merge -> Norm -> LSTM
draw_arrow(0.34, 0.50, 0.34, 0.40)
draw_arrow(0.34, 0.30, 0.34, 0.21)

# LSTM -> Fusion
draw_arrow(0.50, 0.15, 0.54, 0.25, path=[(0.50, 0.15), (0.52, 0.15), (0.52, 0.25), (0.54, 0.25)])

# Emo Vec -> Fusion
draw_arrow(0.82, 0.50, 0.70, 0.305, path=[(0.82, 0.50), (0.82, 0.40), (0.70, 0.40), (0.70, 0.305)])

# Fusion -> Out
draw_arrow(0.70, 0.195, 0.70, 0.11)

out_file = "flowchart_compact.png"
fig.savefig(out_file, bbox_inches="tight")
print(f"Saved -> {out_file}")
