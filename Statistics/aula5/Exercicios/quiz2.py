# Reproduce the partitioned scatter plot and the decision tree, and verify correctness.

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1) Define synthetic coordinates just to place points in the right regions.
#    The prices are the labels; the x/y are Distance and Criminalidade (illustrative).
# -------------------------
# Regions:
# R1: x < 3
# R4: x >= 3 and y < 0.30
# R2: 3 <= x < 6 and y >= 0.30
# R3: x >= 6 and y >= 0.30

points = [
    # R1 (left): values {9, 7, 11}
    {"x": 0.5, "y": 0.65, "price": 9},
    {"x": 1.5, "y": 0.92, "price": 7},
    {"x": 2.2, "y": 0.55, "price": 11},
    # R2 (mid-top): values {5, 4}
    {"x": 4.0, "y": 0.55, "price": 5},
    {"x": 5.5, "y": 0.90, "price": 4},
    # R3 (right-top): values {17, 15, 13}
    {"x": 6.4, "y": 0.35, "price": 17},
    {"x": 8.1, "y": 0.99, "price": 15},
    {"x": 8.6, "y": 0.62, "price": 13},
    # R4 (bottom band): values {4, 2}
    {"x": 5.2, "y": 0.26, "price": 4},
    {"x": 7.4, "y": 0.14, "price": 2},
]

# Helper to classify region
def region_id(p):
    if p["x"] < 3:
        return "R1"
    if p["y"] < 0.30:
        return "R4"
    if p["x"] < 6:
        return "R2"
    return "R3"

# Group points by region
from collections import defaultdict
regions = defaultdict(list)
for p in points:
    regions[region_id(p)].append(p["price"])

# Expected composition
expected = {
    "R1": [9, 7, 11],
    "R2": [5, 4],
    "R3": [17, 15, 13],
    "R4": [4, 2],
}

# Verify exact membership
assert all(sorted(regions[r]) == sorted(v) for r, v in expected.items()), "Region memberships do not match expectation."

# Compute leaf predictions (means) and SSE per region
def mean_sse(vals):
    m = np.mean(vals)
    sse = float(np.sum((np.array(vals) - m) ** 2))
    return float(m), sse

leaf_stats = {r: mean_sse(vals) for r, vals in regions.items()}

# Global R^2 on training
y_all = np.array([p["price"] for p in points])
ybar = y_all.mean()
sst = float(np.sum((y_all - ybar) ** 2))
sse_total = float(sum(sse for _, sse in leaf_stats.values()))
r2 = 1.0 - sse_total / sst

# -------------------------
# 2) Plot the partitioned scatter (reproduction)
# -------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Points and labels
xs = [p["x"] for p in points]
ys = [p["y"] for p in points]
ax.scatter(xs, ys)

for p in points:
    ax.text(p["x"] + 0.05, p["y"] + 0.02, f"{p['price']}", fontsize=10)

# Partition lines
ax.plot([3, 3], [0, 1])                     # x = 3 (full height)
ax.plot([3, 9], [0.30, 0.30])               # y = 0.30 (from x>=3)
ax.plot([6, 6], [0.30, 1])                  # x = 6 (only above y>=0.3)

# Region predictions (means)
ax.text(1.3, 0.90, "ŷ = 9", fontsize=11)
ax.text(4.2, 0.85, "ŷ = 4.5", fontsize=11)
ax.text(7.7, 0.92, "ŷ = 15", fontsize=11)
ax.text(6.6, 0.08, "ŷ = 3", fontsize=11)

ax.set_xlim(0, 9)
ax.set_ylim(0, 1.05)
ax.set_title("Preço do metro quadrado")
ax.set_xlabel("Distância")
ax.set_ylabel("Criminalidade")

plt.tight_layout()
out1 = "/Users/akatsurada/Documents/INSPER/Statistics/aula5/Exercicios/aquiz2.png"
plt.savefig(out1, dpi=160)
plt.close(fig)

# -------------------------
# 3) Draw a simple decision tree diagram
# -------------------------
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.axis("off")

# Node positions (x, y in axes fraction coordinates)
positions = {
    "root": (0.5, 0.92),
    "left": (0.20, 0.60),
    "right": (0.80, 0.60),
    "right_bottom": (0.60, 0.30),
    "right_top": (0.90, 0.30),
    "leaf_L1": (0.20, 0.35),
    "leaf_L4": (0.60, 0.05),
    "leaf_L2": (0.60, 0.55),
    "leaf_L3": (0.90, 0.55),
}

# Draw nodes (text boxes)
def node(ax, key, text):
    x, y = positions[key]
    ax.text(x, y, text, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", pad=0.4))

# Edges
def edge(ax, a, b, label=None):
    xa, ya = positions[a]
    xb, yb = positions[b]
    ax.annotate("", xy=(xb, yb+0.03), xytext=(xa, ya-0.03),
                arrowprops=dict(arrowstyle="-"))  # simple line
    if label:
        ax.text((xa+xb)/2, (ya+yb)/2 + 0.03, label, ha="center", fontsize=10)

# Root and first split
node(ax2, "root", "Raiz\nSplit: Distância < 3 ?")
edge(ax2, "root", "left", "Sim")
edge(ax2, "root", "right", "Não")

# Left leaf
node(ax2, "left", "Folha L1\nŷ = 9\n{9, 7, 11}")

# Right: split by Criminalidade
node(ax2, "right", "Split: Criminalidade < 0,30 ?")
edge(ax2, "right", "right_bottom", "Sim")
edge(ax2, "right", "right_top", "Não")

# Bottom leaf
node(ax2, "right_bottom", "Folha L4\nŷ = 3\n{4, 2}")

# Top: split by Distância < 6 ?
node(ax2, "right_top", "Split: Distância < 6 ?")
edge(ax2, "right_top", "leaf_L2", "Sim")
edge(ax2, "right_top", "leaf_L3", "Não")

# Leaves top
node(ax2, "leaf_L2", "Folha L2\nŷ = 4,5\n{5, 4}")
node(ax2, "leaf_L3", "Folha L3\nŷ = 15\n{17, 15, 13}")

# Footer with metrics
ax2.text(0.02, 0.02, f"SSE total = {sse_total:.2f} | SST = {sst:.2f} | R² (treino) = {r2:.3f}", fontsize=10)

plt.tight_layout()
out2 = "/Users/akatsurada/Documents/INSPER/Statistics/aula5/Exercicios/quiz2.png"
plt.savefig(out2, dpi=160)
plt.close(fig2)

print("Arquivos gerados:")
print(f"1) Partições e pontos: {out1}")
print(f"2) Árvore de decisão: {out2}")
print("\nVerificações automáticas:")
print(" - Composição por região:", {k: sorted(v) for k, v in regions.items()})
print(" - Médias por folha:", {k: round(leaf_stats[k][0], 3) for k in sorted(leaf_stats)})
print(" - SSE por folha:", {k: round(leaf_stats[k][1], 3) for k in sorted(leaf_stats)})
print(f" - SSE total: {sse_total:.2f}")
print(f" - Média global (ȳ): {ybar:.2f} | SST: {sst:.2f} | R²: {r2:.3f}")
