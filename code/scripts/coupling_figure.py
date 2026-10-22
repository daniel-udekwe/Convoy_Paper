"""Peak link force vs link stiffness, from results/coupling.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy import plotstyle as ps  # noqa: E402
res = json.loads((ROOT / "results" / "coupling.json").read_text())
C = {"ff": ps.MAGENTA, "ff_hitch": ps.MAGENTA, "gap": ps.GREEN, "gap_hitch": ps.BLUE}
LBL = {"ff": "PI + feedforward, CoM gap", "ff_hitch": "PI + feedforward, hitch length",
       "gap": "Gap NMPC, CoM gap", "gap_hitch": "Gap NMPC, hitch length (proposed)"}
MK = {"ff": "^", "ff_hitch": "v", "gap": "D", "gap_hitch": "o"}
ps.apply()

W = 450 * 9.81  # cart weight, N
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for ctrl in C:
    rows = sorted([r for r in res if r["ctrl"] == ctrl and r["k"] > 0], key=lambda r: r["k"])
    hitch_pi = ctrl == "ff_hitch"          # same hue as "ff": dashed, open markers
    ax.plot([r["k"] for r in rows], [r["force_peak"] for r in rows], color=C[ctrl],
            marker=MK[ctrl], ms=7, lw=2, ls="--" if hitch_pi else "-",
            mfc="white" if hitch_pi else C[ctrl], label=LBL[ctrl])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Link stiffness (N/m)")
ax.set_ylabel("Peak link force (N)")
ax.axhline(0.1 * W, color=ps.GREY, lw=1, ls="--")
ax.text(6e4, 0.1 * W * 1.08, "10% of cart weight", fontsize=8, color=ps.MUTED)
ax.set_title("GNSS-like path: force the coupling must carry", loc="left")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 1))
out = ROOT / "results" / "figures" / "coupling_force.png"
fig.savefig(out)
print("saved", out)
