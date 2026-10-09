#!/usr/bin/env python
"""Concise patching controls from saved summaries; no model loading."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = [
    ("main", "lure_removed", "Ordinary name: errors removed", "#2B4A9E", "-"),
    ("ctl_lure", "lure_removed", "Other misleading name: errors removed", "#B5321F", "--"),
    ("ctl_word", "damage", "Word control: correct answers changed", "#157A55", ":"),
]


def draw(plt, model, role, stem, title):
    source = ROOT / f"results/summary/{model}/L3/direct/patching.json"
    data = json.loads(source.read_text())
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    fig.subplots_adjust(left=.18, right=.99, bottom=.20, top=.66)
    payload = {"source": str(source.relative_to(ROOT)), "model": model, "role": role,
               "regime": "direct", "series": {}}
    last = 0
    for contrast, metric, label, color, style in SERIES:
        cells = data[f"{contrast}@{role}"]
        points = sorted((int(layer[1:]), cell["anspre"][metric])
                        for layer, cell in cells.items()
                        if layer.startswith("L") and cell.get("anspre", {}).get(metric) is not None)
        assert points, (model, contrast)
        xs, ys = zip(*points)
        last = max(last, max(xs))
        ax.plot(xs, [100*y for y in ys], style, color=color, lw=1.1, label=label)
        all_rate = cells.get("ALL", {}).get("anspre", {}).get(metric)
        if all_rate is not None:
            ax.plot(max(xs)+3, 100*all_rate, "o", color=color, ms=3)
        base = cells["ALL"]["anspre"]
        payload["series"][contrast] = {"metric": metric, "layers": points, "all": all_rate,
            "pairs": base["n"], "baseline_name_errors": base["n_lure_err"],
            "baseline_correct": round(base["n"]*base["base_acc"])}
    ax.set_xticks([0, 10, 20, last+3], ["0", "10", "20", "All"])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Layer replaced", fontsize=8)
    ax.set_ylabel("Answers changed (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.suptitle(title, fontsize=9, y=.99)
    fig.legend(loc="upper left", bbox_to_anchor=(.11, .94), fontsize=7,
               frameon=False, handlelength=1.8, labelspacing=.2)
    dest = ROOT / "paper/figures"
    for ext in ["pdf", "png"]:
        fig.savefig(dest / f"{stem}.{ext}", dpi=180)
    (dest / f"{stem}_data.json").write_text(json.dumps(payload, indent=2)+"\n")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    draw(plt, "llama32-3b", "v1", "story_patching", "Llama-3.2-3B · answer only")
    lines = [r'\begin{tabular}{@{}llrrrrrr@{}}', r'\toprule',
             r'Model & Variable & Pairs & Name errors & Correct & Neutral & Alternative & Word damage \\',
             r'\midrule']
    for model, role, name, variable in [('llama32-3b','v1','Llama-3.2-3B','queried'),
                                       ('llama31-8b','v2','Llama-3.1-8B','intermediate')]:
        d = json.loads((ROOT / f'results/summary/{model}/L3/direct/patching.json').read_text())
        m,w,a = [d[f'{k}@{role}']['ALL']['anspre'] for k in ['main','ctl_word','ctl_lure']]
        assert m['n_lure_err']==a['n_lure_err']
        row = [name, variable, str(m['n']), str(m['n_lure_err']),
               str(round(w['n']*w['base_acc'])), f"{100*m['lure_removed']:.1f}",
               f"{100*a['lure_removed']:.1f}", f"{100*w['damage']:.1f}"]
        lines.append(' & '.join(row)+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (ROOT / 'paper/tables/patching_controls.tex').write_text('\n'.join(lines)+'\n')


if __name__ == "__main__":
    main()
