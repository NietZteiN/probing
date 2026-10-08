#!/usr/bin/env python
"""Every generated figure at full size, one per page, in an appendix that does not count against
the page limit. Full-width figure floats preserve the ACL two-column portrait layout.

    python scripts/66_figure_dump.py

Writes paper/tables/figure_dump.tex, which main.tex \\input{}s after the appendix sections.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import PROJECT_ROOT  # noqa: E402

PAPER = PROJECT_ROOT / "paper"

# name -> caption. Anything on disk but not listed is dumped last with a generic caption.
CAPTIONS = {
    "fig1_master_L3": "Figure~1 at full size.",
    "fig2_tokens_llama32-3b_L3_cot_v2": "Figure~2 at full size: per-token probes, chain-of-thought regime, "
        "intermediate variable.",
    "fig2_tokens_llama32-3b_L3_cot_v1": "Per-token probes, chain-of-thought regime, queried variable.",
    "fig2_tokens_llama32-3b_L3_direct_v2": "Per-token probes, direct regime, intermediate variable.",
    "fig4_errors_llama32-3b_L3_cot_v1": "Probing accuracy split by whether the model answers the problem "
        "correctly on its own, with the layer sweep for each half.",
    "fig5_kudo3_llama32-3b_L3_cot_v1_lureerr_i0": "Top-1 probe prediction at every layer and token, on a problem "
        "answered with the name-suggested value. Layout after \\citeauthor{kudo2026faithful}'s Figure~3.",
    "fig5_kudo3_llama32-3b_L3_cot_v1_lureerr_i1": "A second problem answered with the name-suggested value.",
    "fig5_kudo3_llama32-3b_L3_cot_v1_lureerr_i2": "A third problem answered with the name-suggested value.",
    "fig5_kudo3_llama32-3b_L3_cot_v1_err_i0": "As above, on a problem answered wrongly for another reason.",
    "fig5_kudo3_llama32-3b_L3_cot_v1_err_i1": "A second such problem.",
    "fig5_kudo3_llama32-3b_L3_cot_v2_err_i0": "As above, for the intermediate variable. Red is the model's final answer, "
        "which is the queried variable's value, so a cell that matches it is not a readout of this variable.",
    "fig5_kudo3_llama32-3b_L3_cot_v2_err_i1": "A second problem, intermediate variable.",
    "kudo_fig2_llama32-3b_L3_cot_v2": "Accuracy heatmaps by token and layer for the neutral, congruent and "
        "incongruent conditions, in the layout of \\citeauthor{kudo2026faithful}'s Figure~2.",
    "kudo_fig2_llama32-3b_L3_direct_v2": "The same heatmaps in the direct regime.",
    "kudo_fig5_llama32-3b_L3_cot_v2": "Span-by-layer-window patching grids, three sources (the error-removal panel is "
        "empty: with CoT there are too few name errors to draw), "
        "after \\citeauthor{kudo2026faithful}'s Figures~5--6.",
    "kudo_fig5_llama32-3b_L3_direct_v2": "The same grids in the direct regime.",
    "fig3_patching_llama32-3b_L3_v2": "Name errors removed by patching, by layer, with both controls.",
    "kudo_fig2_llama32-3b_L3_cot_v1": "Accuracy heatmaps as above, chain-of-thought regime, queried variable.",
    "kudo_fig2_llama32-3b_L3_direct_v1": "The same heatmaps in the direct regime, queried variable.",
    "kudo_fig5_llama32-3b_L3_cot_v1": "Span-by-layer-window patching grids, chain-of-thought regime, queried variable; "
        "the error-removal panel is empty for the same reason.",
    "kudo_fig5_llama32-3b_L3_direct_v1": "The same grids in the direct regime, queried variable.",
    "fig3_patching_llama32-3b_L3_v1": "Name errors removed by patching, by layer, with both controls, queried variable.",
    "fig_own_chain_llama32-3b_L3_cot_v1": "The own-chain probe figure at full size.",
    "fig3_patching_llama31-8b_L3_v2": "Llama-3.1-8B, name errors removed by patching, by layer, with both controls, "
        "direct regime, intermediate variable.",
    "kudo_fig2_llama31-8b_L3_cot_v1": "Llama-3.1-8B, chain-of-thought regime, queried variable: the accuracy "
        "heatmaps of the Llama-3.2-3B figures. The value is decodable from the equation that restates the "
        "variable's definition, before the chain writes it.",
    "kudo_fig2_llama31-8b_L3_cot_v2": "Llama-3.1-8B, chain-of-thought regime, intermediate variable.",
    "kudo_fig2_llama31-8b_L3_direct_v2": "Llama-3.1-8B, direct regime, intermediate variable.",
    "kudo_fig5_llama31-8b_L3_cot_v2": "Llama-3.1-8B, span-by-layer-window patching grids, chain-of-thought "
        "regime. The error-removal panel is empty: none of the 500 patched problems is answered with the name-suggested value.",
    "kudo_fig5_llama31-8b_L3_direct_v2": "Llama-3.1-8B, the same grids in the direct regime.",
    "kudo_fig2_llama31-8b_L3_direct_v1": "Llama-3.1-8B, direct regime, queried variable.",
    "kudo_fig5_llama31-8b_L3_cot_v1": "Llama-3.1-8B, patching grids, chain-of-thought regime, queried variable; "
        "the error-removal panel is empty for the same reason.",
    "kudo_fig5_llama31-8b_L3_direct_v1": "Llama-3.1-8B, patching grids, direct regime, queried variable.",
}
# Gemma token figures already appear in the dedicated probe-extension appendix.
SKIP = ("fig2_margin", "kudo_fig3_", "fig2_tokens_gemma3-4b-it", "kudo_fig2_gemma3-4b-it")


def main() -> int:
    figs = sorted(p.stem for p in (PAPER / "figures").glob("*.pdf"))
    figs = [f for f in figs if not any(f.startswith(s) for s in SKIP)]
    ordered = [f for f in CAPTIONS if f in figs] + [f for f in figs if f not in CAPTIONS]
    out = ["% generated by scripts/66_figure_dump.py",
           "\\clearpage", "\\section{Every figure at full size}", "\\label{app:figs}",
           "Each figure below is reproduced as a full-width figure on a portrait page, "
           "within the ACL margins.", ""]
    for stem in ordered:
        cap = CAPTIONS.get(stem, f"\\texttt{{{stem.replace('_', chr(92) + '_')}}}.")
        box = "width=\\textwidth,height=0.85\\textheight,keepaspectratio"
        out += ["\\begin{figure*}[p]\\centering",
                f"\\includegraphics[{box}]{{figures/{stem}.pdf}}",
                f"\\caption{{{cap}}}\\label{{fig:dump-{stem.replace('_', '-')}}}",
                "\\end{figure*}", "\\clearpage", ""]
    (PAPER / "tables" / "figure_dump.tex").write_text("\n".join(out))
    print(f"wrote figure_dump.tex with {len(ordered)} figures "
          f"({sum(1 for f in ordered if f in CAPTIONS)} captioned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
