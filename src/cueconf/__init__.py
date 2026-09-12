"""cueconf: cue-conflict (Stroop) tests of chain-of-thought arithmetic.

Layout (each module is importable on its own; nothing here imports torch at package level so
the generator and prompt tests run on the login node, which caps virtual memory at 8 GB):

    words      number words, neutral words, letters
    generator  Kudo et al. (2026) task levels + naming conditions + matched twins
    prompts    few-shot rendering for the CoT and direct regimes; probe positions P1-P5
    tokcheck   rule 4: token-count matching per tokenizer (needs a tokenizer; run in a job)
    runner     behaviour + hidden-state caching + forced-decoding logits (GPU)
    probes     neutral-only linear probes, margins, lure mass, selectivity (GPU)
    patching   residual-stream activation patching at name positions (GPU)
    stats      bootstrap CIs, clustered logistic regression, FDR, crossover step
"""
__version__ = "0.1.0"
