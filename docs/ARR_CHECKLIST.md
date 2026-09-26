# ARR Responsible NLP checklist — draft answers

*Drafted 2026-09-25 against the paper as of this commit. These are drafts for the author to
check and paste into the submission form; the question wording below is paraphrased, so match
each answer to the live form (OpenReview) before submitting. Section names refer to
`paper/main.tex`; appendix letters are left out because they move when sections are added.*

## A. Every submission

**A1. Limitations described?** Yes. *Limitations* (unnumbered section after the Conclusion).

**A2. Potential risks discussed?** No, with justification: the study is a diagnostic of
model behaviour on synthetic arithmetic problems. It releases no model, no system and no data
about people, and the misleading names it studies are single English number words, which are
too weak to use as an attack. The strongest risk-adjacent point, that correct behaviour does
not imply a clean internal state, is itself a finding reported in the paper.
*(Author's call: answer "Yes" instead if you add a sentence to Limitations.)*

## B. Scientific artifacts — Yes

**B1. Creators of artifacts cited?** Yes. Kudo et al. (task format, levels, probe recipe) in
*Setup* ("What we take from Kudo et al.") and throughout; the model families (Llama 3, Gemma 3,
OLMo 2, Llama-Nemotron) and the software (PyTorch, Transformers, scikit-learn, statsmodels) in
the appendix paragraph *Data, compute and software*.

**B2. Licence or terms of use discussed?** Yes, briefly: *Data, compute and software* says
each model is used under its release licence for research. Verified on the Hugging Face model
cards (2026-09-25): Llama 3.2 Community License (Llama-3.2-3B and -Instruct), Llama 3.1
Community License (Llama-3.1-8B and -Instruct, and the three LoRA/merge variants built on it),
Gemma Terms of Use (Gemma-3-4B pt and it, Gemma-3-12B-it), Apache 2.0 (both OLMo 2 models), and
the NVIDIA Open Model License for Llama-3.1-Nemotron-Nano-8B (its card also points to the Llama
3.1 Community License, "Built with Llama"). All permit non-commercial research use. Kudo et
al.'s repository has no licence (all rights reserved by default); the paper reimplements their
format and recipe from the paper (*Setup*), and the local clone is gitignored, so none of their
code is redistributed.

**B3. Use consistent with intended use?** Yes. All models are used for non-commercial research
inference, which every licence above permits. The released artifacts (generator, word lists,
matched sets) are intended for research on the same question.

**B4. Personal or offensive content checked?** N/A. All data is generated: digits, operators,
and single-token English words (number words and a fixed list of neutral nouns). It contains
no personal information and no free text.

**B5. Documentation of artifacts?** Yes. *Setup* (task, naming conditions, generation rules,
regimes) and the appendix *Tokenizer checks* (word lists, tokenizer rule); the code and word
lists are released with the paper.

**B6. Statistics of data (splits, sizes)?** Yes. The appendix paragraph *Data, compute and
software*: 2,000 matched sets per level, each rendered in every condition, and 10,000
separately generated probe-training problems; probes are trained on the neutral split only
(*Setup*, "Probes"); three demonstration sets (*Setup*, "Demonstration sets").

## C. Computational experiments — Yes

**C1. Model size, compute budget, infrastructure?** Yes. Model sizes are in the model names
(*Setup*, "Models"); the appendix paragraph *Data, compute and software* gives the GPU budget (an upper bound from
SLURM accounting, `scripts/70_compute_budget.py`) and the GPUs used.

**C2. Experimental setup and hyperparameters?** Yes. *Setup* ("Probes", "Patching",
"Regimes") and the appendix *Probe recipe and patching scope*. The probe recipe is Kudo et
al.'s (single linear layer, SGD, fixed learning rate and epochs); nothing is tuned on the test
conditions, and the contrasts and thresholds were preregistered (`PREREGISTRATION.md`).

**C3. Descriptive statistics (error bars, runs)?** Yes. Every effect is a mean over three
demonstration sets with a 95% cluster-bootstrap interval over matched sets, and a claim needs
the same sign in all three sets (Table 1 caption, *Setup*). Every probe is trained with three seeds (*Data, compute and software*). The appendix adds a mixed-effects refit and equivalence bounds.

**C4. Existing packages reported?** Yes. The appendix paragraph *Data, compute and software* names PyTorch,
Transformers, scikit-learn and statsmodels. Versions used: torch 2.11.0 (CUDA 12.9),
transformers 5.14.1, scikit-learn 1.9.1, numpy 2.3.5, scipy 1.18.0.

## D. Human annotators or participants — No

No crowdworkers, annotators or participants. (The introduction cites a prior human study in
the third person; this paper collects no human data.)

## E. AI assistants

**E1. Did you use AI assistants?** Yes. *(Author's call on the exact wording and scope.)*
Draft: "We used Claude (Anthropic), through Claude Code, to write and run experiment and
analysis code, run the cluster jobs, and draft and edit parts of the text. All results were
generated by the released scripts; the authors checked the code, the numbers and the text."
