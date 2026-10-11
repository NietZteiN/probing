# Candidate figures for the combined paper

Three plots of existing verified results, prepared for visual comparison. They
are not yet inserted into the eight-page manuscript. Each has PDF and PNG
versions; error bars are saved pointwise 95% computation-cluster bootstrap
intervals. No new experiments or significance tests were run.

## 1. Showing source operations reduces wrong writes

[Preview](operation_controls.png) · [PDF](operation_controls.pdf)

This is the strongest main-paper addition. It compares values-only examples,
neutral text matched to the expression examples' token counts, numeric
elaboration, and the actual source expression. Both models use the same 285
matched computations and three demonstration sets (855 paired observations).
The operation examples `v = len(xs) = 2` leave much lower excess than the other
formats. Examples illustrate the output formats; `[notes]` stands for neutral
text. The plot measures the excess rate of writing the sum instead of the
computed length, subtracting each ordinary-name twin's rate.

Use this in Section 5 in place of the six-format table, retaining the full table
in the appendix. Its purpose is to show what the length and numeric controls
add to the basic format comparison. The controls do not isolate a single
responsible feature or establish a general solution for arbitrary code.

Suggested caption: **Code format controls on matched programs (95% intervals).**

## 2. The format contrast replicates across independently selected names

[Preview](fresh_names.png) · [PDF](fresh_names.pdf)

Every selected name is shown, for all three affected checkpoints and the
prespecified CodeGemma control. Names were selected by their meaning scores
before inspecting trace behavior; the 200 computations are new. Every program
is crossed with three selected names and three demonstration sets, giving
1,800 matched observations per model and format. Each plotted name pools its
600 observations, clustered over 200 computations. The shared 0–100-point
scale makes the model differences visible. The two formats share the cohort
within each panel. Original and replication cohorts are not paired.

Use this beside the replication paragraph, or as the leading appendix figure
if main-text space is tight. It directly addresses dependence on the original
names and program sample. Zero estimates and zero-width bootstrap intervals
record this sample, rather than establish zero population rates.

Suggested caption: **Format effects across three fresh sum names (95% intervals).**

## 3. Correct calculations can precede wrong final answers

[Preview](correct_calculation.png) · [PDF](correct_calculation.pdf)

Within each naming condition, the original input and completely correct
generated calculation remain fixed. The two final digit-answer queries differ
in whether they repeat the requested variable's name. The left panel shows
12.8 versus 0.5 points of added name-suggested answers for OLMo-2-1B-Instruct;
the right panel makes the accompanying ordinary-name accuracy change visible.
The cohort contains 368 paired observations over 205 computations. The query
without a name asks for `the original requested value`; the named query asks
for the originally queried variable.

Use this as a compact bridge between the arithmetic behavior and internal
readouts. It illustrates final-answer failures after correct calculations,
conditional on those successful calculations. It does not estimate the effect
of making calculations correct, or identify name repetition as an isolated
remedy. Neither Llama model produces name-suggested answers in these formats;
all four formats and all three checkpoints remain in the appendix table.

Suggested caption: **Final digit answers after correct arithmetic calculations (95% intervals).**

## Reproduction

Run `scripts/92_figure_candidates.py` with a Python environment containing
Matplotlib and NumPy. It reads only saved summaries in the sibling probing and
codecue repositories. `provenance.json` records SHA256 source hashes and every
plotted interval in its original proportional units. PDF fonts are embedded.
