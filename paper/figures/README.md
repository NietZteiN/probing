# Rebuilding the figures

Run from the repository root after sourcing `scripts/env.sh`, using the probe Python
environment. These scripts use existing data and run on the CPU; they do not load models.

- `python scripts/76_story_figures.py`: main response-format comparison on two-operation
  tasks. The generated-calculation calibration plot remains available as a historical
  asset; the submission reports those estimates once, in the validation table.
- `python scripts/83_submission_figures.py`: main patching plot with the two controls,
  plus the appendix table of all-layer rates and denominators. The companion JSON records
  the plotted values and source summary.
- `python scripts/66_figure_dump.py`: explicit selection of two supporting appendix
  plots: the intermediate-value token sweep and queried-variable span-patching grid.
  No other files enter the submission automatically. Historical plots remain in the
  repository for inspection and reproduction.
- `python scripts/79_round5_tables.py` and `python scripts/82_round6_tables.py`:
  consolidated task/probe controls, correlation sensitivity and generated-chain validation.

Then run `make paper-submission`. Keep figures to axes, model labels, legends and short
scope labels; put interpretation in the paper text. Captions specify comparisons and
interval conventions. Preserve the distinction between probe accuracy and generated-answer
accuracy, and between removal of a name error and recovery of the correct answer.

The appendix is maintained in `paper/appendix.tex`. It keeps methods, full behavioral
results and controls, the two selected technical plots, and calibration/error-count tables.
Repeated example plots, the full figure gallery and derived-difference tables are excluded
from the submission PDF. The source summaries and historical assets remain available.
