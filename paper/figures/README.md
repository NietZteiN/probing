# Rebuilding the figures

Run from the repository root after sourcing `scripts/env.sh`, using the probe Python
environment. These scripts use existing data and run on the CPU; they do not load models.

- `python scripts/76_story_figures.py`: response-format comparison on two-operation tasks,
  plus the appendix's generated-calculation readout checks. Companion JSON files record every plotted
  estimate and confidence interval; no new aggregate is fitted.
- `python scripts/66_figure_dump.py`: full-size appendix figures with concise research captions. Main story figures are already full width and are not
  repeated in the dump. Existing technical figure generators remain available.

Then run `make paper-submission`. Keep figures to axes, model labels, legends and short
scope labels; put interpretation in the paper text. Captions specify comparisons and
interval conventions. Preserve the distinction between probe accuracy and generated-answer
accuracy, and between removal of a name error and recovery of the correct answer.

The calibration figure is included by `readout_check.tex` in the generated-neutral probe
appendix. `scripts/82_round6_tables.py` preserves that placement when tables are rebuilt.
