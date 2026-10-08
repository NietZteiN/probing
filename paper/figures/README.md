# Rebuilding the main figures

Run from the repository root after sourcing `scripts/env.sh`, using the probe Python
environment. These scripts use existing data and run on the CPU; they do not load models.

- `python scripts/76_story_figures.py`: fixed arithmetic and response-format comparison,
  plus generated-calculation readout checks. Companion JSON files record every plotted
  estimate and confidence interval; no new aggregate is fitted.
- `python scripts/66_figure_dump.py`: full-size appendix figures with independent reading
  instructions for each plot type. Main story figures are already full width and are not
  repeated in the dump. Existing technical figure generators remain available.

Then run `make paper-submission`. Main figure titles state findings; each figure explains
its task, comparison, measure and interval convention. Keep internal predictor accuracy
separate from the model's written answers. Keep patch removal separate from recovery of
correct answers. Use percentage points for added errors and percentages for accuracy.
