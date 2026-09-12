# Everything here is safe on the login node except `test-full`, `tokcheck` and the GPU stages,
# which run under sbatch (the login node caps virtual memory at 8 GB; see CLAUDE.md §1).
PY ?= /work/jvl210002/migration/envs/probe-cu129/bin/python
TECTONIC ?= /work/jvl210002/migration/envs/tex/bin/tectonic
export PYTHONPATH := $(CURDIR)/src

.PHONY: check test test-full data tokcheck dry status paper numbers clean-data

check: test dry
	@echo "OK"

test:
	$(PY) -m pytest tests/ -q

## tokenizer-dependent tests + a model load, on a CPU node
test-full:
	sbatch -p dev -t 00:30:00 -c 4 --mem=32G -J probe_tests -o log/slurm/%j_tests.out \
	  --wrap 'source $(CURDIR)/scripts/env.sh && cd $(CURDIR) && python -m pytest tests/ -q --run-tokenizer'

## build every level's probe-train and matched test sets (CPU, seconds)
data:
	$(PY) scripts/10_build_dataset.py

## rule-4 word lists per model (needs a tokenizer -> dev partition)
tokcheck:
	$(PY) scripts/slurm/pipeline.py --stage tokcheck --models all

dry:
	@$(PY) scripts/slurm/pipeline.py --stage all --models llama32-3b --dry-run > /dev/null && echo "pipeline dry-run clean"

status:
	@$(PY) scripts/00_status.py

paper:
	@mkdir -p log && TECTONIC_CACHE_DIR=/work/jvl210002/migration/cache/tectonic TMPDIR=/work/jvl210002/migration/tmp \
	  $(TECTONIC) -X compile $(CURDIR)/paper/main.tex 2>&1 | grep -v 'lineno.sty:296' || true
	@$(PY) scripts/92_page_budget.py

## list every number still unfilled in the draft
numbers:
	@sed 's/%.*//' paper/main.tex | grep -o '\\NUM{[^}]*}' | sort | uniq -c | sort -rn
