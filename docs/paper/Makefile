PAPER := BUDGETED_ENTANGLEMENT_ROUTING
QIP := QIP_ROBUST_ENTANGLEMENT_ROUTING
BUILD_DIR ?= build
PYTHON ?= python
AUDIT ?= ../../experiments/29/29-qip-release-audit_attempt-01

.PHONY: all pdf qip qip-data clean

all: pdf

pdf: $(PAPER).tex $(PAPER).bib
	mkdir -p $(BUILD_DIR)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(PAPER).pdf

# Regenerate numbers and figures from the Stage 29 audit bundle, then build the QIP manuscript.
qip-data:
	$(PYTHON) make_numbers.py --audit $(AUDIT)
	$(PYTHON) make_figures.py --data $(AUDIT)/figure_data --out figures

qip: qip-data $(QIP).tex $(QIP).bib
	mkdir -p $(BUILD_DIR)
	SOURCE_DATE_EPOCH=1793750400 FORCE_SOURCE_DATE=1 latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $(QIP).tex
	cp $(BUILD_DIR)/$(QIP).pdf $(QIP).pdf

clean:
	latexmk -C -outdir=$(BUILD_DIR) $(PAPER).tex $(QIP).tex
	rm -rf $(BUILD_DIR)
