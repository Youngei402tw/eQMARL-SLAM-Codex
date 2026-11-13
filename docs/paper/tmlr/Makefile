# Anonymized TMLR submission. `make` regenerates Stage-30 numbers/figures and builds the PDF.
MAIN := BELLROUTE_TMLR
BUILD_DIR ?= build
PYTHON ?= python

.PHONY: all assets pdf submission clean

all: pdf

assets:
	$(PYTHON) make_tmlr_assets.py

pdf: assets $(MAIN).tex references.bib
	mkdir -p $(BUILD_DIR)
	SOURCE_DATE_EPOCH=1793750400 FORCE_SOURCE_DATE=1 latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf $(MAIN).pdf

# Self-contained source archive for upload (figures and numbers from the parent directory are copied in).
submission: pdf
	rm -rf $(BUILD_DIR)/src && mkdir -p $(BUILD_DIR)/src/figures $(BUILD_DIR)/src/generated
	cp $(MAIN).tex references.bib tmlr.sty tmlr.bst fancyhdr.sty $(BUILD_DIR)/src/
	cp figures/*.pdf ../figures/fig_ablation.pdf ../figures/fig_noise.pdf $(BUILD_DIR)/src/figures/
	cp generated/numbers30.tex ../generated/numbers.tex $(BUILD_DIR)/src/generated/
	sed -i 's#\\input{../generated/numbers.tex}#\\input{generated/numbers.tex}#; s#{../figures/}##' $(BUILD_DIR)/src/$(MAIN).tex
	cd $(BUILD_DIR)/src && latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex >/dev/null && latexmk -c $(MAIN).tex >/dev/null && rm -f $(MAIN).pdf
	cd $(BUILD_DIR) && rm -f $(MAIN)-source.zip && zip -qr $(MAIN)-source.zip src
	cp $(BUILD_DIR)/$(MAIN)-source.zip .

clean:
	rm -rf $(BUILD_DIR)
