PAPER := BUDGETED_ENTANGLEMENT_ROUTING
BUILD_DIR ?= build

.PHONY: all pdf clean

all: pdf

pdf: $(PAPER).tex $(PAPER).bib
	mkdir -p $(BUILD_DIR)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(PAPER).pdf

clean:
	latexmk -C -outdir=$(BUILD_DIR) $(PAPER).tex
	rm -rf $(BUILD_DIR)
