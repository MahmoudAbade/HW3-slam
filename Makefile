TEX = pdflatex
SRC = report.tex
PDF = report.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC) ground_truth_path_2d.png ground_truth_path_3d.png image.png
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f report.aux report.log report.out report.toc report.pdf
