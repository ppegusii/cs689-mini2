all:
	pdf

pdf: report.tex
	pdflatex report
	pdflatex report
	bibtex report
	pdflatex report
	pdflatex report

clean:
	rm -f paper.pdf *.dvi *.log *.aux *.bbl *.blg
