# Forkworld current-results essay

This directory contains the results used in
`Predicting Learned Goals: Evidence, Accessibility, and Competition in
Forkworld`.

The prospectively specified scope is thirteen complete experiment families:
50,290 runs in the original nine and 3,720 runs in four fixed follow-up grids,
for 54,010 runs total. Six explicitly post-hoc or adaptive full panels add 920
runs, bringing the complete reported corpus to 54,930. The stopped E18 pilot
and all other engineering pilots, deterministic replays, and smoke tests are
integrity checks rather than scientific runs. Every reported full-panel run is
complete; the analyzed roots contain no failed artifacts.

From the repository root, regenerate the plotted data and figures with:

```sh
.venv/bin/python paper/forkworld-current-results/analysis_and_plots.py
.venv/bin/python paper/forkworld-current-results/followup_analysis.py
.venv/bin/python paper/forkworld-current-results/e14_analysis.py
.venv/bin/python paper/forkworld-current-results/e15_analysis.py
.venv/bin/python paper/forkworld-current-results/e16_analysis.py
.venv/bin/python paper/forkworld-current-results/e17_analysis.py
.venv/bin/python paper/forkworld-current-results/e19_full_analysis.py
.venv/bin/python paper/forkworld-current-results/e20_full_analysis.py
.venv/bin/python paper/forkworld-current-results/figure_e19.py
.venv/bin/python paper/forkworld-current-results/figure_e20.py
```

For the exact scientific-runtime versions recorded in the follow-up artifacts,
create the environment with Python 3.11.14 and the repository-level
`constraints-paper.txt` as described in the main README.

Then compile the document with:

```sh
cd paper/forkworld-current-results
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Key files:

- `main.tex`: blog post and methodological appendix.
- `main.pdf`: compiled post.
- `analysis_and_plots.py`: strict original-grid analysis and Figures 1--9.
- `followup_analysis.py`: strict analysis of E10--E13 and Figures 10--13.
- `e14_analysis.py`: separate post-hoc audit, analysis, and Figure 14.
- `e15_analysis.py`--`e17_analysis.py`: adaptive multi-goal trajectory,
  support-completion, and winner-knockout analyses and Figures 15--17.
- `e18_analysis.py`: outcome-blind analysis for the stopped engineering pilot;
  it contains no full-panel order estimate.
- `e19_full_analysis.py` and `figure_e19.py`: strict active-input intervention
  analysis and Figure 19.
- `e20_full_analysis.py` and `figure_e20.py`: strict counterbalanced-order
  analysis and Figure 20.
- `figures/`: vector PDF figures plus PNG inspection copies.
- `derived/`: the exact plotted rows and figure manifest.
- `references.bib`: cited primary sources.

The plotting scripts look for Myriad Pro OTF files first in
`FORKWORLD_FONT_DIR` and then in `~/Library/Fonts`. When found, they embed Myriad
Pro in every vector plot; mathematical glyphs use Matplotlib's math font where
Myriad Pro lacks a glyph. If the files are unavailable, the scripts emit a
clear warning and fall back to bundled DejaVu Sans, so analysis remains
portable although text geometry may differ slightly.
