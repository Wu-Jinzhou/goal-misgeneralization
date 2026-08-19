# Reconstructed LaTeX template

The paper does not load a named publisher or conference class. Its effective
template is a custom, Nature-style manuscript built on the standard LaTeX
`article` class with these principal choices:

- 11 pt type on A4 paper with one-inch margins;
- Palatino body type;
- `authblk` for authors and affiliations;
- parenthesized numeric citations with `natbib` and `naturemag.bst`;
- a custom centered abstract heading;
- unnumbered main-text and Methods headings;
- numbered supplementary sections and custom Extended Data float numbering.

The archive also contains `main.sty`, but `main.tex` never loads it. That file
identifies itself internally as the AAAI 2022 conference style and is unrelated
to the formatting actually used by this paper, so it is intentionally excluded
from this template.

All paper-specific prose, author names, institutions, figures, tables,
bibliography entries, and scientific notation have been removed. Comments in
`main.tex` mark the intended manuscript structure and provide inert float
skeletons.

Compile the blank template with:

```sh
latexmk -pdf main.tex
```

After adding citations and BibTeX entries, uncomment
`\bibliography{references}` and run the same command.
