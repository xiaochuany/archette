window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      tags: 'ams',
      package: {"[+]":["ams"]},
      macros: {
        RR: "\\mathbb{R}",
        EE: "\\mathbb{E}",
        NN: "\\mathbb{N}",
        PP: "\\mathbb{P}",
        Var: "\\mathrm{Var}",
      }
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };

  document$.subscribe(() => {
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })