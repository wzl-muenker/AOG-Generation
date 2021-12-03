# AOG-Generation
> Python scripts to generate AND/OR graphs for Assembly Sequence Planning

Several studies in production engineering and operations research focused on disassembly and assembly planning for improving the profitability of remanufacturing. The representation of feasible (dis)assembly sequences by analyzing geometrical and technical precedence constraints is an essential necessity of (dis)assembly planning. An AND/OR graph (AOG) is commonly used to represent a product’s feasible (dis)assembly sequences, including alternative subassemblies and parallel operations. In recent years, researchers studied the automatic extraction of geometrical precedence constraints by collision analysis within 3D models. However, since most of the existing approaches focused on collision analysis to identify complex precedence constraints, the generation of AOGs for complex products from collision analysis results remains too inefficient for industrial use cases. In this study, a Computer-Aided Design (CAD) interface from previous studies is used to extract liaison and moving wedge information from 3D models. A top-down and a bottom-up approach for generating complete AOGs from the extracted data are introduced. The approaches’ computing performances are analyzed on sample test cases and CAD models. The bottom-up approach performed better for the given samples. It has been found that the amount of moving wedge constraints has a strong effect on computing performance. It is a strong indicator to estimate the complexity of products under examination. The exponential behavior of the needed computing resources can be estimated beforehand. For complex products, graph simplification or alternative graph representations with less information richness should be considered. The results contribute to automated (dis)assembly sequence planning from complex products and, consequently, increasing remanufacturing profitability.

## Installation

Installation of all requirements in Python 3.7 environment:

```sh
$ pip install -r requirements.txt
```

## Used CAD examples

Following CAD examples have been used to validate the approaches:

- Centrifugal Pump: https://grabcad.com/library/single-stage-centrifugal-pump-3
- Multiplate Clutch: https://grabcad.com/library/formula-1-multiplate-clutch
- Shear Mold: https://grabcad.com/library/shear-mold


## Meta

Sören Münker – s.muenker@wzl.rwth-aachen.de

This work is part of the research project “Internet of Construction” that is funded by the Federal Ministry of Education and Research of Germany within the indirective on a joint funding initiative in the field of innovation for production, services and labor of tomorrow (funding number: 02P17D081) and supported by the project management agency “Projektträger Karlsruhe (PTKA)”. The authors are responsible for the content. 

[http://internet-of-construction.com/index_en.html](http://internet-of-construction.com/index_en.html)
