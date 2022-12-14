# Conditional Graph Regression for Complex Chemical Systems with Heterogeneous Substructures

## Abstract
Graph neural networks (GNNs) have been widely studied as an efficient and generalized method to predict the physical and chemical properties of chemical systems based on a single homogeneous atomic structure, such as molecule and crystalline material. However, most chemical systems in real-world applications of materials science and engineering contain multiple heterogeneous atomic substructures. Nonetheless, existing GNNs for chemical applications assumed homogeneous graphs with the input node or edge features in the same feature space. In this paper, we reformulate the regression problem on chemical systems as a regression problem on core substructures conditioned by environment substructures. Then, we propose conditional atomic subgraph interaction network (CASIN) that predicts the physical and chemical properties of the input chemical systems by learning the atomic interactions between the heterogeneous core and environment substructures in the chemical systems. For three real-world benchmark chemical datasets, CASIN outperformed state-of-the-art GNNs in predicting the physical and chemical properties of the complex solar cell materials and catalyst systems.

## Run
You can train and evaluate CASIN on the benchmark datasets.
The following Python scripts provide the CASIN implementation for each benchmark dataset as:
- exec_hoip.py: HOIP dataset containing hybrid organic-inorganic perovskites and their band gaps.
- exec_tdhhp.py: TDHHP dataset containing hybrid halide perovskites and their band gaps.
- exec_cathub.py: CatHub dataset containing chemical systems of inorganic catalsysts.


## Datasets
We provide the metadata files and the example data of the datasets rather than the full datasets due to the licenses of the benchmark datasets.
You can access the full benchmark datasets throught the following references.
- HOIP dataset: https://www.nature.com/articles/sdata201757
- TDHHP dataset: https://pubs.acs.org/doi/10.1021/acs.chemmater.0c02290
- CatHub dataset: https://www.nature.com/articles/s41597-019-0081-y
