# Conditional Graph Regression for Complex Chemical Systems with Heterogeneous Substructures

## Abstract
Graph neural networks (GNNs) have been widely studied as an efficient and generalized method to predict the physical and chemical properties of chemical systems based on a single homogeneous atomic structure, such as molecule and crystalline material. However, most chemical systems in real-world applications of materials science and engineering contain multiple heterogeneous atomic substructures. Nonetheless, existing GNNs for chemical applications assumed homogeneous graphs with the input node or edge features in the same feature space. In this paper, we reformulate the regression problem on chemical systems as a regression problem on core substructures conditioned by environment substructures. Then, we propose conditional atomic subgraph interaction network (CASIN) that predicts the physical and chemical properties of the input chemical systems by learning the atomic interactions between the heterogeneous core and environment substructures in the chemical systems. For three real-world benchmark chemical datasets, CASIN outperformed state-of-the-art GNNs in predicting the physical and chemical properties of the complex solar cell materials and catalyst systems.

## Run
You can train and evaluate SIGNNA on the benchmark datasets.
The following Python scripts provide the SIGNNA implementation for each benchmark dataset.
- exec_hoip.py: HOIP dataset containing hybrid organic-inorganic perovskites and their band gaps.
- exec_hoip.py: HOIP2d dataset containing hybrid halide perovskites and their band gaps.
- exec_cathub.py: CatHub dataset containing chemical systems of inorganic catalsysts.


## Datasets
We provide the metadata files and the example data of the datasets rather than the full datasets due to the licenses of the benchmark datasets.
You can access the full benchmark datasets throught the following references.
- HOIP dataset: https://www.nature.com/articles/sdata201757
- HOIP2d dataset: https://pubs.acs.org/doi/10.1021/acs.chemmater.0c02290
- CatHub dataset: https://www.nature.com/articles/s41597-019-0081-y


## Employing Chemically Motivated Features
We can customize the initial node features of the virtual nodes in SIGNNA.
To this end, you should implement a function that returns a node-feature vector of ``numpy.array`` for an input ``pymatgen.Structure`` object and an atomic symbol.
Passing the implemented function through ``vn_method`` when calling the ``load_dataset`` or ``load_cathub_dataset`` function to load the dataset.
