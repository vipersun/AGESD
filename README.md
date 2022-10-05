# AGESD


# Overview

This repository is the implementation code of the attention-based graph embedded software decomposition (AGESD for short) model.

# Build and run

Following the instructions below to build and run the product.

Requirements:

- PyTorch-GPU 1.7 and above
- kneed 0.7 and above
- munkres 1.1 and above
- networkx
- numpy
- scipy 1.7 and above
- sklearn

Running AGESD ( vcperf for example):

1. Clone the repository on your machine.

2. Go to the samples directory.

   ```
   cd samples
   ```

3. Execute the first phase of the model to obtain a pkl file.

   ```
   python ..\pre_agesd.py --name vcperf
   ```

4. Execute the second phase to obtain the results of the decomposition and the evaluation results.

   ```
   python ..\agesd.py --name vcperf
   ```

The output file and evaluation file are saved in the root directory of the D disk.
