# FedAnDE: Federated Learning of AnDE Classifiers

This repository contains the officially anonymized implementation of **FedAnDE**, a federated learning framework for the *Averaged n-Dependence Estimators* (AnDE) family of Bayesian network classifiers.

The framework supports federated training of **Generative**, **Discriminative**, and **Hybrid** AnDE models, with optional **Differential Privacy** (DP) guarantees.

## Key Features

- **Federated AnDE:** Supports arbitrary dependence orders ($n=0$ for Naive Bayes, $n=1$ for A1DE, $n=2$ for A2DE, etc.).
- **Multiple Federation Modes:**
  - **Generative:** Aggregates privacy-preserving sufficient statistics (counts).
  - **Discriminative:** Federates log-linear weights via gradient-based optimization.
  - **Hybrid:** Combines local generative priors with global discriminative weights, providing privacy by design. Optionally, the counts can be also federated as in the generative mode.
- **Differential Privacy:** Implements $\varepsilon$-DP using the Laplace mechanism for robust privacy protection of the generative counts, more vulnerable because exposes directly the data counts. Instead, the discriminative parameters don't have a real meaning, as in neural networks.


## Project Structure

- `src/main/java/fedAnDE/`: Source code for the federated framework.
  - `model/`: Implementation of AnDE models (PT, WDPT).
  - `fusion/`: Aggregation logic for Server (Generative/Discriminative).
  - `privacy/`: Differential Privacy mechanisms (Laplace, Gaussian, ZCDP).
  - `experiments/`: Scripts to reproduce the paper's experiments.

## Usage

### Prerequisites
- Java 17 or higher
- Maven

### Running Experiments
The main entry point for experiments is `fedAnDE.experiments.CCBNExperiment`.

Example command to run a federated experiment:

```bash
java -cp target/fedAnDE-1.0-SNAPSHOT.jar fedAnDE.experiments.CCBNExperiment \
    <folder> <dataset> <nClients> <seed> <nFolds> <nIterations> \
    <structure> <parameterLearning> <maxIterations> \
    <fuseParameters> <fuseProbabilities> <nBins> \
    <alpha> <dpType> <sensitivity> <autoSens> <epsilon>
```

**Parameters:**
- `structure`: `NB`, `A1DE`, `A2DE`
- `parameterLearning`: `Weka` (Generative), `dCCBN` (Discriminative), `wCCBN` (Hybrid)
- `dpType`: `None`, `Laplace`

## 🔬 Reproducible Research
This codebase is designed to be fully reproducible. All random seeds are fixed for data partitioning and differential privacy noise generation.

---
*Anonymized for peer review.*
