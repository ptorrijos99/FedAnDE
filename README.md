# FedAnDE: Federated Learning of AnDE Classifiers

This repository contains the officially anonymized implementation of **FedAnDE**, a federated learning framework for the *Averaged n-Dependence Estimators* (AnDE) family of Bayesian network classifiers.

The framework supports federated training of **Generative**, **Discriminative**, and **Hybrid** AnDE models, with optional **Differential Privacy** (DP) guarantees.

## Key Features

- **Federated AnDE:** Supports arbitrary dependence orders ($n=0$ for Naive Bayes, $n=1$ for A1DE, $n=2$ for A2DE, etc.).
- **Multiple Federation Modes:**
  - **Generative:** Aggregates privacy-preserving sufficient statistics (counts).
  - **Discriminative:** Federates log-linear weights via gradient-based optimization.
  - **Hybrid:** Combines local generative priors with global discriminative weights, providing privacy by design. Optionally, the counts can be also federated as in the generative mode.
- **Differential Privacy:**
  - **Generative (`PT`):** Implements **formal $\varepsilon$-DP** by perturbing sufficient statistics (counts) with Laplace noise. The sensitivity is calibrated to the number of parameters.
  - **Discriminative / Hybrid (`WDPT`):** Implements a **heuristic privacy mechanism** by injecting noise into the probability parameters. **Note:** This does not guarantee formal DP because the sensitivity is not strictly bounded for probabilities.


## Project Structure

The source code is organized in `src/main/java/fedAnDE/`:

- `core/`: Core federated learning components (`Client`, `Server`).
- `model/`: Implementation of AnDE models (`WDPT` for discriminative/hybrid, `PT` for generative).
- `fusion/`: Aggregation logic and fusion strategies.
- `privacy/`: Differential Privacy mechanisms (Laplace, Gaussian, ZCDP).
- `algorithms/`: Local learning algorithms.
- `experiments/`: Experiment execution logic.
  - `ExperimentRunner.java`: Main entry point for running experiments.
- `utils/`: Shared utility classes.

## Usage

### Prerequisites
- Java 17 or higher
- Maven

### Building the Project
To build the project and generate the JAR file with dependencies:

```bash
mvn clean package
```

### Running Experiments
The main entry point is `fedAnDE.experiments.ExperimentRunner`. You can run it directly or via the legacy wrapper `CCBNExperiment`.

**Command-line Usage:**

```bash
java -cp target/fedAnDE-1.0-jar-with-dependencies.jar fedAnDE.experiments.ExperimentRunner \
    <lineIndex> <paramsFile> <nodeName>
```

*   `<lineIndex>`: The 0-based index of the line to execute from the parameters file (e.g., `0` for the first line). This allows running specific experiments from a batch file.
*   `<paramsFile>`: Path to a file containing experiment parameters.
*   `<nodeName>`: Identifier for the node (e.g., "localhost").

**Parameters File Format:**
The parameters file should contain space-separated values for:
`folder dataset nClients seed nFolds nIterations structure parameterLearning maxIterations fuseParameters fuseProbabilities nBins [alpha] [dpType] [sensitivity] [autoSens] [epsilon] [delta/rho]`

**Example:**
To run a default experiment (if no args provided):
```bash
java -cp target/fedAnDE-1.0-jar-with-dependencies.jar fedAnDE.experiments.ExperimentRunner
```