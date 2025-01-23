# Evolutionary Algorithm for Bipartite Graph Problems

This repository contains an implementation of an **Evolutionary Algorithm** designed to solve the **Bipartite Graph Assignment Problem**. The algorithm is implemented in Python and includes features for customization of genetic operations, such as crossover, mutation, and elitism. The repository also includes test scripts and visualization utilities.

## Features
- Customizable evolutionary algorithm with:
  - Crossover operators
  - Mutation operators
  - Elitism and selection methods (tournament, roulette, and random)
- Fitness evaluation based on weighted edges
- Comparison with the **Hungarian Algorithm** for finding optimal solutions
- Visualization of results and graphs

---

## Repository Structure
- `EvolutionaryAlgorithm.py`: Class definition of the Evolutionary Algorithm.
- `evoalgfunc.py`: Supporting utility functions for the algorithm (e.g., graph generation, crossover, mutation, selection methods).
- `EvolutionaryAlgorithmTest.ipynb`: Jupyter notebook for running experiments, visualizations, and testing different parameter configurations.
- Test images (e.g., `EA_test.png`, `EA_test_crossover.png`).

---

## Installation

### Requirements
To set up the project, ensure you have the following:
- Python 3.8 or later
- Required Python libraries (listed in `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/salomeaaleksandra/EvolutionaryAlgorithm.git
   cd EvolutionaryAlgorithm
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
### Running the Algorithm
To execute the algorithm and visualize the results:
1. Import the necessary modules:
   ```python
   from EvolutionaryAlgorithm import EvolutionaryAlgorithm
   from evoalgfunc import generate_bipartite_weighted_graph
   ```

2. Generate a graph:
   ```python
   G = generate_bipartite_weighted_graph(200, 200, max_weight=100)
   ```

3. Initialize and run the algorithm:
   ```python
   ea = EvolutionaryAlgorithm(
       G,
       population_size=100,
       generations=200,
       crossover_prob=0.8,
       mutation_prob=0.1,
       elitism_type="strict",
       elite_count=2,
       selection="tournament",
       seed=42,
       logs=True
   )
   best_solution, best_score, best_solutions, best_scores = ea.run()
   ```

4. Visualize results:
   ```python
   from evoalgfunc import plot_bipartite_graph
   plot_bipartite_graph(G, best_solution)
   ```

### Evaluation with Hungarian Algorithm
Compare results with the Hungarian algorithm:
```python
from scipy.optimize import linear_sum_assignment
import numpy as np

# Extract the cost matrix and solve
optimal_solution = ...
```

---

## Experimental Setup
1. Graph generation method: Bipartite graph with random edge weights.
2. Algorithm parameters:
   - Population size
   - Crossover and mutation probabilities
   - Selection and elitism methods
3. Evaluation metrics:
   - Best score per generation
   - Convergence speed
   - Comparison with the Hungarian algorithm

---

## Contributing
Feel free to submit issues or pull requests for improvements.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Hungarian algorithm: Harold W. Kuhn
- Python libraries: NetworkX, SciPy, Matplotlib
