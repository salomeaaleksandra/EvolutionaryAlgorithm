import random

from evoalgfunc import (
    generate_bipartite_weighted_graph,
    plot_bipartite_graph,
    create_genotype,
    fitness,
    roulette_selection,
    tournament_selection,
    is_valid_solution,
    repair_duplicates,
    crossover,
    elitism,
    mutate,
)

class EvolutionaryAlgorithm:
    def __init__(self, G, population_size, generations, crossover_prob, mutation_prob, elitism_type="strict", elite_count=2, fraction_random=0.3, selection="tournament", scaling=None, seed=None, logs=False):
        self.G = G  # Bipartite graph object
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_type = elitism_type
        self.elite_count = elite_count
        self.fraction_random = fraction_random
        self.selection = selection
        self.scaling = scaling if selection == "roulette" else None  # Ensure scaling is only used when needed
        self.seed = seed
        self.logs = logs

        if seed:
            random.seed(seed)

        # Initialize population
        self.side1 = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
        self.side2 = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
        self.population = [create_genotype(self.side1, self.side2) for _ in range(self.population_size)]
        
        # Initialize best solution from the initial population
        scores = [fitness(ind, self.G) for ind in self.population]
        best_index = scores.index(min(scores))
        self.best_solution = self.population[best_index]
        self.best_score = scores[best_index]

        # Track best solutions and scores across generations
        self.best_solutions = [self.best_solution]
        self.best_scores = [self.best_score]

    def run(self):
        for gen in range(self.generations):
            if self.logs:
                print(f"\nGeneration {gen + 1}:")
            scores = [fitness(ind, self.G) for ind in self.population]
            next_population = []

            # Apply elitism if enabled
            if self.elitism_type:
                elites, elite_scores = elitism(
                    self.population, scores, type=self.elitism_type, n_op=self.elite_count, fraction_random=self.fraction_random
                )
                next_population.extend(elites)
                if self.logs:
                    print(f"  Elites Passed to Next Generation: {elites}, with scores {elite_scores}")

                # Remove elites from the selection pool
                non_elites = [ind for ind in self.population if ind not in elites]
                non_elite_scores = [scores[self.population.index(ind)] for ind in non_elites]
            else:
                non_elites = self.population
                non_elite_scores = scores

            # Apply selection method
            if self.selection == "random":
                # Random selection for remaining population
                while len(next_population) < self.population_size:
                    parent1 = random.choice(non_elites)
                    parent2 = random.choice(non_elites)

                    offspring = crossover(parent1, parent2, self.crossover_prob, self.side1, self.side2, if_validate=True)
                    if offspring:
                        child1, child2 = offspring
                        next_population.extend([child1, child2])
            elif self.selection == "roulette":
                # Roulette selection for remaining population
                non_elites = roulette_selection(non_elites, non_elite_scores, scaling=self.scaling, logs=self.logs)
                while len(next_population) < self.population_size:
                    parent1 = random.choice(non_elites)
                    parent2 = random.choice(non_elites)

                    offspring = crossover(parent1, parent2, self.crossover_prob, self.side1, self.side2, if_validate=True)
                    if offspring:
                        child1, child2 = offspring
                        next_population.extend([child1, child2])
            elif self.selection == "tournament":
                # Tournament selection for remaining population
                while len(next_population) < self.population_size:
                    if len(non_elites) < 2:
                        raise ValueError("Not enough non-elite individuals to perform tournament selection.")

                    parent1 = tournament_selection(non_elites, non_elite_scores, logs=self.logs)
                    parent2 = tournament_selection(non_elites, non_elite_scores, logs=self.logs)
                    if not isinstance(parent1, list) or not isinstance(parent2, list):
                        raise TypeError(f"Invalid parent type: parent1={type(parent1)}, parent2={type(parent2)}")

                    offspring = crossover(parent1, parent2, self.crossover_prob, self.side1, self.side2, if_validate=True)
                    if offspring:
                        child1, child2 = offspring
                        next_population.extend([child1, child2])
            else:
                raise ValueError("Invalid selection method. Use 'random', 'roulette', or 'tournament'.")

            # Apply mutation
            next_population = [
                mutate(ind, self.mutation_prob, self.side2, index=i + 1 if self.logs else None, logs=self.logs)
                for i, ind in enumerate(next_population)
            ]

            self.population = next_population

            # Update best solution from the next_population
            scores = [fitness(ind, self.G) for ind in self.population]
            current_best_index = scores.index(min(scores))
            if scores[current_best_index] < self.best_score:
                self.best_score = scores[current_best_index]
                self.best_solution = self.population[current_best_index]

            # Track best solution and score for the generation
            self.best_solutions.append(self.best_solution)
            self.best_scores.append(self.best_score)

            # Display population if logs are enabled
            if self.logs:
                for i, individual in enumerate(self.population):
                    print(f"  Individual {i+1}: {individual} with score: {fitness(individual, self.G)}")

        if not self.logs:
            print(f"Final Generation Best Score = {self.best_score}")

        return self.best_solution, self.best_score, self.best_solutions, self.best_scores


