import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def generate_bipartite_weighted_graph(num_nodes_side1, num_nodes_side2, max_weight=100):
    """
    Generate a bipartite graph with two disjoint sets of nodes.
    
    Parameters:
        num_nodes_side1 (int): Number of nodes in the first set.
        num_nodes_side2 (int): Number of nodes in the second set.
        max_weight (int): Maximum weight for the edges.
    
    Returns:
        nx.Graph: A bipartite graph with random weights between nodes of the two sets.
    """
    B = nx.Graph()
    # Add nodes for each side
    side1 = [f"A{i}" for i in range(1, num_nodes_side1 + 1)]
    side2 = [f"B{i}" for i in range(1, num_nodes_side2 + 1)]
    B.add_nodes_from(side1, bipartite=0)
    B.add_nodes_from(side2, bipartite=1)
    
    # Add edges with random weights
    for u in side1:
        for v in side2:
            weight = random.randint(1, max_weight)
            B.add_edge(u, v, weight=weight)
    
    return B

def plot_bipartite_graph(B, solution=None):
    """
    Plot the bipartite graph with optional highlighting for the solution.
    
    Parameters:
        B (nx.Graph): The bipartite graph to be plotted.
        solution (list of tuples): Edges in the solution to be highlighted (optional).
    """
    # Get positions for a bipartite layout
    pos = nx.drawing.layout.bipartite_layout(B, nodes=[n for n, d in B.nodes(data=True) if d["bipartite"] == 0])
    
    plt.figure(figsize=(4, 6))
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(B, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(B, pos, edgelist=B.edges(), width=1, edge_color="gray")
    nx.draw_networkx_labels(B, pos, font_size=10, font_color="black")
    
    # Draw edge weights
    edge_labels = nx.get_edge_attributes(B, 'weight')
    nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels, label_pos=0.85, font_size=6)
    
    # Highlight the solution if provided
    if solution:
        nx.draw_networkx_edges(B, pos, edgelist=solution, width=2, edge_color=plt_colors[1])
    
    plt.title("Bipartite Weighted Graph")
    plt.axis("off")
    # plt.show()
    
def create_genotype(side1, side2):
    side2_shuffled = side2.copy()
    random.shuffle(side2_shuffled)
    return list(zip(side1, side2_shuffled))

def fitness(genotype, B):
    total_weight = 0
    used_nodes = set()
    penalty = 0
    for u, v in genotype:
        if v in used_nodes or not B.has_edge(u, v):
            penalty += 1
        else:
            used_nodes.add(v)
            total_weight += B.edges[u, v]['weight']
    return total_weight + (penalty * 1e9)

def roulette_selection(population, scores, logs=False):
    """
    Perform roulette wheel selection on the entire population.

    Parameters:
        population (list): The current population.
        scores (list): Fitness scores of the population.
        logs (bool): Whether to print out the roulette results.

    Returns:
        list: The selected population for the next generation.
    """
    fitness_values = [1 / (s + 1e-9) for s in scores]  # Minimize by inverting scores
    total_fitness = sum(fitness_values)
    probabilities = [fv / total_fitness for fv in fitness_values]

    cumulative_probabilities = []
    cumulative = 0
    for p in probabilities:
        cumulative += p
        cumulative_probabilities.append(cumulative)

    # Select individuals based on roulette probabilities
    selected_population = []
    for _ in range(len(population)):
        rand = random.random()
        for i, cp in enumerate(cumulative_probabilities):
            if rand <= cp:
                selected_population.append(population[i])
                if logs:
                    print(f"  Roulette selected: {population[i]} with probability {probabilities[i]:.4f}")
                break

    return selected_population

def tournament_selection(population, scores, logs=False):
    """
    Select a winner between two individuals.
    
    Parameters:
        population (list): The current population.
        scores (list): Fitness scores of the population.
        logs (bool): Whether to print out the roulette results.

    Returns:
        list: The selected individual from the tournament (winner) for the next generation.
    """
    if len(population) < 2:
        raise ValueError("Population size is less than 2. Cannot perform tournament selection.")

    individual1, individual2 = random.sample(population, 2)
    score1 = scores[population.index(individual1)]
    score2 = scores[population.index(individual2)]
    winner = individual1 if score1 < score2 else individual2
    if logs:
        print(f"  Duel: {score1} {score2} -> Winner: {winner}")
    return winner

def is_valid_solution(genotype):
    """
    Validate a genotype to ensure no duplicate nodes on either side.

    Parameters:
        genotype (list): List of tuples (e.g., [('A1', 'B1'), ...]).

    Returns:
        bool: True if valid, False otherwise.
    """
    side1_nodes = [gene[0] for gene in genotype]
    side2_nodes = [gene[1] for gene in genotype]
    # Ensure no duplicates in either side
    return len(side1_nodes) == len(set(side1_nodes)) and len(side2_nodes) == len(set(side2_nodes))

def repair_duplicates(genotype, side1, side2):
    """
    Repair duplicates in a genotype by replacing the second occurrence of duplicates
    with missing values.

    Parameters:
        genotype (list): List of tuples (e.g., [('A1', 'B1'), ...]).
        side1 (list): List of valid nodes on side 1.
        side2 (list): List of valid nodes on side 2.

    Returns:
        list: Repaired genotype with no duplicates.
    """
    side1_nodes = [gene[0] for gene in genotype]
    side2_nodes = [gene[1] for gene in genotype]

    # Identify duplicates and missing values
    duplicates_side2 = [node for node in side2_nodes if side2_nodes.count(node) > 1]
    missing_side2 = [node for node in side2 if node not in side2_nodes]

    # Replace the second occurrence of duplicates on side 2
    for i, node in enumerate(side2_nodes):
        if node in duplicates_side2:
            # Only replace the second occurrence (not the first)
            if side2_nodes[:i].count(node) > 0:
                genotype[i] = (genotype[i][0], missing_side2.pop(0))
                duplicates_side2.remove(node)

    return genotype

def crossover(parent1, parent2, pk, side1, side2, if_validate=False, logs=False):
    """
    Perform crossover between two parents to produce two offspring.
    Includes debugging to track when crossover is applied or skipped.

    Parameters:
        parent1 (list): First parent's genotype.
        parent2 (list): Second parent's genotype.
        pk (float): Crossover probability.
        side1 (list): Nodes on side 1 of the bipartite graph.
        side2 (list): Nodes on side 2 of the bipartite graph.
        if_validate (bool): If True, validate and repair duplicates in children.
        logs (bool): If print out any info and errors.

    Returns:
        tuple: Two children (list of tuples) or None if crossover is skipped.
    """
    # Debug: Ensure parents are valid lists
    if not isinstance(parent1, list) or not isinstance(parent2, list):
        raise TypeError(f"Invalid parent type: parent1={type(parent1)}, parent2={type(parent2)}")

    if random.random() > pk:
        if logs:
            print(f"  Crossover skipped for parents: {parent1} and {parent2}")
        return parent1, parent2  # Skip crossover with pk probability

    size = len(parent1)
    cutting_point = random.randint(1, size - 1)  # Avoid trivial cuts at 0 or size

    # Create the children by splitting parents at the cutting point
    child1 = parent1[:cutting_point] + parent2[cutting_point:]
    child2 = parent2[:cutting_point] + parent1[cutting_point:]

    # Validate children
    if is_valid_solution(child1) and is_valid_solution(child2):
        if logs:
            print(f"  Crossover success at cut {cutting_point}: {child1} and {child2}")
        return child1, child2
    elif if_validate:
        # Repair duplicates if validation is enabled
        child1 = repair_duplicates(child1, side1, side2)
        child2 = repair_duplicates(child2, side1, side2)
        if (child1 == parent1 and child2 == parent2) or (child1 == parent2 and child2 == parent1):
            if logs:
                print(f"  Repaired children are identical to parents. Skipping.")
            return None
        if is_valid_solution(child1) and is_valid_solution(child2):
            if logs:
                print(f"  Crossover repaired at cut {cutting_point}: {child1} and {child2}")
            return child1, child2
        else:
            if logs:
                print(f"  Repair failed for crossover between {parent1} and {parent2}. Skipping.")
            return None
    else:
        if logs:
            print(f"  Invalid crossover between {parent1} and {parent2}. Skipping.")
        return None
    
def elitism(population, scores, type='strict', n_op=2, fraction_random=0.3):
    """
    Perform elitism based on the type and the number of elite individuals to pass.

    Parameters:
        population (list): The current population.
        scores (list): Fitness scores for the population.
        type (str): Elitism type ('strict' or 'partial').
        n_op (int): Number of elite individuals to pass.
        fraction_random (float): Fraction of the population to randomly select for partial elitism.

    Returns:
        tuple: Elite individuals and their scores.
    """
    if type == 'strict':
        # Select the top n_op individuals directly
        sorted_population_scores = sorted(zip(scores, population), key=lambda x: x[0])
        elite_population = [x[1] for x in sorted_population_scores[:n_op]]
        elite_scores = [x[0] for x in sorted_population_scores[:n_op]]
        return elite_population, elite_scores

    elif type == 'partial':
        # Randomly select a subset of the population
        random_subset_size = int(len(population) * fraction_random)
        random_subset = random.sample(list(zip(population, scores)), random_subset_size)
        
        # Sort the random subset by fitness
        sorted_random_subset = sorted(random_subset, key=lambda x: x[1])
        
        # Select the top n_op individuals from the random subset
        elite_population = [x[0] for x in sorted_random_subset[:n_op]]
        elite_scores = [x[1] for x in sorted_random_subset[:n_op]]
        return elite_population, elite_scores

    else:
        raise ValueError("Unsupported elitism type. Use 'strict' or 'partial'.")

def mutate(individual, pm, side2, index=None, logs=False):
    """
    Perform mutation on an individual by swapping two random indices in side B (side2)
    with a given mutation probability.

    Parameters:
        individual (list): The genotype (list of tuples) to be mutated.
        pm (float): Mutation probability.
        side2 (list): List of valid nodes on side 2.
        index (int): Index of the individual in the population (for debugging).
        logs (bool): If true, the result of mutation is printed out.

    Returns:
        list: Mutated genotype.
    """
    if random.random() > pm:
        return individual  # No mutation occurs

    # Select two random indices to swap in side2
    idx1, idx2 = random.sample(range(len(individual)), 2)
    mutated_individual = individual.copy()

    # Perform the swap on side B (values in the second part of the tuples)
    mutated_individual[idx1], mutated_individual[idx2] = (
        (mutated_individual[idx1][0], mutated_individual[idx2][1]),
        (mutated_individual[idx2][0], mutated_individual[idx1][1]),
    )
    
    if logs:
        print(
            f"  Mutation applied on Individual {index}: Original -> {individual}, "
            f"Swapped indices {idx1} and {idx2} -> {mutated_individual}"
        )
    return mutated_individual

