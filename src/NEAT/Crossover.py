import copy
import random

from src.CoDeepNEAT.BlueprintGenome import BlueprintGenome
from src.CoDeepNEAT.ModuleGenome import ModuleGenome
from src.NEAT.Connection import Connection


def update_connection(conn: Connection, genome):
    # Add the node to the genome if it is not already there
    if conn.from_node.id not in genome.node_ids:
        new_from = copy.deepcopy(conn.from_node)
        genome.add_node(new_from)
        conn.from_node = new_from
    else:
        # If the node is in the genome make the connection point to that node
        conn.from_node = genome.get_node(conn.from_node.id)

    if conn.to_node.id not in genome.node_ids:
        new_to = copy.deepcopy(conn.to_node)
        genome.add_node(new_to)
        conn.to_node = new_to
    else:
        conn.to_node = genome.get_node(conn.to_node.id)


def crossover(parent1, parent2):
    # Choosing the fittest parent
    if parent1.fitness == parent2.fitness:  # if the fitness is the same choose the shortest
        best_parent, worst_parent = (parent2, parent1) \
            if len(parent2.connections) < len(parent1.connections) else (parent1, parent2)
    else:
        best_parent, worst_parent = (parent2, parent1) \
            if parent2.fitness > parent1.fitness else (parent1, parent2)

    # disjoint + excess are inherited from the most fit parent
    d, e = copy.deepcopy(best_parent.get_disjoint_excess(worst_parent))

    # TODO how to inherit connection genes?
    if type(parent1) == ModuleGenome:
        child = ModuleGenome(d + e, [])
    else:
        child = BlueprintGenome(d + e, [])

    for conn in child.connections:
        update_connection(conn, child)

    # Finding the remaining matching genes and choosing randomly between them
    for best_conn in best_parent.connections:
        if best_conn.innovation in worst_parent.innov_nums:
            worst_conn = worst_parent.get_connection(best_conn.innovation)
            choice = copy.deepcopy(random.choice([best_conn, worst_conn]))
            child.add_connection(choice)

            update_connection(choice, child)

    return child
