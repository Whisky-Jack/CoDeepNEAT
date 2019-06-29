import random
from src.NEAT.Connection import Connection
from src.NEAT.NEATNode import NEATNode
from src.NEAT.Mutation import NodeMutation, ConnectionMutation


class Genome:

    def __init__(self, connections, nodes):
        self.fitness = 0
        self.adjusted_fitness = 0

        # TODO make this a map
        self.nodes = nodes
        self.node_ids = set([node.id for node in nodes])

        # TODO make this a map
        self.innov_nums = set()
        self.connections = []
        for connection in connections:
            self.add_connection(connection)

    def add_connection(self, new_connection):
        """Adds a new connection maintaining the order of the connections list"""
        if new_connection.innovation in self.innov_nums:
            return -1

        pos = 0
        for i, connection in enumerate(self.connections, 0):
            if new_connection.innovation < connection.innovation:
                break
            pos += 1

        self.connections.insert(pos, new_connection)
        self.innov_nums.add(new_connection.innovation)
        return pos

    def get_connection(self, innov):
        if innov not in self.innov_nums:
            return None

        for conn in self.connections:
            if conn.innovation == innov:
                return conn

        return None

    def add_node(self, node):
        if node.id not in self.node_ids:
            self.node_ids.add(node.id)
            self.nodes.append(node)

    def get_node(self, id):
        if id not in self.node_ids:
            return None

        for node in self.nodes:
            if node.id == id:
                return node

    def distance_to(self, other_indv, c1=1, c2=1):
        n = max(len(self.connections), len(other_indv.connections))
        self_d, self_e = self.get_disjoint_excess(other_indv)
        other_d, other_e = other_indv.get_disjoint_excess(self)

        d = len(self_d) + len(other_d)
        e = len(self_e) + len(other_e)

        compatibility = (c1 * d + c2 * e) / n
        return compatibility

    # This is not as efficient but it is cleaner than a single pass
    def get_disjoint_excess(self, other):
        """Finds all of self's disjoint and excess genes when compared to other"""
        disjoint = []
        excess = []

        other_max_innov = other.connections[-1].innovation

        for connection in self.connections:
            if connection.innovation not in other.innov_nums:
                if connection.innovation > other_max_innov:
                    excess.append(connection)
                else:
                    disjoint.append(connection)

        return disjoint, excess

    def _check_node_mutation(self, mutation, mutated_node, mutated_from_conn, mutated_to_conn, curr_gen_mutations,
                             innov, node_id):
        # New mutation
        if mutation not in curr_gen_mutations:
            innov += 1
            mutated_from_conn.innovation = innov
            innov += 1
            mutated_to_conn.innovation = innov
            node_id += 1
            mutated_node.id = node_id

            curr_gen_mutations.add(mutated_node)
        # Mutation has already been done
        else:
            for prev_mutation in curr_gen_mutations:
                if mutation == prev_mutation:
                    # Use previous innovation numbers
                    mutated_from_conn.innovation = prev_mutation.from_conn.innovation
                    mutated_to_conn.innovation = prev_mutation.to_conn.innovation
                    mutated_node.id = prev_mutation.node_id
                    break

        return innov, node_id

    def _mutate_add_node(self, conn: Connection, curr_gen_mutations: set, innov: int, node_id: int):
        conn.enabled = False

        mutated_node = NEATNode(node_id + 1, conn.from_node.midpoint(conn.to_node))
        mutated_from_conn = Connection(conn.from_node, mutated_node)
        mutated_to_conn = Connection(mutated_node, conn.to_node)

        mutation = NodeMutation(mutated_node.id, mutated_from_conn, mutated_to_conn)

        innov, node_id = self._check_node_mutation(mutation,
                                                   mutated_node,
                                                   mutated_from_conn,
                                                   mutated_to_conn,
                                                   curr_gen_mutations, innov,
                                                   node_id)

        self.add_connection(mutated_from_conn)
        self.add_connection(mutated_to_conn)
        self.add_node(mutated_node)

        return innov, node_id

    def _mutate_connection(self, node1: NEATNode, node2: NEATNode, curr_gen_mutations: set, innov: int):
        from_node, to_node = (node1, node2) if node1.x < node2.x else (node2, node1)
        mutated_conn = Connection(from_node, to_node)
        # Make sure nodes aren't equal and there isn't already a connection between them
        if node1.id == node2.id or mutated_conn in self.connections or Connection(to_node,
                                                                                  from_node) in self.connections:
            return self

        # Check if the connection exist somewhere else
        mutation = ConnectionMutation(mutated_conn)
        if mutation not in curr_gen_mutations:
            innov += 1
            mutated_conn.innovation = innov
            curr_gen_mutations.add(mutation)
        else:
            for prev_mutation in curr_gen_mutations:
                if mutation == prev_mutation:
                    mutated_conn.innovation = prev_mutation.conn.innovation
                    break

        self.add_connection(mutated_conn)

        if from_node.x > to_node.x or from_node.id == to_node.id:
            print('Mutated and received node with bad x values or connections points from node to same node')

        return innov

    def mutate(self, curr_gen_mutations: set, innov: int, node_id: int, node_chance=0.03, conn_chance=0.5):
        chance = random.random()

        if chance < node_chance:  # Add a new node
            self._mutate_add_node(random.choice(self.connections), curr_gen_mutations, innov, node_id)
        elif chance < conn_chance:  # Add a new connection
            self._mutate_connection(random.choice(self.nodes), random.choice(self.nodes), curr_gen_mutations, innov)
