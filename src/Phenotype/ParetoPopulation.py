import copy
import time

from src.Analysis.DataPlotter import plot_acc_vs_second, plot_histogram
from src.Config import Config
from src.NEAT.PopulationRanking import general_pareto_sorting
from src.Validation import Validation


class ParetoPopulation:
    def __init__(self):
        self.pareto_front = []
        self.candidates = []
        self.best_members = []
        self.worst_das = []

    def queue_candidate(self, candidate):
        # print("queuing candidate:",candidate)
        self.candidates.append(copy.deepcopy(candidate))

    def update_pareto_front(self):
        start_time = time.time()
        # print("updating pareto pop from",len(self.candidates),"candidates and",len(self.pareto_front),"in front", end = " ")
        self.best_members.append(self.get_highest_accuracy(1, check_set=self.candidates))
        if Config.evolve_data_augmentations:
            self.worst_das.append(self.get_worst_da_from_candidates())
            # print("worst das:",self.worst_das)

        self.pareto_front = general_pareto_sorting(self.candidates + self.pareto_front, return_pareto_front_only=True)
        # print("after:",len(self.pareto_front),"in front time:", (time.time() - start_time))
        # print("candidates:",repr(self.candidates))

        if len(self.pareto_front) == 0:
            raise Exception("pareto front empty after step")

        self.candidates = []
        # self.plot_fitnesses()
        # self.plot_all_in_pareto_front()
        # self.get_highest_accuracy(print=True)

    def get_best_network(self, num_augs=1):
        best_graphs = self.get_highest_accuracy(num=max(len(self.best_members) - 8, 1), check_set=self.best_members)
        best = best_graphs[0]
        # print("num:",(len(self.best_members) - 8))
        print("fully training", Config.run_name, "reported acc:", best.fitness_values[0])

        augs = [x.data_augmentation_schemes[0] for x in best_graphs if len(x.data_augmentation_schemes) > 0]
        aug_names = set()
        unique_augs = []
        for aug in augs:
            name = repr(aug).split("Nodes:")[1].replace("'No_Operation'", "").replace("[]", "").replace('\\n',
                                                                                                        "").replace(",",
                                                                                                                    "").replace(
                '"', "").replace(" ", "")

            # print("name:",name)
            if name not in aug_names:
                aug_names.add(name)
                unique_augs.append(aug)
        unique_augs = unique_augs[:num_augs]
        return Validation.get_fully_trained_network(best, unique_augs, num_epochs=Config.num_epochs_in_full_train)

    def plot_fitnesses(self):
        # print("lengths:" , repr([len(x.fitness_values) for x in self.pareto_front]))
        # print("pop:",self.pareto_front)
        accuracies = [x.fitness_values[0] for x in self.pareto_front]
        num_objectives = len(self.pareto_front[0].fitness_values)
        if num_objectives == 1:
            plot_histogram(accuracies)
        elif num_objectives == 2:
            plot_acc_vs_second(accuracies, [x.fitness_values[1] for x in self.pareto_front])
        elif num_objectives == 3:
            pass
        else:
            raise Exception(">3 objectives")

    def plot_all_in_pareto_front(self):
        for graph in self.pareto_front:
            graph.module_graph_root_node.plot_tree_with_graphvis(file="fitnesses=" + repr(graph.fitness_values))

    def get_highest_accuracy(self, num, plot_best=False, check_set=None):
        highest_acc = 0
        best_graph = None
        if check_set is None:
            check_set = self.pareto_front

        if num > 1:
            # print("getting top", num, "graphs from", self.pareto_front )
            acc_sorted = sorted(check_set, key=lambda x: x.fitness_values[0], reverse=True)
            # print('len sorted:', len(acc_sorted))
            num_best_graphs = acc_sorted[:num]
            return num_best_graphs

        elif num == 1:

            for graph in check_set:
                if graph.fitness_values[0] > highest_acc:
                    highest_acc = graph.fitness_values[0]
                    best_graph = graph

            if plot_best:
                best_graph.plot_tree_with_graphvis("best acc graph in pareto population - acc=" + repr(highest_acc))

            return best_graph
        else:
            raise Exception("Number of graphs chosen is negative")

    def get_worst_da_from_candidates(self):
        worst = None
        worst_acc = 9999999
        for mod_graph in self.candidates:
            da = mod_graph.data_augmentation_schemes[0]
            if da.fitness_values[0] < worst_acc:
                worst = da
                worst_acc = da.fitness_values[0]
        return worst
