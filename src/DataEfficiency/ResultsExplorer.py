import os
import sys

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_2 = os.path.split(dir_path)[0]
sys.path.append(dir_path_1)
sys.path.append(dir_path_2)

from src.DataEfficiency import Net,DataEfficiency
import math
import matplotlib.pyplot as plt
import copy

networks = [Net.StandardNet, Net.BatchNormNet, Net.DropOutNet]
#networks = [Net.BatchNormNet]


def get_all_networks():
    all_networks = []
    for network_type in networks:
        for i in range(6):
            size = int(math.pow(2, i))
            model = network_type(size)
            all_networks.append(model)

    return all_networks

def plot_verbose_against_summarised():


    for model_type in networks:
        verbose_points = []
        summarised_points = []

        for i in range(6):
            size = int(math.pow(2, i))
            model = model_type(size)

            verbose, summarised = model.get_verbose_and_summarised_results()
            if verbose is None or summarised is None:
                continue
            verbose_points.append(DataEfficiency.solve_for_learning_rate(verbose))
            summarised_points.append(DataEfficiency.solve_for_learning_rate(summarised))
            print(repr(model_type),size, "verbose:", verbose_points[-1], "summarised:", summarised_points[-1])

        plt.plot(verbose_points, summarised_points, label=repr(model_type))

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels)
    plt.xlabel("Verbose DE")
    plt.ylabel("Summarised DE")
    plt.show()

def plot_size_vs_DE(model_type):

    options = [True, False]
    for verbose in options:
        sizes = []
        DEs = []
        verbose_string = "verbose" if verbose else "summarised"
        for i in range(6):
            sizes.append(int(math.pow(2, i)))
            model = model_type(sizes[-1])
            DE = DataEfficiency.solve_for_learning_rate(model.get_results(verbose))
            if not verbose:
                DE*=15
            DEs.append(DE)
        plt.plot(sizes,DEs, label = repr(model_type) + " "+verbose_string)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels)
    plt.xlabel("Network Size")
    plt.ylabel("DE")
    plt.show()

def plot_data_efficiency_curves(verbose):
    for model_type in networks:
        for i in range(6):
            size = int(math.pow(2, i))
            model = model_type(size)
            if not model.does_net_have_results_file(verbose):
                continue
            verbose_results, summarised = model.get_verbose_and_summarised_results()
            if verbose:
                results = verbose_results
            else:
                results = summarised
            lr = DataEfficiency.solve_for_learning_rate(results)
            DataEfficiency.plot_tuples_with_best_fit(results, lr, title=repr(model_type) + repr(size) + ": lr=" + repr(lr) +
                                                                           " , de=" + repr(DataEfficiency.get_data_efficiency(copy.deepcopy(results))))


if __name__ == "__main__":
    plot_data_efficiency_curves(True)
    #plot_verbose_against_summarised()
    # for model_type in networks:
    #     plot_size_vs_DE(model_type)