import torch

device = torch.device('cuda:0')
# device = torch.device('cpu')

second_objective = "network_size"  # network_size
third_objective = ""

test_in_run = True
dummy_run = True
protect_parsing_from_errors = False

number_of_epochs_per_evaluation = 3

save_best_graphs = True
print_best_graphs = False
print_best_graph_every_n_generations = 2
save_failed_graphs = True
