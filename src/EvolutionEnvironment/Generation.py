import copy

import src.Config.NeatProperties as Props
import src.Validation.DataLoader
from src.NEAT.Population import Population
from src.NEAT.PopulationRanking import single_objective_rank, cdn_rank, nsga_rank
from src.Validation import Evaluator
from src.CoDeepNEAT import PopulationInitialiser as PopInit
from src.Analysis import RuntimeAnalysis
from src.Config import Config
from data import DataManager
from src.NeuralNetwork.ParetoPopulation import ParetoPopulation
from src.Validation import DataLoader
from src.Validation import Validation

import multiprocessing as mp


class Generation:

    def __init__(self):
        self.speciesNumbers = []
        self.module_population, self.blueprint_population, self.da_population = None, None, None
        self.initialise_populations()
        self.generation_number = -1
        self.pareto_population = ParetoPopulation()

    def initialise_populations(self):
        # Picking the ranking function
        rank_fn = single_objective_rank if Config.second_objective == '' else (
            cdn_rank if Config.moo_optimiser == "cdn" else nsga_rank)

        self.module_population = Population(PopInit.initialise_modules(),
                                            rank_fn,
                                            PopInit.initialize_mutations(),
                                            Props.MODULE_POP_SIZE,
                                            2,
                                            2,
                                            Props.MODULE_TARGET_NUM_SPECIES)

        self.blueprint_population = Population(PopInit.initialise_blueprints(),
                                               rank_fn,
                                               PopInit.initialize_mutations(),
                                               Props.BP_POP_SIZE,
                                               2,
                                               2,
                                               Props.BP_TARGET_NUM_SPECIES)

        if Config.evolve_data_augmentations:
            self.da_population = Population(PopInit.initialise_da(),
                                            rank_fn,
                                            PopInit.da_initial_mutations(),
                                            Props.DA_POP_SIZE,
                                            1,
                                            1,
                                            Props.DA_TARGET_NUM_SPECIES)

    def step(self):
        """Runs CDN for one generation - must be called after fitness evaluation"""
        self.pareto_population.update_pareto_front()
        # self.pareto_population.plot_fitnesses()
        self.module_population.step()
        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.reset_number_of_module_species(self.module_population.get_num_species())

        self.blueprint_population.step()

        if Config.evolve_data_augmentations:
            self.da_population.step()

        for blueprint_individual in self.blueprint_population.individuals:
            blueprint_individual.end_step()

        for module_individual in self.module_population.individuals:
            module_individual.end_step()  # this also sets fitness to zero

        DataManager.save_generation_state(self)

    def evaluate(self, generation_number):
        self.generation_number = generation_number

        procs = []
        manager = mp.Manager()
        results_dict = manager.dict()
        bp_index = manager.Value('i', 0)
        lock = mp.Lock()

        for i in range(Config.num_gpus):
            procs.append(mp.Process(target=self._evaluate, args=(lock, bp_index, results_dict), name=str(i)))
            procs[-1].start()
        [proc.join() for proc in procs]

        accuracies, second_objective_values, third_objective_values = [], [], []
        bp_pop_size = len(self.blueprint_population)
        bp_pop_indvs = self.blueprint_population.individuals

        for bp_key, (fitness, evaluated_bp, module_graph) in results_dict.items():
            if fitness == 'defective':
                bp_pop_indvs[bp_key % bp_pop_size].defective = True
                continue

            # Validation
            if evaluated_bp.eq(bp_pop_indvs[bp_key % bp_pop_size]):
                raise Exception('Evaled bp topology not same as main one')
            if not evaluated_bp.modules_used_index:
                raise Exception('Modules used index is empty in evaluated bp', evaluated_bp.modules_used_index)
            if not evaluated_bp.modules_used:
                raise Exception('Modules used is empty in evaluated bp', evaluated_bp.modules_used)

            # Fitness assignment
            bp_pop_indvs[bp_key % bp_pop_size].report_fitness(fitness)

            if Config.evolve_data_augmentations and evaluated_bp.da_scheme_index != -1:
                self.da_population[evaluated_bp.da_scheme_index].report_fitness(fitness)

            print("reporting fitnesses to: ",  evaluated_bp.modules_used_index)
            for species_index, member_index in evaluated_bp.modules_used_index:
                print("reporting fitness",fitness, " to : ", self.module_population.species[species_index][member_index])
                self.module_population.species[species_index][member_index].report_fitness(fitness)

            # Gathering results for analysis
            accuracies.append(fitness[0])
            if len(fitness) > 1:
                second_objective_values.append(fitness[1])
            if len(fitness) > 2:
                third_objective_values.append(fitness[2])

            self.pareto_population.queue_candidate(module_graph)

        RuntimeAnalysis.log_new_generation(accuracies, generation_number,
                                           second_objective_values=(
                                               second_objective_values if second_objective_values else None),
                                           third_objective_values=(
                                               third_objective_values if third_objective_values else None))

    def _evaluate(self, lock, bp_index, result_dict):
        inputs, targets = DataLoader.sample_data(Config.get_device())
        blueprints = self.blueprint_population.individuals
        bp_pop_size = len(blueprints)

        while bp_index.value < Props.INDIVIDUALS_TO_EVAL:
            with lock:
                blueprint_individual = copy.deepcopy(blueprints[bp_index.value % bp_pop_size])
                bp_index.value += 1
                curr_index = bp_index.value - 1

            # Evaluating individual
            try:
                module_graph, blueprint_individual, results = self.evaluate_blueprint(blueprint_individual, inputs)
                result_dict[curr_index] = results, blueprint_individual, module_graph
            except Exception as e:
                result_dict[curr_index] = 'defective', False, False
                if not Config.protect_parsing_from_errors:
                    raise Exception(e)

    def evaluate_blueprint(self, blueprint_individual, inputs):
        # Validation
        if blueprint_individual.modules_used_index:
            raise Exception('Modules used index is not empty', blueprint_individual.modules_used_index)
        if blueprint_individual.modules_used:
            raise Exception('Modules used is not empty', blueprint_individual.modules_used)

        blueprint = blueprint_individual.to_blueprint()
        module_graph = blueprint.parseto_module_graph(self)
        net = src.Validation.Validation.create_nn(module_graph, inputs)

        if Config.dummy_run:
            acc = hash(net)
            if Config.evolve_data_augmentations:
                da_indv = blueprint_individual.pick_da_scheme(self.da_population)
                da_scheme = da_indv.to_phenotype()
        else:
            if Config.evolve_data_augmentations:
                da_indv = blueprint_individual.pick_da_scheme(self.da_population)
                da_scheme = da_indv.to_phenotype()
            else:
                da_scheme = None

            acc = Evaluator.evaluate(net, Config.number_of_epochs_per_evaluation, Config.get_device(), 256, augmentor=da_scheme)

        objective_names = [Config.second_objective, Config.third_objective]
        results = [acc]
        for objective_name in objective_names:

            if objective_name == "network_size":
                results.append(net.module_graph.get_net_size())
            elif objective_name == "":
                pass
            else:
                print("Error: did not recognise second objective", Config.second_objective)

        module_graph.delete_all_layers()
        return module_graph, blueprint_individual, results
