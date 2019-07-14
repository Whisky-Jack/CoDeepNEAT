# Most values taken from https://arxiv.org/pdf/1902.06827.pdf in the appendix
# Modules
MODULE_POP_SIZE = 60
MODULE_NODE_MUTATION_CHANCE = 0.08#0.08
MODULE_CONN_MUTATION_CHANCE = 0.08#0.08
MODULE_TARGET_NUM_SPECIES = 5

# Blueprints
BP_POP_SIZE = 30
BP_NODE_MUTATION_CHANCE = 0.16#0.16
BP_CONN_MUTATION_CHANCE = 0.12#0.12
BP_TARGET_NUM_SPECIES = 1

INDIVIDUALS_TO_EVAL = 100

PERCENT_TO_REPRODUCE = 0.2

MUTATION_TRIES = 100  # number of tries a mutation gets to pick acceptable individual

MIN_CHILDREN_PER_SPECIES = 2
ELITE_TO_KEEP = 0.1

# Speciation
SPECIES_DISTANCE_THRESH = 1
SPECIES_DISTANCE_THRESH_MOD = 0.15
TARGET_NUM_SPECIES = 4
