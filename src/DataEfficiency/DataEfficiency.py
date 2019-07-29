import math
import sys
import matplotlib.pyplot as plt

"""
DataEfficiency function:

acc = max_acc * ln( set_fraction + A / A) * A^fix           solve for A given max_acc and best fit line of all the (set_fraction,acc) tuples

where   A = 1/learning_rate                                 the primary data efficiency component
        fix = log(0.434294482/log((100+A)/A))/log(A)        a solution to fix the point (100,max_acc)

        learning rate >0 ≠ 1                                as learning rate->∞ the model gets closer to instant learning
                a measure of how quickly a model learns given its data. models with higher learning_rates would perform better with less data

        data_efficiency = max_acc * learning_rate^2         ?possibly change to keep DE constant for a model under data restriction

"""


def get_data_efficiency(tuples):
    """
    :param tuples: list of (training_set_fraction, accuracy) tuples
    :return: the data efficiency value for the given (training_set_fraction, accuracy) tuples
    """
    _,max_acc = tuples[-1]
    return math.pow(solve_for_learning_rate(tuples), 0.1) * max_acc

def solve_for_learning_rate(tuples):
    """
    adaptively tries to minimise MSE wrt learning rate
    :param tuples: a list of (training_set_fraction, accuracy) tuples to fit
    :return: the closest learning rate which could be found, which best fits the DE curve over the given tuples
    """

    first_set,  first_acc = tuples[0]
    if round(first_set) == 0:
        print("starting with 0")
        tuples[0] = (0,0)
        for i in range(1,len(tuples)):
            set,acc = tuples[i]
            tuples[i] = (set, acc - first_acc)

    learning_rate = 0.000001

    min_MSE = sys.maxsize
    best_learning_rate = -1

    change_frac = 2#decreasing change frac means more sensitive updating of learning rate

    previous_MSE = min_MSE
    dir = 1 #either 1 or -1. 1 for increase lr, -1 for decrease lr
    for i in range(10000):
        MSE = get_mean_squared_error(tuples, learning_rate)
        if MSE<min_MSE:
            min_MSE = MSE
            best_learning_rate = learning_rate
            #print("new best learning rate:",best_learning_rate, "MSE:",MSE)

        if MSE > previous_MSE:
            """gotten worse - change dir, decrease change_frac"""
            dir*=-1
            change_frac = math.pow(change_frac,0.85)

        learning_rate*= math.pow(change_frac,dir)
        previous_MSE = MSE
    return best_learning_rate


def get_mean_squared_error(tuples, learning_rate):
    _,max_acc = tuples[-1]
    squared_error = 0
    for set_fraction, acc in tuples:
        predicted_acc = get_predicted_accuracy(max_acc, set_fraction, learning_rate)
        squared_error += math.pow(predicted_acc-acc, 2)
    return squared_error

def get_predicted_accuracy(max_accuracy, training_set_fraction, learning_rate):
    A = 1/learning_rate
    return max_accuracy * math.log((training_set_fraction + A)/A) * math.pow(A,get_fix_number(learning_rate))

def get_fix_number(learning_rate):
    A = 1/learning_rate
    if A == 1:
        A = 1.01
    try:
        return math.log(0.434294482/math.log((100+A)/A, 10), 10)/math.log(A,10)
    except:
        raise Exception("error getting fix number at lr=",learning_rate,"A=",A)

def plot_tuples_with_best_fit(tuples, learning_rate = None, title = ""):
    if learning_rate is None:
        learning_rate = solve_for_learning_rate(tuples)
    _,max_acc = tuples[-1]


    plt.plot([list(x)[0] for x in tuples], [list(y)[1] for y in tuples], label = "data")
    plt.plot([x/5 for x in range(500)], [get_predicted_accuracy(max_acc, x/5, learning_rate) for x in range(500)], label = "best fit" )
    plt.title("best fit for " + title)
    plt.xlabel("% of full training set")
    plt.ylabel("% classification accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.show()


