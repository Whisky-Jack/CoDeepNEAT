import sys
import os

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_2 = os.path.split(dir_path)[0]
sys.path.append(dir_path_1)
sys.path.append(dir_path_2)

import torch
from src.DataEfficiency import Net
from src.Validation.DataLoader import load_data
import math
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim



criterion = nn.CrossEntropyLoss()
networks = [Net.BatchNormNet, Net.StandardNet, Net.DropOutNet]
total_batches = None


def test_model(model):
    model.eval()
    correct = 0
    total = 0

    trainloader, testloader = load_data(dataset="cifar10")

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(torch.device("cuda:0")), labels.to(torch.device("cuda:0"))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    #print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))

    return accuracy

def run_epoch_for_n_batches(model,optimiser, num_batches = -1):
    trainloader, testloader = load_data(dataset="cifar10")

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if i+2 >= num_batches > 0:
            return
        #print("training",model)
        inputs, labels = data
        inputs,labels = inputs.to(torch.device("cuda:0")), labels.to(torch.device("cuda:0"))

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

def run_model_over_different_batch_numbers(num_epochs, model_type, size, verbose):
    accuracies = []#tuples of (%training_set, %accuracy)

    total_trains = num_epochs * total_batches
    model = model_type(size).to(torch.device("cuda:0"))
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    fractions = [0,1,2,3,4,5,7,9,12,16,20,40,70,100]
    for fraction in fractions:
        if fraction == 0:
            num_batches = 1
        else:
            num_batches = fraction*total_batches/100

        if verbose and fraction > fractions[0]:
            model = model_type(size).to(torch.device("cuda:0"))
            optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        elif not verbose:
            """if summarised - the same model is used to train on each batch subset, on fewer epochs each"""
            num_epochs = round(total_trains/num_batches)
            print('num batches:',num_batches, "num epochs:",num_epochs)

        if model.does_net_have_results_file(verbose):
            print("model",model.get_name(), "already has results saved:", model.get_results(verbose))
            return []

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            run_epoch_for_n_batches(model,optimiser,num_batches=num_batches)

        accuracy = test_model(model)
        training_proportion = 100*num_batches/total_batches

        print(model.get_name(),("verbose" if verbose else "summarised"),"num_batches:",num_batches,"/",total_batches,("("+str(round(training_proportion))+"%)"),"acc:",accuracy,"%")


        accuracies.append((training_proportion,accuracy))

    model.save_results(accuracies, verbose)


    return accuracies

def test_all_networks(num_epochs, verbose):
    plot_points = []

    for network_type in networks:
        for i in range(7):
            size = int(math.pow(2,i))
            accuracies = run_model_over_different_batch_numbers(num_epochs,network_type,size, verbose)
            #plot_model_accuracies(accuracies, network_type)
            plot_points.append((accuracies,network_type(size).get_name()))

    #plot_all_verbose_accuracies(plot_points)


def test_max_accuracy_of_networks(num_epochs):
    for network_type in networks:
        for i in range(6):
            size = int(math.pow(2, i))
            model = network_type(size).to(torch.device("cuda:0"))
            optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            for epoch in range(num_epochs):  # loop over the dataset multiple times

                run_epoch_for_n_batches(model, optimiser, num_batches=total_batches)

            accuracy = test_model(model)
            print(model.get_name(),"max acc:",accuracy)

def plot_all_accuracies(values):
    for model_values in values:
        accuracies, model_name = model_values
        plt.plot([list(x)[0] for x in accuracies], [list(x)[1] for x in accuracies], label = model_name)
        plt.xlabel("% of full training set")
        plt.ylabel("% classification accuracy")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles,labels)
    plt.show()

def run_tests():

    global total_batches
    trainloader, testloader = load_data(dataset="cifar10")
    total_batches = len(trainloader)
    #test_max_accuracy_of_networks(num_epochs)
    test_all_networks(20, True)
    test_all_networks(2, False)



if __name__ == "__main__":
    run_tests()
