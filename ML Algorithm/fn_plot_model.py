import matplotlib.pyplot as plt


def plot_model(name, fig_num, data_list, leg_loc):
    """
    Plots the training loss, training accuracy, and validation accuracy histories of the model
    :param name: name of the model
    :param fig_num: figure number of the matplotlib figure
    :param data_list: the list of data that is being plotted
    :param leg_loc: the location of the legend on the figure
    :return: the matplotlib figure
    """
    title = "Train Loss, Train Accuracy, and Test Accuracy of " + name
    plot = plt.figure(fig_num)
    for data in data_list:
        x = range(1, len(data)+1)   # Creates the x-axis
        plt.plot(x, data)   # Plots the data
    plt.legend(["Training Loss", "Training Accuracy", "Testing Accuracy"], loc=leg_loc)
    plt.xlabel('Epoch')
    plt.ylabel('Error/Accuracy')
    plt.title(title)
    plt.savefig(name + ".png")
    return plot
