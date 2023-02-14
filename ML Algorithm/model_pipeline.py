import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
from fn_train_model import *
from fn_plot_model import *
from fn_csv_to_tensor import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from fn_model_1_3 import *  # The model that is being trained
data_path = r"C:\Users\ccm51\Documents\SD_SG_GESTURE\master_dataset.csv"    # The path for the dataset you are training on
train_size = 0.8    # The size of the training group size out of 1 (test/validation group size is 1-train_size)
batch_size = 100    # The number of datapoint in a batch
learning_rate = 10**-3  # The learning rate of the model
weight_decay_rate = 10**-4  # The weight-decay ratio of the model. Prevents over-fitting
epochs = 100    # Number of epochs to train for
update_freq = 10    # Number of epochs between training updates
model_name = "nn_model_1-3"     # Saved name of the model's parameters


data = csv_to_tensor(data_path)     # Loads the csv file and converts it to a normalized tensor
train, test = train_test_split(data, train_size=train_size, test_size=(1-train_size))   # Splits training and test groups
model = init_model()    # Initializes the untrained model
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)     # Initializes the optimizer equation for the untrained model
print(model.eval())     # Prints model structure to validate the correct model has been loaded
t_loss, t_acc, v_acc = train_model(model,
                                   train,
                                   test,
                                   nn.CrossEntropyLoss(),
                                   optimizer,
                                   epochs,
                                   batch_size,
                                   update_freq)     # Trains the model and stores its training history
plot_model(model_name, 1, [t_loss, t_acc, v_acc], 'upper right')  # Plots the training history of the trained model
torch.save(model.state_dict(), model_name + ".params")  # Saves the parameters of the trained model
