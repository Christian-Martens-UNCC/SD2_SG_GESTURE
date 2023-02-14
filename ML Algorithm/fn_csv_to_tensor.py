import numpy as np
from fn_try_gpu import *


def csv_to_tensor(csv_path):
    """
    Takes a csv file and formats it to a normalized tensor for training an ML model
    :param csv_path: the csv file path of the data you with to convert to a normalized tensor
    :return: the normalized features and the output class associated with that input in the same tensor, saved to
             the gpu if one is available
    """

    data_numpy = np.loadtxt(csv_path, dtype=np.float32, delimiter=",", skiprows=1)  # Loads the csv as a numpy array
    data_raw = torch.from_numpy(data_numpy)     # Converts the numpy array to a tensor

    data_features = data_raw[:, :-1]    # Calls all the features of the tensor (all but the last column)
    torch.reshape(data_features, (-1, data_features.shape[1]))  # Reshapes the tensor to the correct dimensions

    data_class = data_raw[:, -1].long()     # Calls the classification column of the tensor
    data_class = torch.unsqueeze(data_class, dim=-1)    # Reshapes the tensor to the correct dimensions

    data_mean = torch.mean(data_features, dim=1).unsqueeze(-1)  # Calculates the mean of the tensor
    data_var = torch.var(data_features, dim=1).unsqueeze(-1)    # Calculates the variance of the tensor
    data_normalized = (data_features - data_mean) / torch.sqrt(data_var)    # Calculates the normalization of the tensor

    return torch.cat((data_normalized, data_class), dim=1).to(device=try_gpu())
