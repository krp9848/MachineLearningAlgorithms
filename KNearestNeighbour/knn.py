import numpy as np 
from collections import Counter 
import timeit

def calculate_mode(arr: list) -> str:
    """Calculate the mode for the given series

    Args:
        arr (list): Given Series 

    Returns:
        str: mode for the given series
    """

    if not (isinstance(arr, list) or isinstance(arr, np.ndarray)):
        raise TypeError("Argument not provided as list or numpy array")

    item_counter = Counter(arr)
    return max(item_counter, key= lambda k: item_counter.get(k))


def predict(k, train, train_labels, test, problem_type= 'classification'):
    """Predict the output for given test point

    Args:
        k (int): number of nearest neighbours selected
        train (list): training data
        train_labels (list): corresponding labels for training data
        test (list): test point for which the response or output is preidcted
        problem_type (str, optional): Specified problem type. Defaults to 'classification'.

    Returns:
        str or float: response or output label predicted for test point
    """

    if not isinstance(k, int):
        raise TypeError("Value of k must be an integer")

    if not (isinstance(train, (list, np.ndarray))):
        raise TypeError("Training dataset must in the form of numpy array or list")

    if not (isinstance(train_labels, (list, np.ndarray))):
        raise TypeError("Training labels must be provided in the form of numpy array or list")

    if not (isinstance(test, (list, np.ndarray))):
        raise TypeError("Test Point must in the form of numpy array or list")

    if not k in range(1, len(train_labels) + 1):
        raise ValueError(f"k must be in range 1<=k<={len(train_labels)}")

    if not len(train) == len(train_labels):
        raise ValueError(f"Lenght of train {len(train)} and train_labels {len(train_labels)}\
                            must match")
    

    #calculate the euclidean distance between every point in training dataset and test point 
    distances = [np.sqrt(np.sum(np.square(np.array(test) - np.array(x)))) for x in train]

    #store the response value for selected neighbours
    selected_neighbours = np.array(train_labels)[np.argsort(distances)[:k]]

    #return the label with majority vote in case of classification else mean for regression problem
    return calculate_mode(selected_neighbours) if problem_type == 'classification' \
        else np.mean(selected_neighbours)


