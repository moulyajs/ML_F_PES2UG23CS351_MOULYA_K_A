import torch


def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    """
    target = tensor[:, -1]  # last column is target
    classes, counts = torch.unique(target, return_counts=True)
    probs = counts.float() / counts.sum()

    entropy = -torch.sum(probs * torch.log2(probs))
    return float(entropy)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    """
    values, counts = torch.unique(tensor[:, attribute], return_counts=True)
    total = tensor.shape[0]

    avg_info = 0.0
    for v, count in zip(values, counts):
        subset = tensor[tensor[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += (count.item() / total) * subset_entropy

    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    """
    total_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)

    info_gain = total_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.
    """
    num_attributes = tensor.shape[1] - 1  # exclude target
    info_gains = {}

    for attr in range(num_attributes):
        info_gains[attr] = get_information_gain(tensor, attr)

    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr
