from utils import load_split_dataset, normalise_from_dataset_disjoint, normalise_from_dataset_joint
from utils import seed_reproducer
import os
import visuals


if __name__ == '__main__':
    data, train, test = load_split_dataset()
    
    norm_train = normalise_from_dataset_joint(train)
    norm_test = normalise_from_dataset_joint(test)
    norm_data = normalise_from_dataset_joint(data)
    
    visuals.save_distribution(norm_train, "norm_train.png")
    visuals.save_distribution(norm_test, "norm_test.png")
    visuals.save_distribution(norm_data, "norm_data.png")

    
    visuals.save_samples(data, norm_data, "samples")


