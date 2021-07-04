from utils import load_split_dataset, normalise_from_dataset_disjoint, normalise_from_dataset_joint
from utils import seed_reproducer
import visuals


if __name__ == '__main__':
    data, train, test = load_split_dataset()
    
    norm_train = normalise_from_dataset_joint(train)
    norm_test = normalise_from_dataset_joint(test)
    norm_data = normalise_from_dataset_joint(data)
    
    # Classes distribution
    visuals.save_distribution(norm_train, "norm_train.png")
    visuals.save_distribution(norm_test, "norm_test.png")
    visuals.save_distribution(norm_data, "norm_data.png")

    # Samples visualization
    visuals.save_samples(data, norm_data, "samples")

    # Statistics on testing of a trained model
    visuals.save_test_results("dense55ruleresults")

    # Visualizing some duplicates that are being dropped
    visuals.visualize_duplicates(data, "test", 3)
