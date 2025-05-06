import numpy as np


def compute_class_weights(labels, ds_total_len=2, augment_prob=0.5):
    """
    Compute sampling weights for original and augmented data.
    :param labels: np.array, original dataset labels
    :param ds_total_len: Dataset size multiplier (default = 2x original)
    :param augment_prob: Probability of selecting augmented samples
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Calculate the category sampling weight (the fewer category samples, the greater the weight)
    class_weights = {label: 1.0 / count for label, count in zip(unique_labels, counts)}

    # Original data weight
    original_sample_weights = np.array([class_weights[label] for label in labels])

    # Copy labels so that augmented_labels and original_labels keep the same category ratio
    augmented_labels = np.tile(labels, max(int(ds_total_len - 1), 1))
    augmented_sample_weights = np.array([class_weights[label] for label in augmented_labels])

    # Calculate how much original` vs. augmented should be in batch
    num_original = len(original_sample_weights)
    num_augmented = len(augmented_sample_weights)
    total_samples = num_original + num_augmented

    # Control the proportion of augmented data in batch
    if num_augmented == 0:
        augmented_ratio = 0
    else:
        augmented_ratio = augment_prob * total_samples / num_augmented

    original_ratio = (1 - augment_prob) * total_samples / num_original

    final_sample_weights = np.concatenate([
        original_sample_weights * original_ratio,  
        augmented_sample_weights * augmented_ratio 
    ])

    return final_sample_weights