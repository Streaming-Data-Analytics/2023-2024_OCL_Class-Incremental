import pandas as pd
import numpy as np

def AF(accuracy_df, steps_per_experience, n_experiences):
    # This dictionary contains the test accuracies on all the test Experiences,
    # organized by the Experiences seen during training.
    # e.g. accuracies_exp_3 contains as many rows as the training steps done for experience 3,
    # and has as many columns as the number of test experiences. 
    accuracies_by_experience = {
    f"accuracies_exp_{i}": accuracy_df.loc[(i * steps_per_experience) : (i+1) * steps_per_experience - 1] for 
        i in range(n_experiences)
    }
    # Computes the accuracy achieved at the end of training for each experience:
    final_accuracy_by_experience = [
            accuracies_by_experience[f"accuracies_exp_{i}"][f"accuracy_task_{i}"].values[-1] for 
                i in range(n_experiences)
    ]
    # Difference between the accuracy on past experiences achieved at each step and the accuracy
    # achieved at the final step of the training experience, averaged on all the past experiences. 
    average_forg = {
        f"AF_experience_{j}": (accuracies_by_experience[f"accuracies_exp_{j}"] - final_accuracy_by_experience)[
            [f"accuracy_task_{i}" for i in range(0, j)]].mean(axis=1) for j in range(1, n_experiences)
    }
    return pd.concat([af for af in average_forg.values()])

def min_ACC(accuracy_df, steps_per_experience, n_experiences):
    """
    Returns a metric for each Experience, except for the first one (since there are no
    experiences before that).
    """
    # This dictionary contains the test accuracies on all the test Experiences,
    # organized by the Experiences seen during training.
    # e.g. accuracies_exp_3 contains as many rows as the training steps done for experience 3,
    # and has as many columns as the number of test experiences. 
    accuracies_by_experience = {
    f"accuracies_exp_{i}": accuracy_df.loc[(i * steps_per_experience) : (i+1) * steps_per_experience - 1] for 
        i in range(n_experiences)
    }
    # This dict contains the minimum accuracy achieved at the j-th experience on past test experiences. 
    # It has as many rows as the number of previous experiences that came before the j-th one
    minimum_accuracy_previous_experiences = {
    f"average_acc_exp_{j}": accuracies_by_experience[f"accuracies_exp_{j}"][
        [f"accuracy_task_{i}" for i in range(0, j)]].min(axis=0) for j in range(1, n_experiences)
    }
    # Now we take the minimum (mean) accuracy achieved during each Experience
    average_minimum_accuracy = [
        np.mean(value) for value in minimum_accuracy_previous_experiences.values()
    ]
    return average_minimum_accuracy

def WC_ACC(accuracy_df, steps_per_experience, n_experiences):
    # This dictionary contains the test accuracies on all the test Experiences,
    # organized by the Experiences seen during training.
    # e.g. accuracies_exp_3 contains as many rows as the training steps done for experience 3,
    # and has as many columns as the number of test experiences. 
    accuracies_by_experience = {
    f"accuracies_exp_{i}": accuracy_df.loc[(i * steps_per_experience) : (i+1) * steps_per_experience - 1] for 
        i in range(n_experiences)
    }
    # Computes the accuracy achieved at the end of training for each experience:
    final_accuracy_by_experience = [
            accuracies_by_experience[f"accuracies_exp_{i}"][f"accuracy_task_{i}"].values[-1] for 
                i in range(n_experiences)
    ]
    # Compute the minACC using the already implemented function
    min_acc = min_ACC(
        accuracy_df=accuracy_df,
        steps_per_experience=steps_per_experience,
        n_experiences=n_experiences
    )
    wc_acc = [(final_accuracy_by_experience[i] / i) + (1 - 1 / i) * min_acc[i-1] for i in range(1, n_experiences)]
    return wc_acc