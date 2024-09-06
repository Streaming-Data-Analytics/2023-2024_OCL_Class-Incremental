import os
import torchvision
import torch
from typing import Optional
from avalanche.benchmarks.utils import AvalancheDataset, DataAttribute, as_taskaware_classification_dataset
from avalanche.training.storage_policy import ClassBalancedBuffer

def load_CLEAR(folder: str, transform=None):
    folders = sorted(os.listdir(folder))
    all_images = {}
    for subfolder in folders:
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            class_folders = sorted(os.listdir(subfolder_path))

            for class_folder in class_folders:
                class_folder_path = os.path.join(subfolder_path, class_folder)
                if os.path.isdir(class_folder_path):
                    images = []

                    for filename in os.listdir(class_folder_path):
                        img_path = os.path.join(class_folder_path, filename)
                        img = torchvision.io.read_image(
                            img_path,
                            mode=torchvision.io.ImageReadMode.RGB
                            )
                        img = transform(img)
                        if img is not None:
                            images.append(img)

                    if subfolder not in all_images:
                        all_images[subfolder] = {}
                    all_images[subfolder][class_folder] = torch.stack(images) / 255
    return all_images

"""def build_CLEAR_train_experiences(CLEAR_dict, classes_pairs, mb_size: int=10, n_classes: int=2):
    tasks = {task_label: task_classes for task_label, task_classes in enumerate(classes_pairs)}
    experiences = []
    for task_label, task_classes in tasks.items():
        for time in range(1, 11):
            # Concatenate the features for both the classes
            x_data = torch.vstack(
                tuple(
                    CLEAR_dict[str(time)][curr_class] for curr_class in task_classes if curr_class is not None
                )
            )
            # Compute the number of samples for each class
            n_samples_per_class = [
                len(CLEAR_dict[str(time)][curr_class]) \
                                    for curr_class in task_classes if curr_class is not None
            ]
            # Build the class labels for both the classes
            class_labels = torch.concatenate(
                tuple(
                    torch.ones(n_samples, dtype=torch.int8) * (i + n_classes * task_label) \
                        for i, n_samples in enumerate(n_samples_per_class)
                )
            )
            # Build the task labels for both the classes
            task_labels = torch.ones(len(x_data), dtype=torch.int8) * task_label
            # Split the current task (n classes + fixed year) in minibatches, 
            # and shuffle so that each minibatch contains both classes
            permutations = torch.randperm(len(x_data))
            features_perm = x_data[permutations]
            class_labels_perm = class_labels[permutations]
            task_labels_perm = task_labels[permutations]
            for mb_pos in range(0, len(features_perm), mb_size):
                # Shuffle the samples
                features_mb = features_perm[mb_pos:(mb_pos+mb_size)]
                class_labels_mb = class_labels_perm[mb_pos:(mb_pos+mb_size)]
                task_labels_mb = task_labels_perm[mb_pos:(mb_pos+mb_size)]
                # Build the torch dataset and the Avalanche dataset 
                torch_data = torch.utils.data.dataset.TensorDataset(features_mb)
                av_dataset = AvalancheDataset(
                    datasets=torch_data,
                    data_attributes=[
                        DataAttribute(task_labels_mb, name="targets_task_labels", use_in_getitem=True),
                        DataAttribute(class_labels_mb, name="targets", use_in_getitem=True)
                    ]
                )
                class_dataset = as_taskaware_classification_dataset(av_dataset)
                # Append the current dataset to the whole set of experiences
                experiences.append(class_dataset)
    return experiences"""

def build_CLEAR_train_experiences(CLEAR_dict, classes_pairs, n_classes: int=2,
                                  subsample: Optional[int]=None):
    tasks = dict(enumerate(classes_pairs))
    experiences = []
    for task_label, task_classes in tasks.items():
        curr_experience_features = []
        curr_experience_labels = []
        for time in range(1, 11):
            if subsample:
                # Concatenate the features for both the classes
                x_data = torch.vstack(
                    tuple(
                        CLEAR_dict[str(time)][curr_class][:subsample] for curr_class in task_classes if curr_class is not None
                    )
                )
                # Compute the number of samples for each class
                n_samples_per_class = [
                    len(CLEAR_dict[str(time)][curr_class][:subsample]) \
                                        for curr_class in task_classes if curr_class is not None
                ]
            else:
                # Concatenate the features for both the classes
                x_data = torch.vstack(
                    tuple(
                        CLEAR_dict[str(time)][curr_class] for curr_class in task_classes if curr_class is not None
                    )
                )
                # Compute the number of samples for each class
                n_samples_per_class = [
                    len(CLEAR_dict[str(time)][curr_class]) \
                                        for curr_class in task_classes if curr_class is not None
                ]
            # Build the class labels for both the classes
            class_labels = torch.concatenate(
                tuple(
                    torch.ones(n_samples, dtype=torch.int8) * (i + n_classes * task_label) \
                        for i, n_samples in enumerate(n_samples_per_class)
                )
            )
            # Split the current task (n classes + fixed year) in minibatches, 
            # and shuffle so that each minibatch contains both classes
            permutations = torch.randperm(len(x_data))
            features_perm = x_data[permutations]
            class_labels_perm = class_labels[permutations]
            curr_experience_features.append(features_perm)
            curr_experience_labels.append(class_labels_perm)
        curr_experience_features = torch.vstack(curr_experience_features)
        curr_experience_labels = torch.concatenate(curr_experience_labels)
        task_labels = torch.ones(len(curr_experience_features), dtype=torch.int8) * task_label
        torch_data = torch.utils.data.dataset.TensorDataset(curr_experience_features)
        av_dataset = AvalancheDataset(
            datasets=torch_data,
            data_attributes=[
                DataAttribute(task_labels, name="targets_task_labels", use_in_getitem=True),
                DataAttribute(curr_experience_labels, name="targets", use_in_getitem=True)
            ]
        )
        class_dataset = as_taskaware_classification_dataset(av_dataset)
        # Append the current dataset to the whole set of experiences
        experiences.append(class_dataset)
    return experiences

def build_CLEAR_test_experiences(CLEAR_dict, classes_pairs, n_classes: int=2):
    tasks = dict(enumerate(classes_pairs))
    experiences = []
    for task_label, task_classes in tasks.items():
        # Concatenate the features for both the classes
        x_data = torch.vstack(
            tuple(
                CLEAR_dict[str(time)][curr_class] for curr_class in task_classes if curr_class is not None \
                    for time in range(1, 11)
            )
        )
        # Compute the number of samples for each class
        n_samples_per_class = [
            sum(len(CLEAR_dict[str(time)][curr_class]) for time in range(1, 11)) \
                for curr_class in task_classes if curr_class is not None
        ]
        
        # Build the class labels for both the classes
        class_labels = torch.concatenate(
            tuple(
                torch.ones(n_samples, dtype=torch.int8) * (i + n_classes * task_label) \
                    for i, n_samples in enumerate(n_samples_per_class)
            )
        )
        # Build the task labels for both the classes
        task_labels = torch.ones(len(x_data), dtype=torch.int8) * task_label
        # Build torch and Avalanche dataset
        torch_data = torch.utils.data.dataset.TensorDataset(x_data)
        av_dataset = AvalancheDataset(
            datasets=torch_data,
            data_attributes=[
                DataAttribute(task_labels, name="targets_task_labels", use_in_getitem=True),
                DataAttribute(class_labels, name="targets", use_in_getitem=True)
            ]
        )
        class_dataset = as_taskaware_classification_dataset(av_dataset)
        # Append the current dataset to the whole set of experiences
        experiences.append(class_dataset)
    return experiences