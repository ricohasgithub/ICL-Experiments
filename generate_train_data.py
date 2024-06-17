import torch
import numpy as np
from datasets.dataset import Dataset
from utils import generate_input_vis, generate_label_vis, generate_prompt_vis
from train import Trainer
from datasets.dataset import SeqGenerator
from datasets.dataset import OmniglotDatasetForSampling
import seaborn
import matplotlib.pyplot as plt

INPUT_DIM = 10
N_CLASSES = 3
PROMPT_LENGTH = 10

seq_generator_factory = SeqGenerator(
    dataset_for_sampling=OmniglotDatasetForSampling(
        omniglot_split="all",  # 1623 total classes
        exemplars="all",  # 'single' / 'separated' / 'all'
        augment_images=False,
    ),
    n_rare_classes=1603,  # 1623 - 20
    n_common_classes=10,
    n_holdout_classes=10,
    zipf_exponent=1,
    use_zipf_for_common_rare=False,
    noise_scale=0.0,
    preserve_ordering_every_n=None,
)

data_generator = lambda: seq_generator_factory.get_fewshot_seq(
    class_type="rare",
    shots=4,  # fs_shots
    ways=2,  # ways
    labeling="unfixed",
    randomly_generate_rare=False,  # randomly_generate_rare
    grouped=False,  # grouped
)

for sequence in data_generator():

    plt.figure(figsize=(20, 10))

    for i in range(len(sequence["example"])):
        plt.subplot(1, len(sequence["example"]), i + 1)
        seaborn.heatmap(sequence["example"][i][:, :, 0])
        plt.title("Label: " + str(sequence["label"][i]))

    plt.show()
