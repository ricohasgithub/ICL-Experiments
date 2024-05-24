import torch
import numpy as np
from datasets.dataset import Dataset
from utils import generate_input_vis, generate_label_vis, generate_prompt_vis

INPUT_DIM = 10
N_CLASSES = 3
PROMPT_LENGTH = 10

train_dataset = Dataset(INPUT_DIM, N_CLASSES, PROMPT_LENGTH)

prompt = train_dataset.get_prompt()

# generate_input_vis(prompt)
# generate_label_vis(prompt)
generate_prompt_vis(prompt)