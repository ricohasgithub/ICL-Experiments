import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn

def generate_label_vis(prompt, title = "Label Visualiation"):
    
    prompt_np = prompt.detach().numpy()
    
    labels = prompt_np[-1].reshape(1, -1)
    
    plt.figure()
    
    seaborn.heatmap(labels)
    
    plt.xlabel("Index")
    plt.title(title)
    
    plt.show()

def generate_input_vis(prompt, title = "Input Visualization"):

    prompt_np = prompt.detach().numpy()
    
    inputs = prompt_np[0:-1]
    
    plt.figure()
    
    seaborn.heatmap(inputs)
    
    plt.xlabel("Index")
    plt.title(title)
    
    plt.show()

