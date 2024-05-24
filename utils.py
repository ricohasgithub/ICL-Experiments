
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_label_vis(prompt, title = "Label Visualiation"):
    
    prompt_np = prompt.detach().numpy()
    
    labels = prompt_np[-1].reshape(1, -1)
    
    plt.figure()
    
    sns.heatmap(labels)
    
    plt.xlabel("Index")
    plt.title(title)
    
    plt.show()

def generate_input_vis(prompt, title = "Input Visualization"):

    prompt_np = prompt.detach().numpy()
    
    inputs = prompt_np[0:-1]
    
    plt.figure()
    
    sns.heatmap(inputs)
    
    plt.xlabel("Index")
    plt.ylabel("Input Values")
    plt.title(title)
    
    plt.show()

def generate_prompt_vis(prompt, title = "Prompt Visualization"):
    prompt_np = prompt.detach().numpy()
    
    inputs = prompt_np[0:-1]
    labels = prompt_np[-1].reshape(1, -1)
    
    plt.figure()
    
    plt.subplot(1, 2, 1)
    
    sns.heatmap(inputs)
    
    plt.ylabel("Input Values")
    plt.xlabel("Index")
    plt.title(title)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(labels)
    
    plt.xlabel("Index")
    plt.title(title)
    
    plt.show()
    

def generate_attention_vis(attention_matrix, title = "Attention Visualization"):
    plt.figure()
    
    sns.heatmap(attention_matrix)
    
    plt.xlabel("Token 1")
    plt.ylabel("Token 2")
    plt.title(title)
    
    plt.show()


