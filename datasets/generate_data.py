import torch

import numpy as np


INPUT_DIM = 105
N_CLASSES = 1623

def uniform_class(n_classes):
    return np.random.randint(n_classes)

class Dataset:
    
    def __init__(self, input_dim, n_classes, prompt_length, n_prompts, class_generator = uniform_class):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.prompt_length = prompt_length
        self.n_prompts = n_prompts
        
        
        # Create class means
        self.class_means = np.random.rand(self.input_dim, self.n_classes)
        
        # Parameters for the Gaussian noise on each class
        self.noise_mean = np.zeros(shape = (input_dim, self.n_classes))
        self.noise_cov = np.stack([np.identity(self.input_dim) for _ in range(self.n_classes)], axis = 0)
        
        

        self.labels = np.random.permutation(self.n_classes)
        
        self.train_data = np.zeros(shape = (self.n_prompts, self.input_dim + 1, self.prompt_length))
        
        for n in range(self.n_prompts):
            for t in range(self.prompt_length):
                
                cur_class = class_generator(self.n_classes)
                
                cur_mean = self.class_means[:, cur_class]
                cur_noise = np.random.multivariate_normal(self.noise_mean[:, cur_class], self.noise_cov[cur_class])
                
                cur_input = cur_mean + cur_noise
                cur_label = self.labels[cur_class]
                
                self.train_data[n, :, t] = np.vstack([cur_input, cur_label])

