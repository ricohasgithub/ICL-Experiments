import torch
import numpy as np


def uniform_class(n_classes):
    return np.random.randint(n_classes)

def custom_class_dist(n_classes, dist_array):
    
    if sum(dist_array) != 1:
        raise Exception("Probabilities must sum to 1.")
    
    if len(dist_array) != n_classes:
        raise Exception("Distribution array length must match number of classes.")
    
    return np.random.choice(n_classes, 1, p = dist_array)

class Dataset:
    
    def __init__(self, input_dim, n_classes, prompt_length, class_generator = uniform_class):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.prompt_length = prompt_length
        self.class_generator = class_generator
        
        # Create class means
        self.class_means = np.random.rand(self.input_dim, self.n_classes)
        
        # Parameters for the Gaussian noise on each class
        self.noise_mean = np.zeros(shape = (input_dim, self.n_classes))
        self.noise_cov = np.stack([np.identity(self.input_dim) for _ in range(self.n_classes)], axis = 0)
        
        self.labels = np.random.permutation(self.n_classes)
        
    def get_prompt(self):
        prompt = np.zeros(shape = (self.input_dim + 1, self.prompt_length))
        for t in range(self.prompt_length):
            cur_class = self.class_generator(self.n_classes)
                    
            cur_mean = self.class_means[:, cur_class].reshape(1, -1)
            cur_noise = np.random.multivariate_normal(self.noise_mean[:, cur_class], self.noise_cov[cur_class])
            
            cur_input = cur_mean + cur_noise
            cur_label = self.labels[cur_class].reshape(1, -1)

            prompt[:, t] = np.concatenate([cur_input, cur_label], axis = 1)
        
        return torch.from_numpy(prompt)

