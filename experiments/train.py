
import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer import Transformer
from models.embedding import InputEmbedder

from datasets.dataset import SeqGenerator

class IterDataset(torch.utils.data.IterableDataset):

    def __init__(self, generator):
        super(IterDataset, self).__init__()
        self.generator = generator

    def __iter__(self):
        return iter(self.generator())

class Trainer:

    def __init__(self, model, data_generator, loss=nn.CrossEntropyLoss, optimizer=optim.Adam, batch_size=16):

        # Instance of model
        self.model = model
        self.batch_size = batch_size

        # Data generator here refers to something like SeqGenerator().get_random_seq
        self.data_generator = data_generator
        self.train_dataset = IterDataset(self.data_generator)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

        # Training loop parameters
        self.loss = loss
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, lr=1e-5, eval_after=5):

        optim = self.optimizer(self.model.parameters(), lr=lr)
        criterion = self.loss()

        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        for i, batch in enumerate(self.train_loader):

            x, labels = batch["example"].to(self.device), batch["target"].to(self.device)
            optim.zero_grad()

            preds = self.model(x)
            curr_loss = criterion(preds, labels)
            curr_loss.backward()

            optim.step()
            total_loss += curr_loss.item()
            if i % eval_after == 0:
                avg_loss = running_loss / eval_after
                print(f"Batch {i}, avg loss after {eval_after} batches:", avg_loss)
                running_loss = 0
