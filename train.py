
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.dataset import SeqGenerator, _convert_dict

class IterDataset(torch.utils.data.IterableDataset):

    def __init__(self, generator):
        super(IterDataset, self).__init__()
        self.generator = generator

    def __iter__(self):
        return iter(self.generator())

class Trainer:

    def __init__(self, model, data_generator, loss_fn=nn.CrossEntropyLoss, optimizer=optim.Adam, num_classes=1623, batch_size=16):

        # Instance of model
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size

        # Data generator here refers to something like SeqGenerator().get_random_seq
        self.data_generator = data_generator
        self.train_dataset = IterDataset(self.data_generator)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

        # Training loop parameters
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, lr=1e-5, eval_after=5):

        optim = self.optimizer(self.model.parameters(), lr=lr)
        criterion = self.loss_fn()

        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        for i, batch in enumerate(self.train_loader):

            batch = _convert_dict(batch)
            x, labels = batch["example"].to(self.device), batch["label"].to(self.device)
            optim.zero_grad()

            preds = self.model(x)

            target = nn.functional.one_hot(labels, self.num_classes)
            losses_all = criterion(preds, target)

            # Compute query mask on loss to only retain loss for the query entry (last column)
            query_mask = torch.full_like(losses_all, False)
            query_mask[:, -1] = True

            losses_weighted = torch.matmul(losses_all, query_mask)
            loss = torch.sum(losses_weighted) / torch.sum(query_mask)
            loss.backward()

            optim.step()
            
            total_loss += loss.item()
            if i % eval_after == 0:
                avg_loss = running_loss / eval_after
                print(f"Global batch {i}, avg loss after {eval_after} batches:", avg_loss)
                running_loss = 0
