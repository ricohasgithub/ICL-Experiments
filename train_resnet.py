import wandb

import torch
import torch.nn as nn
import torch.optim as optim

# import matplotlib.pyplot as plt

from datasets.dataset import SeqGenerator, _convert_dict

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="icl-resnet"
)


class IterDataset(torch.utils.data.IterableDataset):

    def __init__(self, generator):
        super(IterDataset, self).__init__()
        self.generator = generator

    def __iter__(self):
        return iter(self.generator())


class Trainer:

    def __init__(
        self,
        model,
        data_generator,
        data_generator_factory,
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        scheduler=optim.lr_scheduler.LambdaLR,
        num_classes=1623,
        batch_size=32,
    ):

        # Instance of model
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size

        # Data generator here refers to something like SeqGenerator().get_random_seq
        self.data_generator = data_generator
        self.data_generator_factory = data_generator_factory
        self.train_dataset = IterDataset(self.data_generator)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size
        )

        # Training loop parameters
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            # else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def _linear_warmup_and_sqrt_decay(self, step, warmup_steps=200, lr_max=5e-6):
        if step < warmup_steps:
            return lr_max * step / warmup_steps
        else:
            return lr_max * (warmup_steps ** 0.5) / (step ** 0.5)
        

    def train(self, lr=1e-2, eval_after=100):

        def _apply_masks(values):
            # query_mask = torch.full_like(losses_all, False)
            # query_mask[:, -1] = True

            # print("Value * mask: ", torch.sum(query_mask * values))
            # print("Mask: ", torch.sum(query_mask))

            values_query = torch.sum(values)
            return values_query

        optim = self.optimizer(self.model.parameters(), lr=lr)
        scheduler = self.scheduler(optim, lr_lambda=self._linear_warmup_and_sqrt_decay)

        criterion = self.loss_fn()

        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        running_accuracy = 0

        for i, batch in enumerate(self.train_loader):

            batch = _convert_dict(batch)
            examples, labels, target = (
                batch["examples"].to(self.device),
                batch["labels"].to(self.device),
                batch["target"].to(self.device),
            )
            optim.zero_grad()

            # for i in range(9):
            #     print(f'labels: {labels[0][i]}')
            #     print(examples[0][i].shape)
            #     plt.imshow(examples[0][i])
            #     plt.show()
                
            
            #     input()


            print(f'examples: {examples.shape}, labels: {labels.shape}, target: {target.shape}')

            preds = self.model(torch.permute(examples, (0, 1, 4, 2, 3)))

            # print(target[0])
            target = target[:, 0::2]
            # print(target[0])
            target_one_hot = (
                nn.functional.one_hot(target.to(torch.int64), self.num_classes)
                .to(torch.float32)
            )
            # print(f'predicted labels: {torch.argmax(preds, axis=2)[0]}')
            # print(f'target labels: {torch.argmax(target_one_hot, axis=2)[0]}')
            # print(f'preds: {preds.shape}, target_one_hot: {target_one_hot.shape}')
            # print(f'preds: {preds[0][0]}, target_one_hot: {target_one_hot[0][0]}')
            # print(f'preds: {preds.shape}, target_one_hot: {target_one_hot.shape}')
            # print(f'input 1: {preds.view(-1, self.num_classes).shape}, input 2: {target_one_hot.view(-1, self.num_classes).shape}')
            # criterion = nn.CrossEntropyLoss()
            # losses_all = criterion(preds.view(-1, self.num_classes), target_one_hot.view(-1, self.num_classes))
            losses_all = criterion(preds.permute((0,2,1)), target_one_hot.permute((0,2,1)))
            # print(f'losses_all: {losses_all.shape}')
            loss = losses_all
            # print(f'loss: {loss}')
            # # Compute query mask on loss to only retain loss for the query entry (last column)
            # query_mask = torch.full_like(losses_all, False)
            # # print(preds.shape, target_one_hot.shape, losses_all.shape, query_mask.shape)
            # query_mask[:, -1] = True

            # losses_weighted = losses_all * query_mask
            # # print(losses_weighted.shape, target_one_hot.shape, preds.shape)
            # loss = torch.sum(losses_weighted) / torch.sum(query_mask)
            loss.backward()

            optim.step()
            scheduler.step()

            running_loss += loss.item()

            # Compute accuracy
            # with torch.no_grad():

            predicted_labels = torch.argmax(preds, axis=2)

            correct = predicted_labels == target

            correct = correct.to(torch.float32)

            accuracy_query = _apply_masks(correct) / 9

            running_accuracy += accuracy_query.item()



            if i % eval_after == 0:
                avg_loss = running_loss / eval_after
                avg_accuracy = running_accuracy / eval_after

                print(f'avg_accuracy: {avg_accuracy}')

                wandb.log(
                    {
                        "global_step": i,
                        "loss": avg_loss,
                        "acc": avg_accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                if i % (eval_after * 20) == 0:
                    print(
                        f"Global batch {i}, avg loss after {eval_after} batches:",
                        round(avg_loss, 3),
                        f" | Global batch {i}, avg accuracy after {eval_after} batches:",
                        round(avg_accuracy * 100, 2),
                    )
                else:
                    print(
                        f"Global batch {i}, avg loss after {eval_after} batches:",
                        round(avg_loss, 3),
                        f" | avg accuracy (total, from_support) after {eval_after} batches:",
                        round(avg_accuracy * 100, 2),
                        end="\r",
                    )

                running_loss = 0
                running_accuracy = 0


    def get_eval_seq(self, seq_type):

        # seq_type = "icl" or "iwl"
        # Corresponds to fewshot_holdout and no_support_zipfian

        if seq_type == "icl":
            seq_generator = lambda x: self.data_generator_factory.get_fewshot_seq(
                "holdout",
                4,  # fs_shots
                2,  # ways
                "unfixed",
                False,  # randomly_generate_rare
                False,  # grouped
            )

        if seq_type == "iwl":
            pass
