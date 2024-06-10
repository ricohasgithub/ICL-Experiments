import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.dataset import SeqGenerator, _convert_dict

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="icl-omniglot"
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
        num_classes=1623,
        batch_size=16,
    ):

        # Instance of model
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size

        # Data generator here refers to something like SeqGenerator().get_random_seq
        self.data_generator = data_generator
        self.icl_seq_generator = lambda : self.data_generator_factory.get_fewshot_seq(
            "holdout",
            4,  # fs_shots
            2,  # ways
            "unfixed",
            False,  # randomly_generate_rare
            False,  # grouped
        )
        self.iwl_seq_generator = (
            lambda : self.data_generator_factory.get_no_support_seq(
                "zipfian",
                9,  # seq_len
                False,  # all_unique
                "ordered",  # labeling_common
                False,  # randomly_generate_rare
            )
        )

        self.data_generator_factory = data_generator_factory
        self.train_dataset = IterDataset(self.data_generator)
        self.icl_eval_dataset = IterDataset(self.icl_seq_generator)
        self.iwl_eval_dataset = IterDataset(self.iwl_seq_generator)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size
        )

        self.icl_eval_loader = torch.utils.data.DataLoader(
            self.icl_eval_dataset, batch_size=self.batch_size
        )
        self.iwl_eval_loader = torch.utils.data.DataLoader(
            self.iwl_eval_dataset, batch_size=self.batch_size
        )

        # Training loop parameters
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            # else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def train(self, lr=1e-5, eval_after=100):

        optim = self.optimizer(self.model.parameters(), lr=lr)
        criterion = self.loss_fn(reduction="none")

        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        running_accuracy = 0
        running_common_accuracy = 0
        running_rare_accuracy = 0
        running_fewshot_accuracy = 0

        running_support_accuracy = 0
        running_support_common_accuracy = 0
        running_support_rare_accuracy = 0
        running_support_fewshot_accuracy = 0

        for i, batch in enumerate(self.train_loader):

            self.model.train()

            batch = _convert_dict(batch)
            examples, labels, target = (
                batch["examples"].to(self.device),
                batch["labels"].to(self.device),
                batch["target"].to(self.device),
            )
            optim.zero_grad()

            preds = self.model(examples, labels).transpose(1, 2)

            target_one_hot = (
                nn.functional.one_hot(target.to(torch.int64), self.num_classes)
                .transpose(1, 2)
                .to(torch.float32)
            )
            losses_all = criterion(preds, target_one_hot)

            def _apply_masks(values, losses_all):
                query_mask = torch.full_like(losses_all, False)
                query_mask[:, -1] = True

                values_query = torch.sum(query_mask * values) / torch.sum(query_mask)
                return values_query

            # Compute query mask on loss to only retain loss for the query entry (last column)
            query_mask = torch.full_like(losses_all, False)
            # print(preds.shape, target_one_hot.shape, losses_all.shape, query_mask.shape)
            query_mask[:, -1] = True

            losses_weighted = losses_all * query_mask
            # print(losses_weighted.shape, target_one_hot.shape, preds.shape)
            loss = torch.sum(losses_weighted) / torch.sum(query_mask)
            loss.backward()

            optim.step()

            running_loss += loss.item()

            # Compute accuracy
            with torch.no_grad():

                self.model.eval()

                predicted_labels = torch.argmax(preds, axis=1)

                correct = predicted_labels == target

                correct = correct.to(torch.float32)

                accuracy_query = _apply_masks(correct, losses_all)

                running_accuracy += accuracy_query.item()

                n_rare_classes = self.data_generator_factory.n_rare_classes
                n_holdout_classes = self.data_generator_factory.n_holdout_classes
                n_classes = self.data_generator_factory.n_classes

                # For labeling_common = "ordered" and labeling_rare = "ordered"
                common_start_idx = n_rare_classes
                common_labels = range(common_start_idx, n_classes - n_holdout_classes)
                rare_labels = range(n_rare_classes)

                # Compute whether query predictions were from common or rare classes.
                from_common_all = torch.isin(
                    predicted_labels, torch.tensor(common_labels).to(self.device)
                )
                from_rare_all = torch.isin(
                    predicted_labels, torch.tensor(rare_labels).to(self.device)
                )
                from_common = _apply_masks(
                    from_common_all, losses_all
                )  # average for query only
                from_rare = _apply_masks(from_rare_all, losses_all)

                running_common_accuracy += accuracy_query.item() * from_common.item()
                running_rare_accuracy += accuracy_query.item() * from_rare.item()

                # Compute whether query predictions were from the fewshot classes.
                fewshot_ways = 2
                from_fewshot_all = torch.isin(
                    predicted_labels, torch.arange(fewshot_ways).to(self.device)
                )
                from_fewshot = _apply_masks(from_fewshot_all, losses_all)  # for query only

                running_fewshot_accuracy += accuracy_query.item() * from_fewshot.item()

                # Compute whether query predictions were from common or rare classes.
                support_labels = target[:, :-2:2]
                batch_size, seq_len = predicted_labels.shape
                support_len = support_labels.shape[1]
                predicted_labels_reshaped = predicted_labels.reshape(
                    batch_size, seq_len, 1
                )
                support_labels_reshaped = support_labels.reshape(
                    batch_size, 1, support_len
                )
                from_support_all = predicted_labels_reshaped == support_labels_reshaped
                from_support_all = (
                    from_support_all.sum(-1).type(torch.bool).to(self.device)
                )
                from_support = _apply_masks(
                    from_support_all, losses_all
                )  # avg for query only
                from_support_common = _apply_masks(
                    from_support_all * from_common_all, losses_all
                )
                from_support_rare = _apply_masks(
                    from_support_all * from_rare_all, losses_all
                )
                from_support_fewshot = _apply_masks(
                    from_support_all * from_fewshot_all, losses_all
                )

                running_support_accuracy += accuracy_query.item() * from_support.item()
                running_support_common_accuracy += (
                    accuracy_query.item() * from_support_common.item()
                )
                running_support_rare_accuracy += (
                    accuracy_query.item() * from_support_rare.item()
                )
                running_support_fewshot_accuracy += (
                    accuracy_query.item() * from_support_fewshot.item()
                )

            if i % eval_after == 0:

                self.model.eval()

                with torch.no_grad():
                    icl_batch = _convert_dict(next(iter(self.icl_eval_loader)))
                    iwl_batch = _convert_dict(next(iter(self.iwl_eval_loader)))

                    icl_examples, icl_labels, icl_target = (
                        icl_batch["examples"].to(self.device),
                        icl_batch["labels"].to(self.device),
                        icl_batch["target"].to(self.device),
                    )

                    iwl_examples, iwl_labels, iwl_target = (
                        iwl_batch["examples"].to(self.device),
                        iwl_batch["labels"].to(self.device),
                        iwl_batch["target"].to(self.device),
                    )

                    icl_preds = self.model(icl_examples, icl_labels).transpose(1, 2)
                    iwl_preds = self.model(iwl_examples, iwl_labels).transpose(1, 2)

                    iclAccDict = self.compute_accuracy(
                        icl_preds, icl_target, query_mask
                    )
                    iwlAccDict = self.compute_accuracy(
                        iwl_preds, iwl_target, query_mask
                    )

                    avg_loss = running_loss / eval_after
                    avg_accuracy = running_accuracy / eval_after
                    avg_common_accuracy = running_common_accuracy / eval_after
                    avg_rare_accuracy = running_rare_accuracy / eval_after
                    avg_fewshot_accuracy = running_fewshot_accuracy / eval_after
                    avg_support_accuracy = running_support_accuracy / eval_after
                    avg_support_common_accuracy = (
                        running_support_common_accuracy / eval_after
                    )
                    avg_support_rare_accuracy = (
                        running_support_rare_accuracy / eval_after
                    )
                    avg_support_fewshot_accuracy = (
                        running_support_fewshot_accuracy / eval_after
                    )

                    wandb.log(
                        {
                            "global_step": i,
                            "loss": avg_loss,
                            "train_acc": avg_accuracy,
                            "train_common_acc": avg_common_accuracy,
                            "train_rare_acc": avg_rare_accuracy,
                            "train_fewshot_acc": avg_fewshot_accuracy,
                            "train_support_acc": avg_support_accuracy,
                            "train_support_common_acc": avg_support_common_accuracy,
                            "train_support_rare_acc": avg_support_rare_accuracy,
                            "train_support_fewshot_acc": avg_support_fewshot_accuracy,
                            "icl_eval_acc": iclAccDict["acc"],
                            "icl_eval_common_acc": iclAccDict["common_acc"],
                            "icl_eval_rare_acc": iclAccDict["rare_acc"],
                            "icl_eval_fewshot_acc": iclAccDict["fewshot_acc"],
                            "icl_eval_support_acc": iclAccDict["support_acc"],
                            "icl_eval_support_common_acc": iclAccDict[
                                "support_common_acc"
                            ],
                            "icl_eval_support_rare_acc": iclAccDict["support_rare_acc"],
                            "icl_eval_support_fewshot_acc": iclAccDict[
                                "support_fewshot_acc"
                            ],
                            "iwl_eval_acc": iwlAccDict["acc"],
                            "iwl_eval_common_acc": iwlAccDict["common_acc"],
                            "iwl_eval_rare_acc": iwlAccDict["rare_acc"],
                            "iwl_eval_fewshot_acc": iwlAccDict["fewshot_acc"],
                            "iwl_eval_support_acc": iwlAccDict["support_acc"],
                            "iwl_eval_support_common_acc": iwlAccDict[
                                "support_common_acc"
                            ],
                            "iwl_eval_support_rare_acc": iwlAccDict["support_rare_acc"],
                            "iwl_eval_support_fewshot_acc": iwlAccDict[
                                "support_fewshot_acc"
                            ],
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
                            round(avg_support_accuracy * 100, 2),
                            end="\r",
                        )

                    running_loss = 0
                    running_accuracy = 0
                    running_common_accuracy = 0
                    running_rare_accuracy = 0
                    running_fewshot_accuracy = 0

                    running_support_accuracy = 0
                    running_support_common_accuracy = 0
                    running_support_rare_accuracy = 0
                    running_support_fewshot_accuracy = 0

    def compute_accuracy(self, logits, target, query_mask):

        def apply_query_mask(values):
            values_query = torch.sum(query_mask * values) / torch.sum(query_mask)
            return values_query

        predicted_labels = torch.argmax(logits, axis=1)

        correct = predicted_labels == target
        correct = correct.to(torch.float32)

        accuracy_query = apply_query_mask(correct)

        n_rare_classes = self.data_generator_factory.n_rare_classes
        n_holdout_classes = self.data_generator_factory.n_holdout_classes
        n_classes = self.data_generator_factory.n_classes

        # For labeling_common = "ordered" and labeling_rare = "ordered"
        common_start_idx = n_rare_classes
        common_labels = range(common_start_idx, n_classes - n_holdout_classes)
        rare_labels = range(n_rare_classes)

        # Compute whether query predictions were from common or rare classes.
        from_common_all = torch.isin(
            predicted_labels, torch.tensor(common_labels).to(self.device)
        )
        from_rare_all = torch.isin(
            predicted_labels, torch.tensor(rare_labels).to(self.device)
        )
        from_common = apply_query_mask(from_common_all)  # average for query only
        from_rare = apply_query_mask(from_rare_all)

        # Compute whether query predictions were from the fewshot classes.
        fewshot_ways = 2
        from_fewshot_all = torch.isin(
            predicted_labels, torch.arange(fewshot_ways).to(self.device)
        )
        from_fewshot = apply_query_mask(from_fewshot_all)  # for query only

        # Compute whether query predictions were from common or rare classes.
        support_labels = target[:, :-2:2]
        batch_size, seq_len = predicted_labels.shape
        support_len = support_labels.shape[1]
        predicted_labels_reshaped = predicted_labels.reshape(batch_size, seq_len, 1)
        support_labels_reshaped = support_labels.reshape(batch_size, 1, support_len)
        from_support_all = predicted_labels_reshaped == support_labels_reshaped
        from_support_all = from_support_all.sum(-1).type(torch.bool).to(self.device)
        from_support = apply_query_mask(from_support_all)  # avg for query only
        from_support_common = apply_query_mask(from_support_all * from_common_all)
        from_support_rare = apply_query_mask(from_support_all * from_rare_all)
        from_support_fewshot = apply_query_mask(from_support_all * from_fewshot_all)

        return {
            "acc": accuracy_query.item(),
            "common_acc": from_common.item() * accuracy_query.item(),
            "rare_acc": from_rare.item() * accuracy_query.item(),
            "fewshot_acc": from_fewshot.item() * accuracy_query.item(),
            "support_acc": from_support.item() * accuracy_query.item(),
            "support_common_acc": from_support_common.item() * accuracy_query.item(),
            "support_rare_acc": from_support_rare.item() * accuracy_query.item(),
            "support_fewshot_acc": from_support_fewshot.item() * accuracy_query.item(),
        }
