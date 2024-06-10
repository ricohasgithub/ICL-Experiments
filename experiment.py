import sys

from train import Trainer
from datasets.dataset import SeqGenerator

from models.transformer import Transformer
from models.embedding import InputEmbedder
from datasets.dataset import OmniglotDatasetForSampling
from datasets.dataset import GaussianVectorDatasetForSampling

import torch


def experiment_base(dataset, p_bursty):

    if dataset == "synthetic":
        input_embedding = InputEmbedder(linear_input_dim=64, example_encoding="linear")
        seq_generator_factory = SeqGenerator(
            dataset_for_sampling=GaussianVectorDatasetForSampling(),
            n_rare_classes=1603,  # 1623 - 20
            n_common_classes=10,
            n_holdout_classes=10,
            zipf_exponent=0,
            use_zipf_for_common_rare=False,
            noise_scale=0.0,
            preserve_ordering_every_n=None,
        )
    elif dataset == "omniglot":
        input_embedding = InputEmbedder(
            linear_input_dim=11025, example_encoding="resnet"
        )
        seq_generator_factory = SeqGenerator(
            dataset_for_sampling=OmniglotDatasetForSampling("train"),
            n_rare_classes=1603,  # 1623 - 20
            n_common_classes=10,
            n_holdout_classes=10,
            zipf_exponent=0,
            use_zipf_for_common_rare=False,
            noise_scale=0.0,
            preserve_ordering_every_n=None,
        )

    model = Transformer(input_embedder=input_embedding)

    data_generator = lambda: seq_generator_factory.get_bursty_seq(
        seq_len=9,
        shots=3,
        ways=2,
        p_bursty=p_bursty,
        p_bursty_common=0,
        p_bursty_zipfian=1,
        non_bursty_type="zipfian",
        labeling_common="ordered",
        labeling_rare="ordered",
        randomly_generate_rare=False,
        grouped=False,
    )

    trainer = Trainer(model, data_generator, seq_generator_factory)
    torch.autograd.set_detect_anomaly(True)
    trainer.train()


if __name__ == "__main__":

    # Read command line inputs to decide which experiment to run
    experiment_id = sys.argv[1]
    dataset = sys.argv[2]
    p_bursty = float(sys.argv[3])

    if experiment_id == "base":
        experiment_base(dataset)
    elif experiment_id == "mixed":
        pass
