import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from resnet import CustomResNet


class BatchApply(nn.Module):
    def __init__(self, function):
        super(BatchApply, self).__init__()
        self.function = function

    def forward(self, x):
        # Apply the function to each item in the batch independently
        batch_size = x.size(0)
        return torch.stack([self.function(x[i]) for i in range(batch_size)])


def _create_positional_encodings(inputs, max_time=30.0):
    """Generates positional encodings for the input.
    Args:
        inputs: A tensor of shape [batch_size, seq_len, emb_size].
        max_time: (default 10000) Constant used to scale position by in the
        encodings.

    Returns:
        pos_emb: as defined above, of size [1, seq_len, emb_size].
    """

    _, seq_len, embedding_size = inputs.shape

    if embedding_size % 2 == 1:
        raise ValueError("Embedding sizes must be even if using positional encodings.")

    # Generate a sequence of positions and frequencies.
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.arange(0, embedding_size, 2, dtype=torch.float32)
    inverse_freqs = 1.0 / (max_time ** (freqs / embedding_size))

    # We combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
    pos_emb = torch.einsum("i,j->ij", pos, inverse_freqs)

    # Concat sines and cosines and return.
    pos_emb = torch.cat([torch.sin(pos_emb), torch.cos(pos_emb)], -1)

    return pos_emb


class InputEmbedder(nn.Module):
    """Input embedder."""

    def __init__(
        self,
        linear_input_dim,
        n_classes=1623,
        emb_dim=64,
        seq_shape=11025,
        example_encoding="linear",
        flatten_superpixels=False,
        example_dropout_prob=0.0,
        concatenate_labels=False,
        use_positional_encodings=True,
        positional_dropout_prob=0.1,
        name=None,
    ):
        """Initialize the input embedder.

        Args:
          n_classes: Total nber of output classes.
          emb_dim: Dimensionality of example and label embeddings.
          example_encoding: How to encode example inputs.
            'resnet': simple resnet encoding
            'linear': flatten and pass through a linear layer
            'embedding': pass through an embedding layer
          flatten_superpixels: Whether to flatten the output of the resnet (instead
            of taking a mean over superpixels).
          example_dropout_prob: Dropout probability on example embeddings. Note that
            these are applied at both train and test.
          concatenate_labels: Whether to concatenate example and label embeddings
            into one token for each (example, label) pair, rather than being fed to
            the transformer as two separate tokens.
          use_positional_encodings: Whether to use positional encoding.
          positional_dropout_prob: Positional dropout probability.
          name: Optional name for the module.
        """
        super(InputEmbedder, self).__init__()

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._n_classes = n_classes
        self._emb_dim = emb_dim
        self._example_encoding = example_encoding
        self._flatten_superpixels = flatten_superpixels
        self._example_dropout_prob = example_dropout_prob
        self._concatenate_labels = concatenate_labels
        self._use_positional_encodings = use_positional_encodings
        self._positional_dropout_prob = positional_dropout_prob

        # self._linear_input_dim = examples.shape[2] * examples.shape[3] * examples.shape[4]
        self._linear_input_dim = linear_input_dim

        self.linear = nn.Linear(self._linear_input_dim, self._emb_dim).to(self.device)
        self.embedding_layer = nn.Embedding(self._n_classes, self._emb_dim).to(
            self.device
        )
        self.resnet = BatchApply(CustomResNet(
            (2, 2, 2, 2), (16, 32, 32, self._emb_dim), flatten_superpixels=False
        ).to(self.device))

        self.example_dropout_layer = nn.Dropout(self._example_dropout_prob)
        self.positional_dropout_layer = nn.Dropout(self._positional_dropout_prob)

    def forward(self, examples, labels, is_training=True):
        """Call to the input embedder.

        Args:
          examples: input sequence of shape
            [batch_size, seq_len, height, width, channels]
          labels: input sequence of shape [batch_size, seq_len]
          is_training: if is currently training.

        Returns:
          outputs: output of the transformer tower
            of shape [batch_size, seq_len, channels].
        """
        # Encode the example inputs into shape (B, SS, E)
        # if self._example_encoding == 'resnet':
        #   if self._flatten_superpixels:
        #     resnet_emb_dim = int(self._emb_dim / 16)
        #   else:
        #     resnet_emb_dim = self._emb_dim
        #   example_encoding = resnet.SimpleResNet(
        #       blocks_per_group=(2, 2, 2, 2),
        #       channels_per_group=(16, 32, 32, resnet_emb_dim),
        #       flatten_superpixels=self._flatten_superpixels,
        #   )
        #   example_encoding_with_is_training = partial(example_encoding, is_training=is_training)
        #   batch_apply = BatchApply(example_encoding_with_is_training)
        #   h_example = batch_apply(examples)
        if self._example_encoding == "linear":
            h_example = examples.flatten(start_dim=2)
            h_example = self.linear(h_example)
        elif self._example_encoding == "embedding":
            h_example = self.embedding_layer(examples)
        elif self._example_encoding == "resnet":
            (B, SS, H, W, C) = examples.shape
            input_tensor_reshaped = examples.reshape(-1, C, H, W)
            h_example = self.resnet(input_tensor_reshaped)
            h_example = h_example.reshape(B, SS, self._emb_dim)
        else:
            raise ValueError("Invalid example_encoding: %s" % self._example_encoding)

        # Add dropout to example embeddings.
        # Note that this is not restricted to training, because the purpose is to
        # add noise to the examples, not for regularization.
        if self._example_dropout_prob:
            h_example = self.example_dropout_layer(h_example)

        # Embed the labels.
        n_emb_classes = self._n_classes
        labels_to_embed = labels
        if self._concatenate_labels:
            # Dummy label for final position, where we don't want the label
            # information to be available.
            n_emb_classes += 1
            labels_to_embed[:, -1] = n_emb_classes - 1

        embs = torch.nn.init.normal_(
            torch.empty(n_emb_classes, self._emb_dim), std=0.02
        ).to(self.device)
        # embs = hk.get_parameter(
        #     'embs', [n_emb_classes, self._emb_dim],
        #     init=init.TruncatedNormal(stddev=0.02))
        h_label = embs[labels_to_embed]  # (B, SS, E)

        if self._concatenate_labels:
            # Concatenate example and label embeddings
            hh = torch.cat((h_example, h_label), axis=2)  # (B,SS,E*2)
        else:
            # Interleave example and label embeddings
            hh = torch.empty(
                (h_example.shape[0], h_example.shape[1] * 2 - 1, h_example.shape[2]),
                dtype=h_example.dtype,
            )
            hh[:, 0::2] = h_example
            hh[:, 1::2] = h_label[:, :-1]
            # hh is (B,S,E) where S=SS*2-1

        # Create positional encodings.
        if self._use_positional_encodings:
            positional_encodings = _create_positional_encodings(hh)
            if is_training:
                positional_encodings = self.positional_dropout_layer(
                    positional_encodings
                )
            # Add on the positional encoding.
            hh = hh + positional_encodings

        return hh


if __name__ == "__main__":
    examples = torch.randn(2, 3, 105, 105, 3)
    emb = InputEmbedder(11025, n_classes=1623, example_encoding='resnet')
    labels = torch.randint(0, 1623, (2, 3))
    out = emb(examples, labels)
    print(out.shape)
