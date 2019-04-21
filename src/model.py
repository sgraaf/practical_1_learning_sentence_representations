import torch
import torch.nn as nn

from utils import get_features

class InferSent(nn.Module):
    """
    Sentence embedding model that is trained on natural language inference data
    """

    def __init__(self, input_dim, hidden_dim, output_dim, embedding, encoder):
        """
        Initialize the InferSent model

        :param int input_dim: the dimensionality of the input
        :param int hidden_dim: the dimensionality of the hidden states
        :param int output_dim: the dimensionality of the output
        :param Encoder encoder: the encoder to use
        """
        super(InferSent).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = encoder

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, premise_batch, hypothesis_batch):
        # get the sentences and their lengths for the premises
        premise_sentences = premise_batch[0]
        premise_lens = premise_batch[1]

        # get the sentences and their lengths for the hypotheses
        hypothesis_sentences = hypothesis_batch[0]
        hypothesis_lens = hypothesis_batch[1]

        # pass them through the encoder
        u_batch = self.encoder.forward(premise_sentences, premise_lens)
        v_batch = self.encoder.forward(hypothesis_sentences, hypothesis_lens)

        # get the features from these encoded sentences
        features = get_features(u_batch, v_batch)
        
        # pass the features through the model
        out = self.model(features)

        return out






