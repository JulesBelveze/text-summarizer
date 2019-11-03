import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from vars import *


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, input):
        embedded = self.embeddings(input)
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=voc_size
        )
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        embedded = self.embeddings(input)
        output, (hidden, cell) = self.lstm(embedded, hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output, (hidden, cell)


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target):
        encoder_output, encoder_hidden = self.encoder(input)
        decoder_input = torch.tensor([2] * batch_size,
                                     dtype=torch.long, device=device).view(batch_size, -1)

        decoder_hidden = encoder_hidden

        batch_output = torch.Tensor()
        for i in range(MAX_LEN_HIGHLIGHT):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            batch_output = torch.cat((batch_output, decoder_output), 1)

        return batch_output
