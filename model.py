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

class AttnDecoder(nn.Module):
    def __init__(self, max_length):
        super(AttnDecoder, self).__init__()
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
        # attention layer (we use the defition in the paper hence tanh)
        self.attn =  nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.max_length),
            nn.Tanh(dim=2)
            )
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=voc_size
        )
        self.softmax = nn.LogSoftmax(dim=2)

    # BEWARE NEED to Change the training to get encoder outputs
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embeddings(input)

        # compute the attention at the moment
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # combine them
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        
    
        output, (hidden, cell) = self.lstm(output, hidden)
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
