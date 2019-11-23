import torch
import torch.nn as nn
import torch.nn.functional as F
import random
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
    def __init__(self):
        super(AttnDecoder, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=hidden_dim
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        # attention layer (we use the definition in the paper hence tanh)
        self.W = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh()
        )
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(.1)
        self.fc = nn.Linear(
            in_features=hidden_dim*2,
            out_features=voc_size
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, decoder_hidden, cell, encoder_outputs):
        (b, t_k, n) = encoder_outputs.shape
        embedded = self.embeddings(input.view(batch_size, -1))  # batch * hidden_dim
        embedded = self.dropout(embedded)

        _, (decoder_hidden, cell) = self.lstm(embedded, (decoder_hidden, cell))

        decoder_hidden_expanded = decoder_hidden.permute(1,0,2) # batch_size * hidden_dim
        decoder_hidden_expanded = decoder_hidden_expanded.expand(b, t_k, n).contiguous() # batch_size * seq_len * hidden_dim

        att_features = torch.cat((decoder_hidden_expanded, encoder_outputs),2)# batch_size * seq_lens * 2*hidden_dim
        e_t = self.W(att_features).squeeze(2) # batch_size * seq_lens
        a_t = self.softmax(e_t) # batch_size * seq_lens

        a_applied = torch.bmm(a_t.unsqueeze(1), encoder_outputs).squeeze(1)
        s_t_h_t = torch.cat((decoder_hidden.squeeze(0), a_applied),1) # batch_size * 2*hidden_dim
        output = self.softmax(self.fc(s_t_h_t))
        return output, (decoder_hidden, cell)


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


class Seq2seqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target):
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(input)

        # initializing the decoder with the <START> token
        decoder_input = torch.tensor([2] * batch_size,
                                     dtype=torch.long, device=device)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        batch_output = torch.Tensor().to(device)
        for i in range(MAX_LEN_HIGHLIGHT):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell,
                                                                          encoder_output)

            batch_output = torch.cat((batch_output, decoder_output.unsqueeze(1)),1)

            # using teacher forcing
            if random.uniform(0, 1) > .5:
                decoder_input = target[:, i]
            else:
                decoder_input = decoder_output.argmax(dim=1)

        return batch_output
