import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from vars import *


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.num_layer = 1
        self.embeddings = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=hidden_dim
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def init_hidden(self):
        return (torch.zeros(2 * self.num_layer, batch_size, hidden_dim, dtype=torch.float, device=device),
                torch.zeros(2 * self.num_layer, batch_size, hidden_dim, dtype=torch.float, device=device))

    def forward(self, input, state):
        embedded = self.embeddings(input)
        output, (hidden, cell) = self.lstm(embedded, state)
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=hidden_dim
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim * 2,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=voc_size
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        embedded = self.embeddings(input).unsqueeze(1)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(output.squeeze(1))
        output = self.logsoftmax(output)
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
            hidden_size=hidden_dim * 2,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        # attention layer (we use the definition in the paper hence tanh)
        self.W = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, 1, bias=False)
        )
        self.fc = nn.Linear(
            in_features=hidden_dim * 4,
            out_features=voc_size
        )
        self.W_gen_sig = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim * 2 + hidden_dim, 1),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, decoder_hidden, cell, encoder_outputs, story_extended, extra_zeros):
        (b, t_k, n) = encoder_outputs.shape
        embedded = self.embeddings(input).unsqueeze(1)  # batch * hidden_dim

        _, (decoder_hidden, cell) = self.lstm(embedded, (decoder_hidden, cell))

        decoder_hidden_expanded = decoder_hidden.permute(1, 0, 2)  # batch_size * hidden_dim
        decoder_hidden_expanded = decoder_hidden_expanded.expand(b, t_k,
                                                                 n).contiguous()  # batch_size * seq_len * hidden_dim

        att_features = torch.cat((decoder_hidden_expanded, encoder_outputs), 2)  # batch_size * seq_lens * 2*hidden_dim
        e_t = self.W(att_features).squeeze(2)  # batch_size * seq_lens
        a_t = self.softmax(e_t)  # batch_size * seq_lens

        a_applied = torch.bmm(a_t.unsqueeze(1), encoder_outputs).squeeze(1)

        s_t_h_t = torch.cat((decoder_hidden.squeeze(0), a_applied), 1)  # batch_size * 2*hidden_dim
        p_vocab = self.softmax(self.fc(s_t_h_t))

        cat_gen = torch.cat((a_applied, decoder_hidden.squeeze(0), embedded.squeeze(1)), 1)

        p_gen = self.W_gen_sig(cat_gen).clamp(min=1e-8)
        p_vocab = p_vocab * p_gen
        a_t_p_gen = (1 - p_gen) * a_t

        p_vocab_cat = torch.cat((p_vocab, extra_zeros), 1)
        output = p_vocab_cat.scatter_add(1, story_extended, a_t_p_gen).clamp(min=1e-8)
        return torch.log(output), (decoder_hidden, cell)


class PointerGenerator(nn.Module):
    def __init__(self, encoder, decoder):
        super(PointerGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, story_extended, extra_zeros):
        null_state_encoder = self.encoder.init_hidden()
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(input, null_state_encoder)

        encoder_l, encoder_r = encoder_hidden[0], encoder_hidden[1]
        encoder_hidden = torch.cat((encoder_l, encoder_r), 1).unsqueeze(0)

        cell_l, cell_r = encoder_cell[0], encoder_cell[1]
        encoder_cell = torch.cat((cell_l, cell_r), 1).unsqueeze(0)

        # initializing the decoder with the <START> token
        decoder_input = torch.tensor([2] * batch_size,
                                     dtype=torch.long, device=device)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        batch_output = torch.Tensor().to(device)
        for i in range(MAX_LEN_HIGHLIGHT):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, decoder_hidden, decoder_cell,
                                                                          encoder_output, story_extended, extra_zeros)
            batch_output = torch.cat((batch_output, decoder_output.unsqueeze(1)), 1)

            # using teacher forcing
            if random.uniform(0, 1) > .5 and self.training:
                decoder_input = target[:, i]
            else:
                decoder_input = decoder_output.argmax(dim=1)

        return batch_output
