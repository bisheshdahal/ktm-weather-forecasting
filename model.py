"""
model.py — contains LSTM class

Use for any single-number regression task (time-series forecasting: weather, stock price, sales, sensor data, AQI).
Change input_size for different features.

Not for: classification, multi-step forecasting, sequence-to-sequence, raw text/NLP.
"""

import torch
from torch import nn


class LSTM(nn.Module):
    """
    A general-purpose LSTM regression model.

    Reads a sequence step by step (like reading a sentence word by word),
    builds up memory, and makes ONE prediction at the end.

    Input shape:  (batch_size, seq_length, input_size)
    Output shape: (batch_size, output_size)

    """

    def __init__(self, input_size, hidden_size, num_stacked_layers, output_size=1):
        """
        input_size : number of FEATURE COLUMNS per time-step.

        hidden_size: how much "memory" each LSTM cell carries. (32, 64, 128 are common)

        num_stacked_layers:  how many LSTM layers stacked on top of each other. 
                                
        output_size:         how many numbers you want predicted per sequence.
                                 Put >1 if predicting multiple values at once
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_stacked_layers,
            batch_first=True,
        )
        # output = (weights × hidden size) + bias
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Starting memory tensors
        # Short-term Memory
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        #Long-term Memory
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)

        # runs through entire lstm to get output whuch gets put in h0,c0
        out, _ = self.lstm(x, (h0, c0))

        #one output
        out = self.fc(out[:, -1, :])
        return out