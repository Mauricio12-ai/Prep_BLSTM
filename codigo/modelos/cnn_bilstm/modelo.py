"""Dos ramas temporales: RGB (características ResNet18) y landmarks XY/máscaras."""
import torch
from torch import nn


class Modelo(nn.Module):
    def __init__(self, modo='fusion', hidden=64, dropout=0.3, clases=34):
        super().__init__()
        if modo not in ('fusion','rgb','landmarks'):
            raise ValueError('Modo inválido')
        self.modo = modo
        if modo != 'landmarks':
            self.rgb = nn.LSTM(512, hidden, batch_first=True, bidirectional=True)
        if modo != 'rgb':
            self.landmarks = nn.LSTM(300, hidden, batch_first=True, bidirectional=True)
        width = hidden * (4 if modo == 'fusion' else 2)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, clases))

    def forward(self, rgb_features, landmark_features):
        vectors = []
        if self.modo != 'landmarks':
            _, (h, _) = self.rgb(rgb_features)
            vectors.append(torch.cat((h[-2],h[-1]), dim=1))
        if self.modo != 'rgb':
            _, (h, _) = self.landmarks(landmark_features)
            vectors.append(torch.cat((h[-2],h[-1]), dim=1))
        return self.classifier(torch.cat(vectors,dim=1))
