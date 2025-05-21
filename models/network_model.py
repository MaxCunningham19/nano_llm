from typing import List
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .AbstractModel import AbstractModel
from .Loader import Loader
from .constants import SPECIAL_CHAR


class Layer(ABC):
    @abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def params(self) -> List[torch.Tensor]:
        pass


class LinearLayer(Layer):
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        self.W = torch.randn(fan_in, fan_out)
        self.b = torch.randn(fan_out) if bias else None

    def __call__(self, X: torch.Tensor):
        O = X @ self.W
        if self.b is not None:
            O += self.b
        return O

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class EmbeddingLayer(Layer):
    def __init__(self, input_dim, embedding_dims):
        self.embedding = torch.randn(input_dim, embedding_dims)

    def __call__(self, X: torch.Tensor):
        return self.embedding[X]

    def params(self):
        return [self.embedding]


class LeReLU(Layer):
    def __init__(self, alpha: int = 0.1):
        self.alpha = alpha

    def __call__(self, X: torch.Tensor):
        return torch.where(X > 0, X, X * self.alpha)

    def params(self):
        return []


class Network:
    def __init__(self, input_dim: int, embedding_dims: int, hidden_lyrs: int):
        self.layers: List[Layer] = [EmbeddingLayer(input_dim, embedding_dims)]
        for _ in range(hidden_lyrs):
            self.layers.append(LinearLayer(embedding_dims, embedding_dims, bias=True)),
            self.layers.append(LeReLU())

        self.layers.append(LinearLayer(embedding_dims, input_dim, bias=True))

    def __call__(self, X: torch.Tensor):
        for layer in self.layers:
            X = layer(X)
        return X

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()


class SimpleNetwork(Loader, AbstractModel):
    def __init__(self, fill: int = 1, embedding_dims: int = 10, hidden_lyrs: int = 2):
        self.embedding_dims = embedding_dims
        self.hidden_lyrs = hidden_lyrs
        self.fill = fill

    def load_data(self, data_path: str):
        data = self.load_from_file(data_path)
        unique_chars = set("".join(data))
        unique_chars.add(SPECIAL_CHAR)
        self.unique_chars = sorted(unique_chars)
        self.s_to_i = {s: i for i, s in enumerate(unique_chars)}
        self.i_to_s = {i: s for i, s in enumerate(unique_chars)}
        self.network = Network(len(self.unique_chars), self.embedding_dims, self.hidden_lyrs)

        self.XX = []
        self.YY = []
        for word in data:
            data = "".join([SPECIAL_CHAR] * self.fill)
            for char in word + SPECIAL_CHAR:
                self.XX.append(data[-self.fill :])
                self.YY.append(char)
                data += char

        print(self.XX[:10])
        print(self.YY[:10])

    def train(self, itterations: int = 100, batch_size: int = -1, **kwargs):
        return

    def visualize(self, path: str):
        return

    def predict(self, data: str):
        return SPECIAL_CHAR
