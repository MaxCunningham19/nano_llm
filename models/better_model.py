from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True, W_scale: float = 1.0, b_scale: float = 1.0):
        self.W = torch.randn(fan_in, fan_out) * W_scale
        self.b = torch.randn(fan_out) * b_scale if bias else None

    def __call__(self, X: torch.Tensor):
        O = X @ self.W
        if self.b is not None:
            O += self.b
        return O

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class BatchNorm(Layer):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.01):
        self.momentum = momentum
        self.eps = eps
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, X: torch.Tensor):
        if self.training:
            mean = X.mean(0, keepdim=True)
            var = X.var(0, keepdim=True, unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            X_hat = (X - mean) / torch.sqrt(var + self.eps)
        else:
            X_hat = (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.gamma * X_hat + self.beta

    def params(self):
        return [self.gamma, self.beta]


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
    def __init__(self, input_dim: int, embedding_dims: int, hidden_lyrs: int, fill: int):
        self.fill = fill
        self.embedding_layer = EmbeddingLayer(input_dim, embedding_dims)
        self.layers: List[Layer] = []
        for _ in range(hidden_lyrs):
            self.layers.append(LinearLayer(embedding_dims * fill, embedding_dims * fill, bias=False, W_scale=0.1)),
            self.layers.append(BatchNorm(embedding_dims * fill))
            self.layers.append(LeReLU())

        self.layers.append(LinearLayer(embedding_dims * fill, input_dim, bias=True, W_scale=0.1, b_scale=0.001))

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        X = self.embedding_layer(X)
        X = X.view((X.shape[0], -1))
        for layer in self.layers:
            X = layer(X)
        return X

    def params(self) -> List[torch.Tensor]:
        params = self.embedding_layer.params()
        for layer in self.layers:
            params += layer.params()
        return params


class BatchedNetwork(Loader, AbstractModel):
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
        self.network = Network(len(self.unique_chars), self.embedding_dims, self.hidden_lyrs, self.fill)
        for p in self.network.params():
            p.requires_grad = True

        self.XX = []
        self.YY = []

        for word in data:
            context = [self.s_to_i[SPECIAL_CHAR]] * self.fill
            for char in word + SPECIAL_CHAR:
                self.XX.append(context[:])
                self.YY.append(self.s_to_i[char])
                context = context[1:] + [self.s_to_i[char]]

        self.XX = torch.tensor(self.XX, dtype=torch.long)
        self.YY = torch.tensor(self.YY, dtype=torch.long)

    def train(self, itterations: int = 100, batch_size: int = -1, lr: float = 0.01, loss_steps: int = 1, **kwargs):
        losses = []
        for k in range(itterations):
            ixs = self.sample(batch_size)
            X = self.XX[ixs]
            Y = self.YY[ixs]
            O: torch.Tensor = self.network(X)
            loss = F.cross_entropy(O, Y)
            if k % loss_steps == 0:
                losses.append(loss.item())
            loss.backward()
            for p in self.network.params():
                p.data += -lr * p.grad
                p.grad.zero_()
        return losses

    def sample(self, batch_size: int):
        if batch_size == -1:
            return torch.tensor(range(len(self.XX)))
        return torch.randint(0, len(self.XX), (batch_size,))

    def visualize(self, path: str):
        return

    def predict(self, data: str):
        if len(data) < self.fill:
            data = "".join([SPECIAL_CHAR] * (self.fill - len(data))) + data
        X = []
        for char in data:
            X.append(self.s_to_i[char])
        X = torch.tensor([X[-self.fill :]])
        O = self.network(X)
        probs = O.softmax(dim=1)
        ix = torch.multinomial(probs.squeeze(0), 1).item()
        return self.i_to_s[ix]
