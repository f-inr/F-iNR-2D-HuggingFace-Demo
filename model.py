# MIT License

# Copyright (c) [2026] [Tim Büchner, Sai Karthikeya Vemuri, Joachim Denzler]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__all__ = ["get_model_2D", "MLPType", "EmbeddingType", "DecompositionType"]

from abc import ABC
from enum import Enum
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLPType(Enum):
    TANH = "TANH"
    RELU = "ReLU"
    WIRE = "WIRE"
    SIREN = "SIREN"
    SIREN2 = "SIREN2"
    FINER = "FINER"
    NEURBF = "NeuRBF"


class EmbeddingType(Enum):
    PE000 = "PE000"
    PE010 = "PE010"
    PE020 = "PE020"
    PE100 = "PE100"
    HE = "HE"


class DecompositionType(Enum):
    BASELINE = "Baseline"
    CP = "CP"
    TT = "TT"
    TU = "TU"
    TR = "TR"


class NeuRBF1D(nn.Module):
    num_rbfs: int
    feature_dim: int

    @nn.compact
    def __call__(self, x):
        centers = self.param("centers", nn.initializers.uniform(), (self.num_rbfs, 1))
        log_sigma = self.param("log_sigma", nn.initializers.zeros, (self.num_rbfs, 1))
        sigma = jnp.exp(log_sigma) + 1e-6
        freq = self.param("freq", nn.initializers.normal(stddev=5.0), (1, self.feature_dim))
        bias = self.param("bias", nn.initializers.zeros, (1, self.feature_dim))
        features = self.param("features", nn.initializers.normal(stddev=0.1), (self.num_rbfs, self.feature_dim))
        x_exp = x[:, None, :]
        c = centers[None, :, :]
        s = sigma[None, :, :]
        sq_dist = ((x_exp - c) ** 2) / (s**2)
        rbf_vals = 1.0 / (1.0 + sq_dist.sum(-1))
        composed = jnp.sin(rbf_vals[:, :, None] * freq + bias)
        modulated = composed * features[None, :, :]
        aggregated = jnp.sum(modulated, axis=1)
        h = nn.Dense(self.feature_dim)(aggregated)
        h = jnp.sin(h * freq[0]) + h
        return nn.Dense(self.feature_dim)(h)


class RealGaborLayer(nn.Module):
    in_features: int
    out_features: int
    bias: bool = True
    is_first: bool = False
    omega0: float = 10.0
    sigma0: float = 10.0

    def setup(self):
        self.omega_0 = self.omega0
        self.scale_0 = self.sigma0
        self.freqs = nn.Dense(self.out_features, use_bias=self.bias)
        self.scale = nn.Dense(self.out_features, use_bias=self.bias)

    def __call__(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        return jnp.cos(omega) * jnp.exp(-(scale**2))


class SineLayer(nn.Module):
    in_features: int
    out_features: int
    bias: bool = True
    is_first: bool = False
    omega_0: float = 30.0
    init_weights: bool = True

    def setup(self):
        self.linear = nn.Dense(self.out_features, use_bias=self.bias, kernel_init=self.init_weights_fn())

    def init_weights_fn(self):
        if self.is_first:

            def init(key, shape, dtype=jnp.float32):
                limit = 1.0 / shape[0]
                return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
        else:

            def init(key, shape, dtype=jnp.float32):
                limit = jnp.sqrt(6.0 / shape[0]) / self.omega_0
                return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

        return init

    def __call__(self, input):
        return jnp.sin(self.omega_0 * self.linear(input))


class SimpleHashEncoder1D(nn.Module):
    L: int
    F: int
    N_min: int
    N_max: int
    T: int = 2**14

    @property
    def b(self) -> jax.Array:
        return jnp.exp((jnp.log(self.N_max) - jnp.log(self.N_min)) / (self.L - 1))

    @nn.compact
    def __call__(self, x: jax.Array, bound: float) -> jax.Array:
        x = (x + bound) / (2 * bound)
        scales = self.N_min * (self.b ** jnp.arange(self.L)) - 1
        x_scaled = x[:, None] * scales[None, :] + 0.5
        indices = jnp.floor(x_scaled).astype(jnp.int32) % self.T
        embeddings = self.param("hash_table", lambda key, shape: jax.random.uniform(key, shape, minval=-0.001, maxval=0.001), (self.T, self.F))
        return embeddings[indices].reshape(x.shape[0], -1)


class BACKEND(ABC, nn.Module):
    features: list
    r: int
    in_dim: int
    out_dim: int
    embedding: EmbeddingType
    mlp: MLPType
    L: int = 16
    F: int = 2
    N_min: int = 16
    N_max: int = 524288
    T: int = 2**14

    def setup(self):
        if self.embedding == EmbeddingType.HE:
            self.hash_encoder = SimpleHashEncoder1D(L=self.L, F=self.F, N_min=self.N_min, N_max=self.N_max, T=self.T)

    def encode(self, input):
        if self.mlp == MLPType.NEURBF:
            return input
        if self.embedding == EmbeddingType.HE:
            return self.hash_encoder(input, 1.0)
        elif self.embedding == EmbeddingType.PE000:
            return input
        elif self.embedding == EmbeddingType.PE010:
            pos_enc = 10
        elif self.embedding == EmbeddingType.PE020:
            pos_enc = 20
        elif self.embedding == EmbeddingType.PE100:
            pos_enc = 100
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding}")
        freq = jnp.array([[2**k for k in range(-((pos_enc - 1) // 2), ((pos_enc + 1) // 2))]])
        return jnp.concatenate((jnp.sin(input @ freq), jnp.cos(input @ freq)), axis=1)

    def create_subnetwork(self, decomposition: Optional[DecompositionType] = None):
        layers = []
        if self.mlp == MLPType.RELU:
            init = nn.initializers.glorot_uniform()
            for fs in self.features[:-1]:
                layers.append(nn.Dense(fs, kernel_init=init))
                layers.append(nn.relu)
            layers.append(nn.Dense(self.r * self.out_dim, kernel_init=init))
            return nn.Sequential(layers)
        elif self.mlp == MLPType.TANH:
            init = nn.initializers.xavier_normal()
            for fs in self.features[:-1]:
                layers.append(nn.Dense(fs, kernel_init=init))
                layers.append(nn.tanh)
            layers.append(nn.Dense(self.r * self.out_dim, kernel_init=init))
            return nn.Sequential(layers)
        elif self.mlp == MLPType.WIRE:
            omega, sigma = 5, 5
            for idx, fs in enumerate(self.features[:-1]):
                layers.append(RealGaborLayer(fs, fs, is_first=(idx == 0), omega0=omega, sigma0=sigma))
            layers.append(nn.Dense(self.r * self.out_dim, kernel_init=self.custom_init(False)))
            return nn.Sequential(layers)
        elif self.mlp == MLPType.SIREN:
            for idx, fs in enumerate(self.features[:-1]):
                if idx == 0:
                    layers.append(nn.Dense(fs, kernel_init=self.custom_init(True)))
                    layers.append(self.scaled_sine_activation)
                else:
                    layers.append(nn.Dense(fs, kernel_init=self.custom_init(False)))
                    layers.append(self.sine_activation)
            layers.append(nn.Dense(self.r * self.out_dim, kernel_init=self.custom_init(False)))
            return nn.Sequential(layers)
        elif self.mlp == MLPType.FINER:
            for fs in self.features[:-1]:
                layers.append(nn.Dense(fs, kernel_init=self.finer_init(0.5)))
                layers.append(self.finer_activation)
            layers.append(nn.Dense(self.r * self.out_dim, kernel_init=self.finer_init(0.5)))
            return nn.Sequential(layers)
        elif self.mlp == MLPType.NEURBF:
            return NeuRBF1D(num_rbfs=self.r, feature_dim=self.r * self.out_dim)
        raise ValueError(f"Unsupported MLP type: {self.mlp}")

    def custom_init(self, is_first):
        def init(key, shape, dtype=jnp.float32):
            limit = 1.0 / shape[0] if is_first else jnp.sqrt(6.0 / shape[0]) / 100
            return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

        return init

    def finer_init(self, scale=1.0):
        def init(key, shape, dtype=jnp.float32):
            limit = scale / jnp.sqrt(shape[0])
            return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

        return init

    @staticmethod
    def sine_activation(x):
        return jnp.sin(30 * x)

    @staticmethod
    def scaled_sine_activation(x):
        return jnp.sin(100.0 * x)

    @staticmethod
    def finer_activation(x):
        return jnp.sin((jnp.abs(x) + 1.0) * x)


class INR_Baseline2D(BACKEND):
    def setup(self):
        super().setup()
        self.network = self.create_subnetwork()

    def __call__(self, x, y):
        x, y = self.encode(x), self.encode(y)
        X = jnp.concatenate([x, y], axis=1)
        return self.network(X)


class FINR_CP_2D(BACKEND):
    def setup(self):
        super().setup()
        self.network_x = self.create_subnetwork()
        self.network_y = self.create_subnetwork()

    def __call__(self, x, y):
        x, y = self.encode(x), self.encode(y)
        out_x, out_y = self.network_x(x), self.network_y(y)
        out_x, out_y = jnp.transpose(out_x, (1, 0)), jnp.transpose(out_y, (1, 0))
        pred = []
        for i in range(self.out_dim):
            pred.append(jnp.einsum("fx, fy->xy", out_x[self.r * i : self.r * (i + 1)], out_y[self.r * i : self.r * (i + 1)]))
        return pred


def get_model_2D(backend=MLPType.RELU, embedding=EmbeddingType.PE100, decomp=DecompositionType.CP, rank=128, **kwargs):
    if decomp == DecompositionType.BASELINE:
        return INR_Baseline2D(r=rank, embedding=embedding, mlp=backend, in_dim=2, out_dim=3, **kwargs)
    elif decomp == DecompositionType.CP:
        return FINR_CP_2D(r=rank, embedding=embedding, mlp=backend, in_dim=2, out_dim=3, **kwargs)
    raise ValueError(f"Unsupported decomposition type: {decomp}")
