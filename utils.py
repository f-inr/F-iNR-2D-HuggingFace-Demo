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

import jax.numpy as jnp
import numpy as np


def img_train_generator(u: np.ndarray):
    x = jnp.linspace(-1, 1, u.shape[0]).reshape(-1, 1)
    y = jnp.linspace(-1, 1, u.shape[1]).reshape(-1, 1)
    return x, y, (u[:, :, 0], u[:, :, 1], u[:, :, 2])


def baseline_train_generator(u):
    x = np.linspace(-1, 1, u.shape[1])
    y = np.linspace(-1, 1, u.shape[0])
    X, Y = np.meshgrid(x, y)
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    coordinates = np.concatenate([X, Y], axis=1)
    flat_u = u.reshape(u.shape[0] * u.shape[1], 3)
    return X, Y, coordinates, flat_u


def img_loss(apply_fn, *train_data):
    x, y, u = train_data

    def fn(params):
        rp, gp, bp = apply_fn(params, x, y)
        return jnp.mean(jnp.square(rp - u[0])) + jnp.mean(jnp.square(gp - u[1])) + jnp.mean(jnp.square(bp - u[2]))

    return fn
