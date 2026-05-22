import sys
import types
import importlib.util

import gin
import pytest
import torch

from rave import blocks

gin.enter_interactive_mode()


def _parse_v4_config():
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        ["v4.gin"],
        [
            # "NFFT=256",
            # "LATENT_SIZE=8",
            # "CAPACITY=2",
            # "CAPACITY_DECODER=2",
            # "blocks.Generator2D.n_freqs=256",
            # "blocks.Generator2D.n_out=1",
        ],
    )


def _run_v4_encoder2d_generator2d_forward(x, n_channels):
    _parse_v4_config()

    encoder = gin.get_configurable("blocks.VariationalEncoder")(
        n_channels=n_channels)
    decoder = gin.get_configurable("blocks.Generator2D")(
        n_channels=n_channels)


    with torch.no_grad():
        encoded = encoder(x)
        z, kl = encoder.reparametrize(encoded)
        y = decoder(z)

    assert encoded.shape == (2, 256, 2)
    assert z.shape == (2, 128, 2)
    assert kl.ndim == 0
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_v4_encoder2d_generator2d_forward():
    x = torch.randn(2, 1, 2, 512, 16)
    _run_v4_encoder2d_generator2d_forward(x, n_channels=1)
    print("Sucess Mono ! ")


def test_v4_encoder2d_generator2d_multichannel_forward():
    x = torch.randn(2, 2, 2, 512, 16)
    _run_v4_encoder2d_generator2d_forward(x, n_channels=2)
    print("Sucess Multi channel ")

if __name__ == "__main__":
    test_v4_encoder2d_generator2d_forward()
    test_v4_encoder2d_generator2d_multichannel_forward()
