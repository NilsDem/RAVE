"""Exercise the fine-tuning path with small networks, without audio dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


# Load the actual methods without importing optional audio packages.
source = ast.parse((Path(__file__).parents[1] / 'rave/model.py').read_text())
model = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'RAVE')
names = {'configure_optimizers', '_prepare_finetune_step', '_finetune_update',
         'training_step', 'on_save_checkpoint', 'on_load_checkpoint'}
methods = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name in names]
for method in methods:
    method.decorator_list = []
namespace = {'torch': torch, 'Profiler': lambda: SimpleNamespace(tick=lambda *args: None),
             '_pqmf_encode': lambda pqmf, x: x}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
             'rave/model.py', 'exec'), namespace)


class Encoder(nn.Linear):
    beta = 1.

    def set_warmed_up(self, state):
        pass

    def reparametrize(self, z):
        return z, self.beta * z.square().mean()


class Decoder(Encoder):
    pass


class Discriminator(nn.Linear):
    def forward(self, x):
        return [[super().forward(x)]]


class Toy(nn.Module):
    def __init__(self, mode, delay=0):
        super().__init__()
        self.encoder = Encoder(3, 3)
        self.decoder = Decoder(3, 3)
        self.discriminator = Discriminator(3, 1)
        self.finetune_mode = mode
        self.finetune_steps = 0
        self.encoder_start_step = delay
        self.finetune_lrs = (1e-3, 1e-2, 1e-2)
        self.finetune_regularization = None
        self.warmed_up = True
        self.output_mode = 'raw'
        self.pqmf = None
        self.multiband_audio_distance = None
        self.valid_signal_crop = False
        self.feature_matching_fun = lambda x, y: (x - y).abs().mean()
        self.num_skipped_features = 0
        self.beta_factor = 1.
        self.weights = {}
        self.gradient_clip_gen = self.gradient_clip_dis = None
        self.losses = {}
        selected = [name for name, _ in self.named_parameters()
                    if mode != 'encoder' or name.startswith('encoder.')]
        self.opts = self.configure_optimizers(selected)

    def encode(self, x, return_mb=False):
        z = self.encoder(x)
        return (z, x) if return_mb else z

    def split_features(self, features):
        return ([[f[:2] for f in scale] for scale in features],
                [[f[2:] for f in scale] for scale in features])

    def gan_loss(self, real, fake):
        return (real - 1).square().mean() + fake.square().mean(), (fake - 1).square().mean()

    def audio_distance(self, x, y):
        return {'mse': (x - y).square().mean()}

    def optimizers(self):
        return self.opts

    def manual_backward(self, loss):
        loss.backward()

    def loss_weight(self, key, step):
        return 1.

    def log(self, *args):
        pass

    def log_dict(self, losses):
        self.losses = losses


for name in names:
    setattr(Toy, name, namespace[name])


def changed(model, before):
    return {name.split('.')[0] for name, p in model.named_parameters()
            if not torch.equal(p, before[name])}


def step(model):
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.training_step(torch.randn(2, 3), 0)
    return changed(model, before)


def test_encoder_reconstruction_and_regularization():
    model = Toy('encoder')
    assert step(model) == {'encoder'}
    assert 'adversarial' not in model.losses
    assert model.losses['regularization'].item() > 0
    model.finetune_regularization = 0.
    assert step(model) == {'encoder'}
    assert model.losses['regularization'].item() == 0


def test_full_delay_and_learning_rates():
    model = Toy('full', delay=1)
    assert [group['lr'] for group in model.opts[0].param_groups] == [1e-3, 1e-2]
    assert step(model) == {'decoder', 'discriminator'}
    assert step(model) == {'encoder', 'decoder', 'discriminator'}


def test_alternating_and_resume():
    model = Toy('alternating')
    assert step(model) == {'decoder', 'discriminator'}
    assert 'feature_matching' not in model.losses
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    resumed = Toy('alternating')
    resumed.on_load_checkpoint(checkpoint)
    assert resumed.finetune_steps == 1
    assert step(resumed) == {'encoder'}
    assert 'adversarial' not in resumed.losses
    assert resumed.losses['regularization'].item() > 0
    assert step(resumed) == {'decoder', 'discriminator'}
    with pytest.raises(ValueError):
        Toy('full').on_load_checkpoint(checkpoint)


def test_beta_regularization_and_resume():
    model = Toy('alternating')
    model.encoder.beta = 0.25
    z = torch.ones(2, 3)
    assert model.encoder.reparametrize(z)[1].item() == 0.25
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    assert checkpoint['finetune']['beta'] == 0.25
    resumed = Toy('alternating')
    with pytest.raises(ValueError):
        resumed.on_load_checkpoint(checkpoint)
    resumed.encoder.beta = 0.25
    resumed.on_load_checkpoint(checkpoint)
