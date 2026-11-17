"""Check validation limits passed by the training entry points to Lightning."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.parametrize('script', ['train.py', 'finetune.py', 'train_prior.py'])
@pytest.mark.parametrize('smoke,size,expected', [(False, 105, 100), (False, 20, 20), (True, 105, 1)])
def test_validation_batch_limit(script, smoke, size, expected):
    path = Path(__file__).parents[1] / 'rave' / 'scripts' / script
    module = ast.parse(path.read_text())
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'val_check' for t in node.targets))
    selected = [main.body[start]]
    for node in main.body[start + 1:]:
        if not isinstance(node, ast.If):
            break
        selected.append(node)
    flags = SimpleNamespace(smoke_test=smoke, val_every=1)
    scope = {'FLAGS': flags, 'train': [None, None]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), scope)

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.validation_batches = 0
        def training_step(self, batch, batch_idx):
            return self.weight.square()
        def validation_step(self, batch, batch_idx):
            self.validation_batches += 1
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    model = Model()
    train = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    val = DataLoader(TensorDataset(torch.zeros(size, 1)), batch_size=1)
    trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0,
                         max_steps=1, **scope['val_check'])
    trainer.fit(model, train, val)
    assert model.validation_batches == expected
