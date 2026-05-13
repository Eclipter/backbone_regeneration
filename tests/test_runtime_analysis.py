from base2backbone.runtime.run_artifacts import load_analysis_run_artifacts


class _FakeDataset:
    def __len__(self):
        return 5

    central_virtual = [1, 3]
    edge_virtual = [0, 4]


class _FakeModel:
    def __init__(self):
        self.to_calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self


def test_load_analysis_run_artifacts_collects_run_state(monkeypatch):
    fake_model = _FakeModel()

    monkeypatch.setattr(
        'base2backbone.runtime.run_artifacts.resolve_run_dir',
        lambda run_id: f'/tmp/logs/{run_id}',
    )
    monkeypatch.setattr(
        'base2backbone.runtime.run_artifacts.find_best_checkpoint',
        lambda run_dir: f'{run_dir}/checkpoints/best.ckpt',
    )
    monkeypatch.setattr(
        'base2backbone.runtime.run_artifacts.glob',
        lambda pattern: ['/tmp/logs/torsions/baseline/events.out.tfevents'],
    )
    monkeypatch.setattr(
        'base2backbone.runtime.run_artifacts.torch.load',
        lambda path, weights_only=False: _FakeDataset(),
    )
    monkeypatch.setattr(
        'base2backbone.runtime.run_artifacts.torch.cuda.is_available',
        lambda: False,
    )
    monkeypatch.setattr(
        'base2backbone.runtime.run_artifacts.BackboneLightningModule.load_from_checkpoint',
        lambda path, weights_only=False, map_location='cpu': fake_model,
    )

    got = load_analysis_run_artifacts('torsions/baseline')

    assert got.run_id == 'torsions/baseline'
    assert got.run_dir == '/tmp/logs/torsions/baseline'
    assert got.ckpt_path == '/tmp/logs/torsions/baseline/checkpoints/best.ckpt'
    assert got.test_dataset_path == '/tmp/logs/torsions/baseline/test_dataset.pt'
    assert got.event_files == ['/tmp/logs/torsions/baseline/events.out.tfevents']
    assert got.target_modes == ('all', 'central', 'edge')
    assert got.test_indices_per_mode == {
        'all': [0, 1, 2, 3, 4],
        'central': [1, 3],
        'edge': [0, 4],
    }
    assert len(got.test_datasets['all']) == 5
    assert len(got.test_datasets['central']) == 2
    assert len(got.test_datasets['edge']) == 2
    assert got.model is fake_model
    assert fake_model.eval_called
    assert fake_model.to_calls == ['cpu']
    assert got.device == 'cpu'
