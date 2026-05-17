import json

from base2backbone.data import dna_windows


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            'result_set': [
                {'identifier': '1ABC'},
                {'identifier': '2DEF'},
            ],
        }


def test_get_pdb_ids_updates_latest_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / 'manifests' / 'latest.json'
    monkeypatch.setattr(dna_windows, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(dna_windows.requests, 'post', lambda *args, **kwargs: _FakeResponse())

    pdb_ids = dna_windows.get_pdb_ids(None)

    assert pdb_ids == ['1ABC', '2DEF']
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding='utf-8')) == pdb_ids


def test_get_pdb_ids_reads_non_latest_manifest_from_file(tmp_path, monkeypatch):
    manifest_path = tmp_path / 'manifests' / 'thesis_2026-05-17.json'
    monkeypatch.setattr(dna_windows, 'REPO_ROOT', tmp_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(['3GHI', '4JKL']) + '\n', encoding='utf-8')
    monkeypatch.setattr(
        dna_windows.requests,
        'post',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('RCSB fetch must not run')),
    )

    pdb_ids = dna_windows.get_pdb_ids('thesis_2026-05-17.json')

    assert pdb_ids == ['3GHI', '4JKL']
