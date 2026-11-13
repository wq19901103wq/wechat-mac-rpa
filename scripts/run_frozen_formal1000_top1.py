"""Resume single-example diagnostic from the frozen formal1000 Top-3 plan."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'data/reports/private'
OUT = REPORTS / 'few_shot_ranker_lr_top1_formal1000_diagnostic'
PLAN = REPORTS / 'few_shot_ranker_lr_top3_formal1000_plan'
POOL = REPORTS / 'few_shot_ranker_samples_v10_formal1000/samples.jsonl'
REFERENCE = REPORTS / 'few_shot_ranker_lr_top3_formal1000_parallel16/labels/shard00/manifest.json'


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'frozen file differs: {path}')
        return
    path.write_text(text)
    path.chmod(0o600)


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n'


def prepare():
    reference = json.loads(REFERENCE.read_text())
    for key in ('c0_config', 'judge_config'):
        source = ROOT / reference[key]['path']
        assert digest(source) == reference[key]['sha256'], f'{key} changed since Top-3'
        put(OUT / source.name, source.read_text())
    selections = []
    for row in read_rows(PLAN / 'selections.jsonl'):
        record = {k: row[k] for k in ('target_id', 'relationship')}
        for arm in ('production', 'lr'):
            for field in ('sample_id', 'candidate_id', 'candidate_rank'):
                record[f'{arm}_{field}'] = row[f'{arm}_{field}s'][0]
        record['same_top1'] = record['production_candidate_id'] == record['lr_candidate_id']
        selections.append(record)
    assert len(selections) == 1000
    put(OUT / 'plan/selections.jsonl', ''.join(map(dump, selections)))
    put(OUT / 'plan/manifest.json', dump({
        'kind': 'frozen_lr_top1_diagnostic', 'source_plan_sha256': digest(PLAN / 'selections.jsonl'),
        'files': {'selections_sha256': digest(OUT / 'plan/selections.jsonl')},
        'protocol': {'training': 'none; slice frozen Top-3 selections', 'private': 'production unchanged',
                     'adoption': 'diagnostic only; do not change defaults'},
    }))
    wanted = {r[f'{a}_sample_id'] for r in selections for a in ('production', 'lr')}
    samples = {r['sample_id']: r for r in read_rows(POOL) if r['sample_id'] in wanted}
    assert len(samples) == len(wanted)
    # Reuse only exact samples and exact adopted configuration hashes.
    reused, origins = {}, {}
    for manifest_path in sorted(REPORTS.glob('few_shot*/**/manifest.json')):
        if OUT in manifest_path.parents:
            continue
        metadata = json.loads(manifest_path.read_text())
        if metadata.get('parameters', {}).get('bound_few_shot_count') != 1:
            continue
        if any(metadata.get(k, {}).get('sha256') != reference[k]['sha256'] for k in ('c0_config', 'judge_config')):
            continue
        label_path = manifest_path.parent / 'labels.jsonl'
        source = ROOT / metadata.get('samples_path', '')
        if not label_path.exists() or not source.is_file() or digest(source) != metadata.get('samples_sha256'):
            continue
        matching = {r['sample_id'] for r in read_rows(source) if r.get('sample_id') in samples and r == samples[r['sample_id']]}
        for label in read_rows(label_path):
            sid = label.get('sample_id')
            if sid in matching and sid not in reused and label.get('status') == 'ok':
                reused[sid] = label
                origins[sid] = str(label_path.relative_to(ROOT))
    put(OUT / 'reuse_sources.json', dump(origins))
    targets = sorted({r['target']['case_id'] for r in samples.values()})
    partition = {t: i % 16 for i, t in enumerate(targets)}
    for shard in range(16):
        rows = [r for r in samples.values() if partition[r['target']['case_id']] == shard]
        put(OUT / f'sources/shard{shard:02}.jsonl', ''.join(map(dump, rows)))
        seed = [reused[r['sample_id']] for r in rows if r['sample_id'] in reused]
        checkpoint = OUT / f'labels/shard{shard:02}/labels.jsonl'
        if not checkpoint.exists():
            put(checkpoint, ''.join(map(dump, seed)))
    print(dump({'targets': 1000, 'unique_top1_runs': len(samples), 'reused': len(reused),
                'changed_group': sum(not r['same_top1'] for r in selections)}), flush=True)
    return reference


def main():
    os.umask(0o077)
    reference = prepare()
    if '--prepare-only' in sys.argv:
        return
    def worker(shard):
        command = [sys.executable, 'scripts/label_few_shot_ranker_samples.py',
                   '--samples', str(OUT / f'sources/shard{shard:02}.jsonl'),
                   '--manifest', reference['manifest_path'],
                   '--output-dir', str(OUT / f'labels/shard{shard:02}'),
                   '--all', '--resume', '--retry-errors']
        with (OUT / f'shard{shard:02}.log').open('a') as log:
            return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        codes = list(executor.map(worker, range(16)))
    if any(codes):
        raise RuntimeError(f'shard exit codes: {codes}; resume after resolving errors')
    command = [sys.executable, 'scripts/evaluate_few_shot_ranker_expanded.py',
               '--selections', str(OUT / 'plan/selections.jsonl'), '--candidate-variant', 'lr',
               '--output', str(OUT / 'evaluation')]
    for shard in range(16):
        command += ['--samples', str(OUT / f'sources/shard{shard:02}.jsonl'),
                    '--labels', str(OUT / f'labels/shard{shard:02}/labels.jsonl')]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
