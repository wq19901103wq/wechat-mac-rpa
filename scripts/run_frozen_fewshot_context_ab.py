"""Isolated static-example removal / six-example diagnostics; never adopts a default."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_frozen_formal1000_top1 import REPORTS, PLAN, POOL, REFERENCE, digest, dump, put, read_rows
from scripts.prepare_few_shot_ranker_expanded_evaluation import prepare as prepare_plan
from scripts.build_few_shot_ranker_expanded_label_samples import _bound_topk_sample
from src.reply.few_shot import PersonaFewShotRetriever

OUT = REPORTS / 'few_shot_static_and_top6_formal1000_diagnostic'
SKILL = ROOT / 'skills/casual_chat/SKILL.md'
ARMS = ('no_static_top3', 'static_top6')


class ExportPayloads:
    def __init__(self, export_dirs):
        self.paths = {}
        self.cached_id = None
        self.cached_payload = None
        for directory in export_dirs:
            for path in sorted(directory.glob('*.json')):
                normalized = path.stem
                for prefix in ('私聊_', '群聊_', '曾经的好友_'):
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix):]
                        break
                chat_id = 'chat_' + hashlib.sha256(normalized.strip().encode('utf-8')).hexdigest()[:10]
                self.paths.setdefault(chat_id, []).append(path)

    def get(self, chat_id):
        if chat_id == self.cached_id:
            return self.cached_payload
        self.cached_id = None
        self.cached_payload = None
        paths = self.paths.get(chat_id, [])
        if not paths:
            return None
        from scripts.label_few_shot_ranker_samples import _json_sha256
        payload = json.loads(paths[0].read_text(encoding='utf-8'))
        for path in paths[1:]:
            duplicate = json.loads(path.read_text(encoding='utf-8'))
            if _json_sha256(payload) != _json_sha256(duplicate):
                raise ValueError('duplicate export chat has conflicting payloads')
        self.cached_id, self.cached_payload = chat_id, payload
        return payload


def strip_examples(text):
    blocks = re.findall(r'  <examples\b[^>]*>.*?</examples>\n', text, re.S)
    if len(blocks) != 1 or blocks[0].count('<example>') != 3:
        raise ValueError('expected exactly three casual-chat static examples')
    return text.replace(blocks[0], '', 1)


def prepare():
    ref = json.loads(REFERENCE.read_text())
    for key in ('c0_config', 'judge_config'):
        path = ROOT / ref[key]['path']
        if digest(path) != ref[key]['sha256']:
            raise ValueError('adopted configuration changed')
        put(OUT / path.name, path.read_text())
    put(OUT / 'casual_chat.original.txt', SKILL.read_text())
    put(OUT / 'casual_chat.no_examples.txt', strip_examples(SKILL.read_text()))
    original = read_rows(PLAN / 'selections.jsonl')
    inputs = json.loads((PLAN / 'manifest.json').read_text())['inputs']
    for item in [inputs['train_samples'], *inputs['train_labels'], inputs['expanded_evaluation_samples']]:
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('frozen training/evaluation source changed')
    six_plan = OUT / 'top6_plan'
    if not six_plan.exists():
        prepare_plan(ROOT / inputs['train_samples']['path'],
                     tuple(ROOT / x['path'] for x in inputs['train_labels']), POOL,
                     six_plan, model_kind='lr', selection_count=6,
                     ranker_relationships=('group',))
    six = read_rows(six_plan / 'selections.jsonl')
    previous = {r['target_id']: r for r in original}
    for row in six:
        for variant in ('production', 'lr'):
            if row[f'{variant}_candidate_ids'][:3] != previous[row['target_id']][f'{variant}_candidate_ids']:
                raise ValueError('Top-6 does not extend the frozen Top-3')
    pool = {r['sample_id']: r for r in read_rows(POOL)}
    baseline = {}
    for path in sorted((REPORTS / 'few_shot_ranker_lr_top3_formal1000_parallel16/labels').glob('*/labels.jsonl')):
        for row in read_rows(path):
            if row.get('status') == 'ok':
                baseline[row['sample_id']] = row
    metadata = {'protocol': 'diagnostic only; defaults unchanged; LR group-only',
                'base_c0': ref['c0_config'], 'judge': ref['judge_config'],
                'skill_sha256': digest(SKILL), 'arms': {},
                'legacy_static_field': 'generation.static_few_shot_count refers only to bound retrieval, not skill examples',
                'source_manifest': ref['manifest_path']}
    for arm, selections, k in ((ARMS[0], original, 3), (ARMS[1], six, 6)):
        rows, budgets = [], []
        for selection in selections:
            variants = ['production'] + ([] if selection[f'same_top{k}'] else ['lr'])
            for variant in variants:
                row = _bound_topk_sample([pool[sid] for sid in selection[f'{variant}_sample_ids']],
                                        variant=variant, selection=selection, selection_count=k)
                row['baseline_sample_id'] = row['sample_id'] if k == 3 else None
                row['sample_id'] = f'{arm}:{row["sample_id"]}'
                text, ids = PersonaFewShotRetriever.render(row['candidates'], max_chars=10**8)
                if ids != row['generation']['injected_candidate_ids']:
                    raise ValueError('rendering changed candidate binding')
                budgets.append(len(text))
                rows.append(row)
        budget = 2500 if k == 3 else max(2500, max(budgets))
        put(OUT / arm / 'selections.jsonl', ''.join(map(dump, selections)))
        put(OUT / arm / 'samples.jsonl', ''.join(map(dump, rows)))
        partition = {tid: i % 16 for i, tid in enumerate(sorted(r['target_id'] for r in selections))}
        for shard in range(16):
            subset = [r for r in rows if partition[r['target']['case_id']] == shard]
            put(OUT / arm / f'sources/{shard:02}.jsonl', ''.join(map(dump, subset)))
        metadata['arms'][arm] = {'targets': len(selections), 'unique_runs': len(rows),
            'selection_count': k, 'max_chars': budget,
            'exceed_original_budget': sum(n > 2500 for n in budgets),
            'static_policy': 'remove only casual_chat examples' if k == 3 else 'retain original skills',
            'samples_sha256': digest(OUT / arm / 'samples.jsonl')}
    put(OUT / 'baseline_labels.jsonl', ''.join(dump(baseline[s]) for s in sorted(baseline)))
    put(OUT / 'experiment.json', dump(metadata))
    print(dump(metadata['arms']), flush=True)
    return metadata


def successful_ids(arm):
    return {r['sample_id'] for path in (OUT / arm / 'labels').glob('*.jsonl')
            for r in read_rows(path) if r.get('status') == 'ok'}


def prepare_pending():
    pending = {}
    for arm in ARMS:
        existing = successful_ids(arm)
        pending[arm] = [r for r in read_rows(OUT / arm / 'samples.jsonl')
                        if r['sample_id'] not in existing]
    ids = {arm: [r['sample_id'] for r in rows] for arm, rows in pending.items()}
    directory = OUT / 'resumes' / hashlib.sha256(dump(ids).encode()).hexdigest()[:16]
    for arm, rows in pending.items():
        for shard in range(16):
            put(directory / arm / f'{shard:02}.jsonl', ''.join(map(dump, rows[shard::16])))
    put(directory / 'manifest.json', dump({'pending_sample_ids': ids,
        'samples_sha256': {arm: digest(OUT / arm / 'samples.jsonl') for arm in ARMS}}))
    return directory


def worker(arm, shard, pending_plan=None):
    if arm not in ARMS or shard not in range(16):
        raise ValueError('unknown experiment shard')
    lock_path = OUT / arm / f'{shard:02}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if pending_plan is None:
            _worker(arm, shard)
        else:
            _worker(arm, shard, pending_plan)


def run_c0_with_audit(arm, sample, audit, run, *args, **kwargs):
    from scripts.label_few_shot_ranker_samples import FewShotBindingError
    expected = sample['generation']['injected_candidate_ids']

    def record(skills, ids):
        audit['loaded_skills'] = list(skills)
        audit['injected_candidate_ids'] = list(ids)
        if ids != expected:
            raise FewShotBindingError('C0 did not use exactly the bound few-shots')
        audit['static_treatment_applied'] = arm == ARMS[0] and 'casual_chat' in skills
        if arm == ARMS[0] and not audit['static_treatment_applied']:
            audit['reuse'] = True
            raise RuntimeError('reuse unchanged treatment baseline')

    retriever = kwargs['persona_few_shot_retriever']
    render = retriever.render_selected

    def render_after_routing(*render_args, **render_kwargs):
        content, ids = render(*render_args, **render_kwargs)
        record(kwargs['reply_generator'].last_loaded_skills, ids)
        return content, ids

    if arm == ARMS[0]:
        retriever.render_selected = render_after_routing
    try:
        result = run(*args, **kwargs)
        record(result['trace']['loaded_skills'], result['few_shot_ids'])
        return result
    finally:
        if arm == ARMS[0]:
            del retriever.render_selected


def _worker(arm, shard, pending_plan=None):
    from scripts import label_few_shot_ranker_samples as lab
    from src.reply.c0_config import load_current_c0
    from src.reply.c0_factory import create_current_c0_generator
    metadata = json.loads((OUT / 'experiment.json').read_text())
    for key, filename in (('base_c0', 'current_c0.json'), ('judge', 'current_judge.json')):
        if digest(ROOT / metadata[key]['path']) != metadata[key]['sha256']:
            raise ValueError('default changed during experiment')
    if digest(SKILL) != metadata['skill_sha256']:
        raise ValueError('skill changed during experiment')
    output = OUT / arm / f'labels/{shard:02}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = successful_ids(arm)
    source = (pending_plan / arm / f'{shard:02}.jsonl' if pending_plan is not None
              else OUT / arm / f'sources/{shard:02}.jsonl')
    samples = read_rows(source)
    if all(r['sample_id'] in existing for r in samples):
        return
    manifest = json.loads((ROOT / metadata['source_manifest']).read_text())
    cases = [r for rows in manifest['splits'].values() for r in rows] if manifest.get('splits') else manifest['cases']
    cases = {str(r.get('case_id') or r.get('id')): {
        **r, 'case_id': str(r.get('case_id') or r.get('id')),
        'relationship': 'private' if r.get('relationship') == 'acquaintance' else r.get('relationship'),
        'replies': r.get('replies') or r.get('reply') or []} for r in cases}
    payloads = ExportPayloads(lab.DEFAULT_EXPORT_DIRS)
    judge_run = lab.resolve_judge_run(judge_config_path=OUT / 'current_judge.json')
    judge = lab.load_formal_judge(judge_run.artifact_path)
    overrides = {'casual_chat': (OUT / 'casual_chat.no_examples.txt').read_text()} if arm == ARMS[0] else {}
    generator = create_current_c0_generator(config=load_current_c0(OUT / 'current_c0.json'),
        skill_content_overrides=overrides, persona_few_shot_max_chars=metadata['arms'][arm]['max_chars'])
    baseline = {r['sample_id']: r for r in read_rows(OUT / 'baseline_labels.jsonl')}
    for sample in samples:
        if sample['sample_id'] in existing:
            continue
        audit = {}
        def run_c0(*args, **kwargs):
            return run_c0_with_audit(arm, sample, audit, lab._run_c0, *args, **kwargs)
        try:
            row = lab.label_one(sample, manifest_by_case_id=cases, payloads=payloads, judge=judge,
                c0_generator=generator, c0_runner=run_c0,
                feature_timeout_seconds=judge_run.timeout_seconds, feature_model=judge_run.model,
                feature_reasoning_effort=judge_run.reasoning_effort)
        except Exception as exc:
            if audit.get('reuse'):
                row = dict(baseline[sample['baseline_sample_id']])
                row['sample_id'] = sample['sample_id']
                row['reused_baseline_sample_id'] = sample['baseline_sample_id']
            else:
                row = {'sample_id': sample['sample_id'], 'status': 'error', 'error_type': lab._safe_error_type(exc)}
        row['experiment_arm'] = arm
        row['audit'] = audit
        lab.append_jsonl_durable(output, row)
        print(dump({'sample_id': row['sample_id'], 'status': row['status']}), flush=True)


def summarize():
    baseline = {r['sample_id']: r for r in read_rows(OUT / 'baseline_labels.jsonl')}
    sources = {r['sample_id']: r for r in read_rows(POOL)}
    original = {r['target_id']: r for r in read_rows(PLAN / 'selections.jsonl')}
    report = {'status': 'complete', 'direction': 'lower AI detection is better',
              'default_changed': False, 'comparisons': {},
              'caveats': ['LR applies only to group; private uses production selection',
                          'Repeated use of this evaluation set is diagnostic, not a new untouched holdout',
                          'Static examples are conditional on skill routing; non-hit cases reuse baseline',
                          'No fact-quality adoption review in this diagnostic']}
    for arm in ARMS:
        successes = {}
        for path in sorted((OUT / arm / 'labels').glob('*.jsonl')):
            for row in read_rows(path):
                if row.get('status') == 'ok':
                    successes[row['sample_id']] = row
        samples = read_rows(OUT / arm / 'samples.jsonl')
        missing = [r['sample_id'] for r in samples if r['sample_id'] not in successes]
        if missing:
            raise RuntimeError(f'{arm}: {len(missing)} labels missing; no complete conclusion')
        by_target = {(r['target']['case_id'], r['evaluation_variant']): successes[r['sample_id']] for r in samples}
        selections = read_rows(OUT / arm / 'selections.jsonl')
        stats = {}
        for variant in ('production', 'lr'):
            for relation in ('all', 'group', 'private'):
                pairs = []
                for sel in selections:
                    if relation != 'all' and sel['relationship'] != relation:
                        continue
                    tid = sel['target_id']
                    old = original[tid]
                    old_variant = 'production' if old['same_top3'] else variant
                    bound = _bound_topk_sample([sources[s] for s in old[f'{old_variant}_sample_ids']],
                        variant=old_variant, selection=old, selection_count=3)
                    before = baseline[bound['sample_id']]
                    after = by_target.get((tid, variant), by_target[(tid, 'production')])
                    pairs.append((1 - before['label'], 1 - after['label']))
                n = len(pairs)
                b, a = sum(x for x, _ in pairs), sum(y for _, y in pairs)
                stats[f'{variant}_{relation}'] = {'n': n, 'baseline_ai_detected': b,
                    'candidate_ai_detected': a, 'baseline_rate': b / n, 'candidate_rate': a / n,
                    'delta_percentage_points': 100 * (a - b) / n,
                    'improved': sum(x > y for x, y in pairs), 'regressed': sum(x < y for x, y in pairs)}
        report['comparisons'][arm] = stats
    put(OUT / 'result.json', dump(report))
    print(dump(report), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=2, metavar=('ARM', 'SHARD'))
    parser.add_argument('--workers', type=int, choices=range(1, 17), default=4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--queued-shards', action='store_true')
    parser.add_argument('--resume-balanced', action='store_true')
    parser.add_argument('--pending-plan', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), args.pending_plan)
        return
    prepare()
    if args.prepare_only:
        return
    pending_plan = prepare_pending() if args.resume_balanced else None
    def run(job):
        arm, shard = job
        extra = ['--pending-plan', str(pending_plan)] if pending_plan is not None else []
        with (OUT / arm / f'{shard:02}.log').open('a') as log:
            return subprocess.run([sys.executable, __file__, '--worker', arm, str(shard), *extra],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    shards = range(8, 16) if args.queued_shards else range(16)
    jobs = [(arm, shard) for shard in shards for arm in ARMS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    if any(codes):
        raise RuntimeError(f'worker failures: {codes}')
    if not args.queued_shards:
        summarize()


if __name__ == '__main__':
    main()
