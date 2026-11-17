"""Replay frozen drafts through the same semantic review used before sending."""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reply.digital_human import _load_runtime, load_digital_human_config
from src.reply.exposure_guard import ContextPrefetch, review_replies, shared_review_prompt


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'validation'], required=True)
    parser.add_argument('--workers', type=int, default=1, choices=[1])
    parser.add_argument('--review-client', choices=['private', 'group'])
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--prefetch-context', action='store_true')
    parser.add_argument('--trace-requests', action='store_true',
                        help='Record real review requests and usage without enabling response cache')
    parser.add_argument('--review-model-config', type=Path,
                        help='Use an isolated review client with this full JSON model configuration')
    parser.add_argument('--context-model-config', type=Path,
                        help='Override only context analysis with an isolated full JSON model configuration')
    parser.add_argument('--indexed-context', action='store_true',
                        help='Add explicit zero-based message indices to the review conversation')
    parser.add_argument('--direct-context', action='store_true',
                        help='Review the complete source directly without a separate context summary request')
    parser.add_argument('--primary-prompt', type=Path)
    parser.add_argument('--context-prompt', type=Path)
    parser.add_argument('--verify-prompt', type=Path)
    parser.add_argument('--compare-context-prompt', type=Path,
                        help='Compare production and candidate context prompts, alternating AB/BA')
    parser.add_argument('--compare-reference-prefix', action='store_true',
                        help='Compare conversation-first and reference-first review payloads, alternating AB/BA')
    parser.add_argument('--compare-shared-prefix', action='store_true',
                        help='Compare current review and shared stage/reference prefix, alternating AB/BA')
    parser.add_argument('--compare-merged-generation', action='store_true',
                        help='Generate real drafts with context prefetch versus direct-source review, alternating AB/BA')
    parser.add_argument('--case-id', action='append', help='Select a frozen diagnostic subset, in file order')
    args = parser.parse_args()
    try:
        validate_merged_mode(args)
    except ValueError as error:
        parser.error(str(error))
    if args.trace_requests and (not args.sequential or args.prefetch_context):
        parser.error('--trace-requests requires --sequential without prefetch')
    if args.review_model_config and (not args.trace_requests or not args.sequential
            or args.prefetch_context or args.compare_context_prompt):
        parser.error('--review-model-config requires isolated sequential request tracing')
    if args.context_model_config and (not args.trace_requests or not args.sequential
            or args.prefetch_context or args.compare_context_prompt or args.review_model_config
            or args.direct_context):
        parser.error('--context-model-config requires isolated sequential request tracing without other model overrides')
    if args.indexed_context and (not args.trace_requests or not args.sequential
            or args.prefetch_context or args.compare_context_prompt):
        parser.error('--indexed-context requires isolated sequential request tracing')
    if args.direct_context and (not args.trace_requests or not args.sequential or args.prefetch_context
            or args.compare_context_prompt or args.context_prompt):
        parser.error('--direct-context requires sequential tracing without context prefetch or prompt overrides')
    if args.prefetch_context and not args.sequential:
        parser.error('--prefetch-context requires --sequential')
    if args.prefetch_context and args.context_prompt:
        parser.error('--prefetch-context cannot use --context-prompt')
    if args.compare_context_prompt and (not args.sequential or args.prefetch_context
            or args.primary_prompt or args.context_prompt or args.verify_prompt):
        parser.error('--compare-context-prompt requires isolated --sequential mode')
    if args.compare_reference_prefix and (not args.trace_requests or not args.sequential
            or args.prefetch_context or args.compare_context_prompt or args.primary_prompt or args.context_prompt
            or args.verify_prompt or args.review_model_config or args.context_model_config
            or args.indexed_context or args.direct_context):
        parser.error('--compare-reference-prefix requires isolated sequential request tracing')
    if args.compare_shared_prefix and (not args.trace_requests or not args.sequential
            or args.prefetch_context or args.compare_context_prompt or args.compare_reference_prefix
            or args.primary_prompt or args.context_prompt or args.verify_prompt or args.review_model_config
            or args.context_model_config or args.indexed_context or args.direct_context):
        parser.error('--compare-shared-prefix requires isolated sequential request tracing')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_digital_human_config()
    factory = _load_runtime(cfg['repository'])
    runtime_config = {key: value for key, value in cfg.items() if key not in ('repository', 'review_client', 'direct_context')}
    with closing(factory(**runtime_config)) as runtime:
        evaluate(args, cfg, runtime, initialization_seconds=time.monotonic() - started)


def evaluate(args, cfg, runtime, *, initialization_seconds=0.0):
    started = time.monotonic()
    validate_merged_mode(args)
    merged_comparison = getattr(args, 'compare_merged_generation', False)
    comparison_path = getattr(args, 'compare_context_prompt', None)
    reference_comparison = getattr(args, 'compare_reference_prefix', False)
    shared_comparison = getattr(args, 'compare_shared_prefix', False)
    prefetch_context = getattr(args, 'prefetch_context', False)
    indexed_context = getattr(args, 'indexed_context', False)
    direct_context = getattr(args, 'direct_context', False)
    if type(shared_comparison) is not bool or (shared_comparison and (
            not getattr(args, 'trace_requests', False) or not args.sequential or prefetch_context or comparison_path
            or reference_comparison or any(getattr(args, key, None) for key in (
                'primary_prompt', 'context_prompt', 'verify_prompt', 'review_model_config',
                'context_model_config', 'indexed_context', 'direct_context')))):
        raise ValueError('shared prefix comparison requires isolated sequential request tracing')
    if type(reference_comparison) is not bool or (reference_comparison and (
            not getattr(args, 'trace_requests', False) or not args.sequential or prefetch_context or comparison_path
            or any(getattr(args, key, None) for key in ('primary_prompt', 'context_prompt', 'verify_prompt',
                'review_model_config', 'context_model_config', 'indexed_context', 'direct_context')))):
        raise ValueError('reference prefix comparison requires isolated sequential request tracing')
    if type(direct_context) is not bool or (direct_context and (
            not getattr(args, 'trace_requests', False) or not args.sequential or prefetch_context
            or comparison_path or getattr(args, 'context_prompt', None))):
        raise ValueError('direct context requires sequential tracing without context prefetch or prompt overrides')
    candidate_context = comparison_path.read_text(encoding='utf-8') if comparison_path else None
    comparison = candidate_context is not None or reference_comparison or shared_comparison or merged_comparison
    if candidate_context is not None and not candidate_context.strip():
        raise ValueError('empty prompt override')
    if indexed_context and (not getattr(args, 'trace_requests', False) or not args.sequential
            or prefetch_context or comparison_path):
        raise ValueError('indexed context requires isolated sequential request tracing')
    model_config_path = getattr(args, 'review_model_config', None)
    model_config = json.loads(model_config_path.read_text()) if model_config_path else None
    if model_config_path and (not isinstance(model_config, dict) or not model_config.get('model')
            or not getattr(args, 'trace_requests', False) or not args.sequential or prefetch_context
            or comparison_path):
        raise ValueError('model configuration requires a model and isolated sequential request tracing')
    context_config_path = getattr(args, 'context_model_config', None)
    context_config = json.loads(context_config_path.read_text()) if context_config_path else None
    if context_config_path and (not isinstance(context_config, dict) or not context_config.get('model')
            or not getattr(args, 'trace_requests', False) or not args.sequential or prefetch_context
            or comparison_path or model_config_path or direct_context):
        raise ValueError('context model requires a model and isolated sequential request tracing without other model overrides')
    prompt_overrides = {key: path.read_text(encoding='utf-8') for key in ('primary', 'context', 'verify')
                        if (path := getattr(args, key + '_prompt', None)) is not None}
    if any(not value.strip() for value in prompt_overrides.values()):
        raise ValueError('empty prompt override')
    settings = yaml.safe_load((cfg.get('settings_file') or cfg['repository']/'config/settings-default.yaml').read_text())
    instance_dir = cfg['instances_root'] / cfg['instance']
    pointers_path = instance_dir / 'pointers.json'
    generator_id = json.loads(pointers_path.read_text())['production_gen']
    gen_path = instance_dir / 'generators' / generator_id / 'config.json'
    llm = json.loads(gen_path.read_text())['llm']
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        root/'prompts/reply_review.md', root/'prompts/reply_review_context.md',
        root/'prompts/reply_review_verify.md',
        root/'prompts/reply_review_shared.md',
        root/'src/reply/exposure_guard.py', root/'src/reply/digital_human.py',
        root/'src/reply/dh_runtime.py', root/'src/reply/dh_worker.py',
        Path(__file__).resolve()]}
    review_client = args.review_client or cfg.get('review_client')
    prompts = {key: prompt_overrides.get(key, (root / ('prompts/' + filename)).read_text())
               for key, filename in [('primary', 'reply_review.md'), ('context', 'reply_review_context.md'),
                                     ('verify', 'reply_review_verify.md')]}
    snapshot = dict(split=args.split, llm=llm, review_client=review_client, code_sha256=hashes,
                    parallel=not args.sequential, context_prefetch=prefetch_context,
                    binding=runtime.binding, source=runtime.source,
                    cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                    generator=generator_id, generator_config_sha256=hashlib.sha256(gen_path.read_bytes()).hexdigest(),
                    pointers_sha256=hashlib.sha256(pointers_path.read_bytes()).hexdigest(),
                    max_retries=settings['llm'].get('max_retries', 2),
                    client_identity=runtime.client_identity)
    snapshot['prompt_sha256'] = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in prompts.items()}
    snapshot['case_ids'] = sorted(set(getattr(args, 'case_id', None) or []))
    trace_requests = getattr(args, 'trace_requests', False)
    if trace_requests:
        if not args.sequential or prefetch_context:
            raise ValueError('request tracing requires sequential review without prefetch')
        snapshot['trace_requests'] = True
    if model_config is not None:
        snapshot['review_model_config'] = model_config
    if context_config is not None:
        snapshot['context_model_config'] = context_config
    if indexed_context:
        snapshot['indexed_context'] = True
    if direct_context:
        snapshot['direct_context'] = True
    rows = [json.loads(line) for line in args.cases.read_text().splitlines()]
    rows = [r for r in rows if r['split']==args.split]
    if snapshot['case_ids']:
        if set(snapshot['case_ids']) - {r['id'] for r in rows}:
            raise ValueError('case-id missing from selected split')
        rows = [r for r in rows if r['id'] in snapshot['case_ids']]
    if merged_comparison:
        inputs = {row['id']: generation_input(row, root) for row in rows}
        snapshot['generation_inputs'] = inputs
    order = [(row, arm) for index, row in enumerate(rows)
             for arm in (('candidate', 'baseline') if index % 2 else ('baseline', 'candidate'))] \
            if comparison else [(row, None) for row in rows]
    if comparison:
        snapshot['comparison'] = dict(
            order=[dict(id=row['id'], arm=arm) for row, arm in order])
        if merged_comparison:
            snapshot['comparison']['merged_generation'] = dict(
                baseline='generation_with_context_prefetch', candidate='generation_with_direct_source_review')
        elif shared_comparison:
            snapshot['comparison']['shared_prefix'] = dict(baseline=False, candidate=True)
            snapshot['comparison']['shared_prompt_sha256'] = hashlib.sha256(shared_review_prompt().encode()).hexdigest()
        elif reference_comparison:
            snapshot['comparison']['reference_prefix'] = dict(baseline=False, candidate=True)
        else:
            snapshot['comparison']['context_prompt_sha256'] = hashlib.sha256(candidate_context.encode()).hexdigest()
    config_path = args.output/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == snapshot, 'cannot mix different configs'
    else:
        config_path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
        for key, filename in [('primary', 'reply_review.md'), ('context', 'reply_review_context.md'),
                              ('verify', 'reply_review_verify.md')]:
            (args.output / filename).write_text(prompts[key], encoding='utf-8')
        if candidate_context is not None:
            (args.output / 'reply_review_context_candidate.md').write_text(candidate_context, encoding='utf-8')
        if shared_comparison:
            (args.output / 'reply_review_shared_candidate.md').write_text(shared_review_prompt(), encoding='utf-8')
    results_path = args.output/'results.jsonl'
    previous = [json.loads(x) for x in results_path.read_text().splitlines()] if results_path.exists() else []
    completed = {(r['id'], r.get('arm')) for r in previous}

    def run(case, arm):
        started=time.monotonic()
        result={k:case[k] for k in ('id','kind','category','material_coverage')}
        if not merged_comparison:
            result['expected'] = case['expected']
        result['chat_type']=case['conversation']['chat_type']
        if arm is not None:
            result['arm'] = arm
        try:
            client_name = review_client or case['conversation']['chat_type']
            if merged_comparison:
                return run_generation(runtime, case, arm, inputs[case['id']], client_name, args.output, result)
            review = dict(replies=case['replies'], online_input=case['conversation'],
                          target_indices=case['target_indices'], generation_messages=case['generation_material'],
                          parallel=not args.sequential,
                          prompt_overrides={'context': candidate_context} if arm == 'candidate'
                          and candidate_context is not None else prompt_overrides)
            if reference_comparison:
                review['reference_prefix'] = arm == 'candidate'
            if shared_comparison:
                review['shared_prefix'] = arm == 'candidate'
            if indexed_context:
                review['online_input'] = {**case['conversation'], 'context': [
                    dict(message, index=index) for index, message in enumerate(case['conversation']['context'])]}
            if direct_context:
                review['direct_context'] = True
            if prefetch_context:
                client = runtime.clients[client_name]
                with ThreadPoolExecutor(max_workers=1) as pool:
                    prefetch = ContextPrefetch(pool, client, case['conversation'], case['target_indices'])
                    prefetch.start(case['generation_material'])
                    result['review'] = review_replies(client, **review, prefetch=prefetch)
            else:
                tracing = dict(trace_dir=str(args.output.resolve()),
                               trace_case_id=f"{case['id']}:{arm or 'current'}") if trace_requests else {}
                if model_config is not None:
                    tracing['review_model_config'] = model_config
                if context_config is not None:
                    tracing['context_model_config'] = context_config
                result['review']=runtime.call('review', client=client_name, review=review, **tracing)
            result['status']='ok'
            result['profile'] = review_profile(result['review'])
            if trace_requests:
                result['profile'].update(request_profile(args.output, result['review']['trace_ref'],
                                                         direct_context=direct_context))
        except Exception as error:
            result.update(status='error',error=f'{type(error).__name__}: {error}')
        result['seconds']=round(time.monotonic()-started,3)
        return result

    new_results = 0
    with results_path.open('a') as stream:
        for row, arm in order:
            if (row['id'], arm) in completed:
                continue
            result=run(row, arm)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush()
            previous.append(result)
            new_results += 1
            print(json.dumps(dict(id=result['id'],expected=result.get('expected'),status=result['status'],
                                  arm=arm, decision=result.get('review',{}).get('decision'),seconds=result['seconds']),ensure_ascii=False),flush=True)
    if merged_comparison:
        summary = generation_summary(previous, rows)
    elif not comparison:
        summary = quality_summary(previous)
    else:
        arms = {arm: [r for r in previous if r['arm'] == arm] for arm in ('baseline', 'candidate')}
        paired = dict(complete=0, failed=[], improvements=[], regressions=[])
        for row in rows:
            pair = {r['arm']: r for r in previous if r['id'] == row['id']}
            if set(pair) != set(arms) or any(r['status'] != 'ok' for r in pair.values()):
                paired['failed'].append(row['id'])
                continue
            paired['complete'] += 1
            correct = {arm: r['review']['decision'] == r['expected'] for arm, r in pair.items()}
            if correct['baseline'] != correct['candidate']:
                paired['improvements' if correct['candidate'] else 'regressions'].append(row['id'])
        summary = dict(arms={arm: quality_summary(items) for arm, items in arms.items()}, paired=paired,
                       performance={arm: performance_summary(items) for arm, items in arms.items()})
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    elapsed = time.monotonic() - started
    with (args.output/'runs.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(finished_at=datetime.now(timezone.utc).isoformat(),
                                    initialization_seconds=initialization_seconds, evaluation_seconds=elapsed,
                                    preclose_wall_seconds=initialization_seconds + elapsed,
                                    new_results=new_results)) + '\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def validate_merged_mode(args):
    enabled = getattr(args, 'compare_merged_generation', False)
    if type(enabled) is not bool or (enabled and (
            not getattr(args, 'trace_requests', False) or not args.sequential
            or any(getattr(args, key, None) for key in (
                'prefetch_context', 'compare_context_prompt', 'compare_reference_prefix', 'compare_shared_prefix',
                'primary_prompt', 'context_prompt', 'verify_prompt', 'review_model_config',
                'context_model_config', 'indexed_context', 'direct_context')))):
        raise ValueError('merged generation comparison requires isolated sequential request tracing')


def generation_input(case, root):
    source = json.loads((root / case['source']).read_text())['case']
    conversation = case['conversation']
    if any(source[key] != conversation[key] for key in ('chat_type', 'chat_name', 'context')):
        raise ValueError('generation source does not match frozen conversation')
    if not source.get('source_chat_id'):
        raise ValueError('generation requires the original source chat identity')
    return dict(kind='messages', source_chat_id=source['source_chat_id'], **conversation)


def run_generation(runtime, case, arm, online_input, client_name, output, result):
    started = time.monotonic()
    try:
        trace_id = f"{case['id']}:{arm}"
        draft = runtime.call('draft', request=dict(request_id=trace_id, input=online_input),
                             review=dict(client=client_name, target_indices=case['target_indices'])
                             if arm == 'baseline' else None, trace_dir=str(output.resolve()))
        generation_seconds = time.monotonic() - started
        result['draft'] = draft
        result['generation_seconds'] = round(generation_seconds, 3)
        if draft['sent'] or not draft['draft_only']:
            raise ValueError('experiment requires an unsent draft')
        if draft['replies']:
            result['review'] = runtime.call('review', client=client_name,
                review=dict(replies=draft['replies'], online_input=online_input,
                            target_indices=case['target_indices'], generation_messages=draft['generation_messages'],
                            parallel=False, direct_context=arm == 'candidate'),
                trace_dir=str(output.resolve()), trace_case_id=trace_id + ':review')
            context = result['review'].get('context_review', {})
            result['profile'] = review_profile(result['review'])
            result['profile']['context_wait_seconds'] = context.get('wait_seconds')
            result['profile'].update(request_profile(output, result['review']['trace_ref'],
                direct_context=arm == 'candidate' or context.get('prefetched', False)))
        result['generation_requests'] = request_profile(output, draft['trace_ref'], generation=True)
        result['status'] = 'ok'
    except Exception as error:
        result.update(status='error', error=f'{type(error).__name__}: {error}')
    result['seconds'] = round(time.monotonic() - started, 3)
    return result


def generation_summary(results, cases):
    paired = dict(complete=0, failed=[], silence=[], changed_replies=[], changed_decisions=[])
    reviewed = {arm: [] for arm in ('baseline', 'candidate')}
    for case in cases:
        pair = {row['arm']: row for row in results if row['id'] == case['id']}
        if set(pair) != set(reviewed) or any(row['status'] != 'ok' for row in pair.values()):
            paired['failed'].append(case['id'])
            continue
        if any('review' not in row for row in pair.values()):
            paired['silence'].append(case['id'])
            continue
        paired['complete'] += 1
        for arm, row in pair.items():
            reviewed[arm].append(row)
        if pair['baseline']['draft']['replies'] != pair['candidate']['draft']['replies']:
            paired['changed_replies'].append(case['id'])
        if pair['baseline']['review']['decision'] != pair['candidate']['review']['decision']:
            paired['changed_decisions'].append(case['id'])
    return dict(paired=paired, quality='New drafts require individual review; old expected labels do not apply.',
                performance={arm: performance_summary(rows) for arm, rows in reviewed.items()},
                attempts=len(results), errors=[dict(id=row['id'], arm=row['arm'], error=row.get('error'))
                                              for row in results if row['status'] != 'ok'])


def review_profile(review):
    context = review.get('context_review', {})
    stages = [('context', context)]
    if review.get('reviews'):
        stages += list(zip(('primary', 'verify'), review['reviews']))
    else:
        stages.append(('primary', {**review, 'seconds': None}))
    return dict(stages={name: dict(seconds=value.get('seconds'), output_chars=len(value.get('raw', '')))
                        for name, value in stages},
                completed_chat_outputs=sum('raw' in value for _, value in stages), retries=None)


def request_profile(directory, ref, *, direct_context=False, generation=False):
    trace = json.loads((directory / 'traces' / (ref + '.json')).read_text())
    requests = []
    stage = -1
    stage_names = ('primary', 'verify') if direct_context else ('context', 'primary', 'verify')
    for operation in trace['operations']:
        for event in operation['events']:
            if event['kind'] != 'llm':
                continue
            attempt = event['request']['attempt']
            if attempt == 1:
                stage += 1
            response = event.get('response', {})
            requests.append(dict(stage='generation' if generation else
                                 stage_names[stage] if 0 <= stage < len(stage_names) else 'unknown',
                                 attempt=attempt, status=event['status'], seconds=event['elapsed_ms'] / 1000,
                                 usage=response.get('usage'), output_chars=len(response.get('text', '')),
                                 finish_reason=response.get('finish_reason', response.get('stop_reason')),
                                 content_types=response.get('content_types')))
    return dict(trace_ref=ref, network_attempts=len(requests),
                retries=sum(request['attempt'] > 1 for request in requests), requests=requests)


def performance_summary(rows):
    durations = sorted(row['seconds'] for row in rows)
    def percentile(fraction):
        if not durations:
            return None
        position = (len(durations) - 1) * fraction
        low = int(position)
        return durations[low] + (durations[min(low + 1, len(durations) - 1)] - durations[low]) * (position - low)
    return dict(attempts=len(rows), errors=sum(row['status'] != 'ok' for row in rows),
                total_seconds=sum(durations), p50_seconds=percentile(0.5), p95_seconds=percentile(0.95),
                completed_chat_outputs=sum(row.get('profile', {}).get('completed_chat_outputs', 0) for row in rows),
                context_output_chars=sum(row.get('profile', {}).get('stages', {}).get('context', {}).get('output_chars', 0)
                                         for row in rows), retries=None)


def quality_summary(rows):
    summary={}
    for kind in sorted({r['kind'] for r in rows}):
        subset=[r for r in rows if r['kind']==kind]
        bad=[r for r in subset if r['expected']=='block']
        good=[r for r in subset if r['expected']=='allow']
        caught=[r for r in bad if r.get('review',{}).get('decision')=='block']
        false=[r for r in good if r.get('review',{}).get('decision')=='block']
        summary[kind]=dict(bad=len(bad),caught=len(caught),catch_rate=len(caught)/len(bad) if bad else None,
                           controls=len(good),false_blocks=len(false),false_block_rate=len(false)/len(good) if good else None,
                           errors=[r['id'] for r in subset if r['status']=='error'],
                           misses=[r['id'] for r in bad if r.get('review',{}).get('decision')=='allow'],
                           false_block_ids=[r['id'] for r in false],
                           categories={cat:dict(total=sum(r['category']==cat for r in bad),
                               caught=sum(r['category']==cat for r in caught)) for cat in sorted({r['category'] for r in bad})})
    return summary


if __name__ == '__main__':
    main()
