import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import evaluate_reply_guard as evaluation


@pytest.fixture
def experiment(tmp_path):
    settings = tmp_path / 'settings.yaml'
    settings.write_text('llm:\n  max_retries: 2\n')
    instance = tmp_path / 'instances' / 'test'
    generator = instance / 'generators' / 'g-test'
    generator.mkdir(parents=True)
    (instance / 'pointers.json').write_text('{"production_gen":"g-test"}')
    (generator / 'config.json').write_text('{"llm":{"group":{"model":"test"}}}')
    cases = [dict(id=f'case-{i}', split='dev', kind='historical', expected='allow', category='normal',
                  material_coverage='full_request', replies=['收到'], target_indices=[0],
                  conversation=dict(chat_type='private', chat_name='甲', context=[dict(sender='甲', text='提醒')]),
                  generation_material=[dict(role='system', content='身份资料')]) for i in range(2)]
    cases_path = tmp_path / 'cases.jsonl'
    cases_path.write_text(''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases))
    context = tmp_path / 'compact.md'
    context.write_text('紧凑：独立梳理上下文')
    args = argparse.Namespace(cases=cases_path, output=tmp_path / 'output', split='dev', workers=1,
                              review_client='group', sequential=True, prefetch_context=False,
                              primary_prompt=None, context_prompt=None, verify_prompt=None,
                              compare_context_prompt=context, case_id=None)
    args.output.mkdir()
    cfg = dict(repository=tmp_path, settings_file=settings, instances_root=tmp_path / 'instances',
               instance='test', review_client='group')
    verdict = dict(decision='allow', issues=[], seconds=3.0,
                   context_review=dict(seconds=1.0, raw='context'),
                   reviews=[dict(seconds=0.8, raw='primary'), dict(seconds=1.2, raw='verify')])
    runtime = SimpleNamespace(binding={}, source='test', client_identity={}, call=Mock(return_value=verdict))
    return args, cfg, runtime


def results(args):
    return [json.loads(line) for line in (args.output / 'results.jsonl').read_text().splitlines()]


def test_merged_generation_uses_real_drafts_and_original_sources_without_old_labels(experiment, monkeypatch):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.compare_merged_generation = args.trace_requests = True
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    for case in cases:
        source = args.cases.parent / (case['id'] + '.json')
        source.write_text(json.dumps(dict(case=dict(**case['conversation'], source_chat_id='original',
                                                   human_reply=['must not enter generation']))))
        case['source'] = str(source)
    args.cases.write_text(''.join(json.dumps(case) + '\n' for case in cases))
    verdict = runtime.call.return_value
    verdict['trace_ref'] = 'review-trace'
    verdict['context_review']['prefetched'] = True
    def call(operation, **payload):
        if operation == 'draft':
            return dict(replies=['new real draft'], generation_messages=[dict(content='real material')],
                        sent=False, draft_only=True, status='draft', trace_ref='draft-trace')
        return verdict
    runtime.call.side_effect = call
    monkeypatch.setattr(evaluation, 'request_profile', lambda *a, **kw: {})
    evaluation.evaluate(args, cfg, runtime)
    rows = results(args)
    assert [(row['id'], row['arm']) for row in rows] == [
        ('case-0', 'baseline'), ('case-0', 'candidate'), ('case-1', 'candidate'), ('case-1', 'baseline')]
    for index, row in enumerate(rows):
        draft, review = runtime.call.call_args_list[index * 2:index * 2 + 2]
        assert draft.args == ('draft',) and review.args == ('review',)
        assert draft.kwargs['review'] == (dict(client='group', target_indices=[0])
                                         if row['arm'] == 'baseline' else None)
        online = draft.kwargs['request']['input']
        assert online == dict(kind='messages', source_chat_id='original', **cases[index // 2]['conversation'])
        assert review.kwargs['review'] == dict(replies=['new real draft'], online_input=online,
            target_indices=[0], generation_messages=[dict(content='real material')], parallel=False,
            direct_context=row['arm'] == 'candidate')
        assert draft.kwargs['trace_dir'] == review.kwargs['trace_dir'] == str(args.output.resolve())
        assert 'expected' not in row and 'generation_seconds' in row
    summary = json.loads((args.output / 'summary.json').read_text())
    assert summary['paired']['complete'] == 2
    assert 'arms' not in summary and 'old expected labels do not apply' in summary['quality']
    runtime.call.reset_mock()
    evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()
    source.write_text(json.dumps(dict(case=dict(**cases[-1]['conversation'], source_chat_id='changed'))))
    with pytest.raises(AssertionError, match='different configs'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


@pytest.mark.parametrize('conflict', ['trace_requests', 'sequential', 'prefetch_context', 'direct_context',
                                     'compare_context_prompt', 'compare_shared_prefix'])
def test_merged_generation_rejects_mixed_modes(experiment, conflict):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.compare_merged_generation = args.trace_requests = True
    setattr(args, conflict, conflict not in ('trace_requests', 'sequential'))
    with pytest.raises(ValueError, match='merged generation comparison'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


def test_merged_summary_does_not_treat_silence_or_failure_as_speed_gain():
    rows = [dict(id='case', arm=arm, status='ok', seconds=1, draft=dict(replies=[]))
            for arm in ('baseline', 'candidate')]
    summary = evaluation.generation_summary(rows, [dict(id='case')])
    assert summary['paired']['complete'] == 0 and summary['paired']['silence'] == ['case']
    assert summary['performance']['candidate']['attempts'] == 0
    rows[0]['status'] = 'error'
    summary = evaluation.generation_summary(rows, [dict(id='case')])
    assert summary['paired']['failed'] == ['case'] and len(summary['errors']) == 1


def test_context_comparison_interleaves_arms_and_keeps_all_other_inputs(experiment):
    args, cfg, runtime = experiment
    evaluation.evaluate(args, cfg, runtime)
    rows = results(args)
    assert [(row['id'], row['arm']) for row in rows] == [
        ('case-0', 'baseline'), ('case-0', 'candidate'),
        ('case-1', 'candidate'), ('case-1', 'baseline')]
    for offset in (0, 2):
        left, right = [call.kwargs['review'].copy() for call in runtime.call.call_args_list[offset:offset + 2]]
        overrides = [left.pop('prompt_overrides'), right.pop('prompt_overrides')]
        assert left == right and left['parallel'] is False
        assert sorted(overrides, key=len) == [{}, {'context': args.compare_context_prompt.read_text()}]
    snapshot = json.loads((args.output / 'config.json').read_text())
    assert snapshot['comparison']['order'] == [{'id': row['id'], 'arm': row['arm']} for row in rows]
    assert (args.output / 'reply_review_context_candidate.md').read_text() == args.compare_context_prompt.read_text()
    summary = json.loads((args.output / 'summary.json').read_text())
    assert summary['paired']['complete'] == 2
    assert summary['paired']['improvements'] == summary['paired']['regressions'] == []


def test_context_comparison_resume_runs_only_missing_arm(experiment):
    args, cfg, runtime = experiment
    evaluation.evaluate(args, cfg, runtime)
    path = args.output / 'results.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:3]) + '\n')
    runtime.call.reset_mock()
    evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_called_once()
    assert runtime.call.call_args.kwargs['review']['prompt_overrides'] == {}
    assert len({(row['id'], row['arm']) for row in results(args)}) == 4
    runtime.call.reset_mock()
    evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()
    runs = [json.loads(line) for line in (args.output / 'runs.jsonl').read_text().splitlines()]
    assert [run['new_results'] for run in runs] == [4, 1, 0]


@pytest.mark.parametrize('option', ['reference_prefix', 'shared_prefix'])
def test_prefix_comparison_changes_only_option_and_resumes_missing_arm(experiment, monkeypatch, option):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    setattr(args, 'compare_' + option, True)
    args.trace_requests = True
    runtime.call.return_value['trace_ref'] = 'test'
    monkeypatch.setattr(evaluation, 'request_profile', lambda *a, **kw: {})
    evaluation.evaluate(args, cfg, runtime)
    assert [(row['id'], row['arm']) for row in results(args)] == [
        ('case-0', 'baseline'), ('case-0', 'candidate'), ('case-1', 'candidate'), ('case-1', 'baseline')]
    for offset in (0, 2):
        left, right = [call.kwargs['review'].copy() for call in runtime.call.call_args_list[offset:offset + 2]]
        assert {left.pop(option), right.pop(option)} == {False, True}
        assert left == right and left['prompt_overrides'] == {} and left['parallel'] is False
    snapshot = json.loads((args.output / 'config.json').read_text())
    assert snapshot['comparison'][option] == dict(baseline=False, candidate=True)
    if option == 'shared_prefix':
        assert (args.output / 'reply_review_shared_candidate.md').read_text() == evaluation.shared_review_prompt()
    path = args.output / 'results.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:3]) + '\n')
    runtime.call.reset_mock()
    evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_called_once()
    assert runtime.call.call_args.kwargs['review'][option] is False
    runtime.call.reset_mock()
    setattr(args, 'compare_' + option, False)
    with pytest.raises(AssertionError, match='different configs'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


@pytest.mark.parametrize('conflict', ['trace_requests', 'sequential', 'prefetch_context',
                                     'compare_context_prompt', 'context_prompt', 'review_model_config'])
@pytest.mark.parametrize('option', ['reference_prefix', 'shared_prefix'])
def test_prefix_comparison_rejects_nonisolated_runs(experiment, conflict, option):
    args, cfg, runtime = experiment
    prompt = args.compare_context_prompt
    args.compare_context_prompt = None
    setattr(args, 'compare_' + option, True)
    args.trace_requests = True
    setattr(args, conflict, False if conflict in ('trace_requests', 'sequential') else
            True if conflict == 'prefetch_context' else prompt)
    with pytest.raises(ValueError, match=option.replace('_', ' ') + ' comparison'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


def test_comparison_refuses_changed_prompt_before_requests(experiment):
    args, cfg, runtime = experiment
    evaluation.evaluate(args, cfg, runtime)
    args.compare_context_prompt.write_text('不同的紧凑分析要求')
    runtime.call.reset_mock()
    with pytest.raises(AssertionError, match='different configs'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


@pytest.mark.parametrize('extra', [[], ['--sequential', '--prefetch-context'],
                                  ['--sequential', '--context-prompt', 'other.md'],
                                  ['--sequential', '--primary-prompt', 'other.md'],
                                  ['--sequential', '--verify-prompt', 'other.md']])
def test_comparison_rejects_nonisolated_modes_before_runtime(monkeypatch, tmp_path, extra):
    factory = Mock()
    monkeypatch.setattr(evaluation, '_load_runtime', factory)
    monkeypatch.setattr('sys.argv', ['evaluate_reply_guard', '--cases', str(tmp_path / 'cases.jsonl'),
                                    '--output', str(tmp_path / 'out'), '--split', 'dev',
                                    '--compare-context-prompt', str(tmp_path / 'compact.md'), *extra])
    with pytest.raises(SystemExit):
        evaluation.main()
    factory.assert_not_called()
    assert not (tmp_path / 'out').exists()


def test_failed_arm_is_not_retried_or_counted_as_correct(experiment):
    args, cfg, runtime = experiment
    allowed = runtime.call.return_value
    runtime.call.side_effect = [allowed, RuntimeError('parse failed'), allowed, allowed]
    evaluation.evaluate(args, cfg, runtime)
    runtime.call.reset_mock(side_effect=True)
    evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()
    summary = json.loads((args.output / 'summary.json').read_text())
    assert summary['paired']['complete'] == 1
    assert summary['paired']['failed'] == ['case-0']
    assert summary['arms']['candidate']['historical']['errors'] == ['case-0']


def test_single_arm_remains_unlabelled_and_resumable(experiment):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    evaluation.evaluate(args, cfg, runtime)
    assert len(results(args)) == 2 and all('arm' not in row for row in results(args))
    assert 'historical' in json.loads((args.output / 'summary.json').read_text())
    runtime.call.reset_mock()
    evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


def test_empty_comparison_prompt_is_rejected_without_calls(experiment):
    args, cfg, runtime = experiment
    args.compare_context_prompt.write_text('  ')
    with pytest.raises(ValueError, match='empty prompt'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


def test_profile_does_not_count_overall_guard_time_as_primary_time():
    review = dict(seconds=8.0, raw='primary', context_review=dict(seconds=5.0, raw='context'))
    profile = evaluation.review_profile(review)
    assert profile['stages']['context']['seconds'] == 5.0
    assert profile['stages']['primary']['seconds'] is None
    assert profile['completed_chat_outputs'] == 2
    assert profile['retries'] is None


def test_profile_keeps_separate_primary_and_verify_measurements(experiment):
    _, _, runtime = experiment
    profile = evaluation.review_profile(runtime.call.return_value)
    assert {stage: data['seconds'] for stage, data in profile['stages'].items()} == {
        'context': 1.0, 'primary': 0.8, 'verify': 1.2}
    assert profile['completed_chat_outputs'] == 3


def test_request_profile_counts_real_retries_and_preserves_usage(tmp_path):
    events = [dict(kind='cache_lookup', status='miss')]
    for attempt, status, usage in [(1, 'failed', None), (2, 'ok', {'input_tokens': 80, 'output_tokens': 12}),
                                   (1, 'ok', {'input_tokens': 90, 'output_tokens': 8})]:
        events.append(dict(kind='llm', request=dict(attempt=attempt), status=status, elapsed_ms=1250,
                           response=dict(usage=usage, text='{}', stop_reason='end_turn', content_types=['text'])))
    (tmp_path / 'traces').mkdir()
    (tmp_path / 'traces' / 'test.json').write_text(json.dumps(dict(operations=[dict(events=events)])))
    profile = evaluation.request_profile(tmp_path, 'test')
    assert profile['network_attempts'] == 3 and profile['retries'] == 1
    assert [r['stage'] for r in profile['requests']] == ['context', 'context', 'primary']
    assert [r['usage'] for r in profile['requests']] == [None, {'input_tokens': 80, 'output_tokens': 12},
                                                        {'input_tokens': 90, 'output_tokens': 8}]


def test_direct_context_profile_binds_requests_to_primary_and_verify(tmp_path):
    events = [dict(kind='llm', request=dict(attempt=attempt), status='ok', elapsed_ms=1000,
                   response=dict(text='{}')) for attempt in (1, 2, 1)]
    (tmp_path / 'traces').mkdir()
    (tmp_path / 'traces/test.json').write_text(json.dumps(dict(operations=[dict(events=events)])))
    profile = evaluation.request_profile(tmp_path, 'test', direct_context=True)
    assert [r['stage'] for r in profile['requests']] == ['primary', 'primary', 'verify']
    assert profile['network_attempts'] == 3 and profile['retries'] == 1
    profile = evaluation.review_profile(dict(context_review=dict(skipped=True, seconds=0.0),
        reviews=[dict(raw='primary', seconds=1.0), dict(raw='verify', seconds=1.0)]))
    assert profile['completed_chat_outputs'] == 2
    assert profile['stages']['context'] == dict(seconds=0.0, output_chars=0)


def test_direct_context_is_opt_in_frozen_and_preserves_all_other_inputs(experiment):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.trace_requests = True
    args.case_id = ['case-0']
    (args.output / 'traces').mkdir()
    (args.output / 'traces/test.json').write_text('{"operations":[]}')
    runtime.call.return_value = {**runtime.call.return_value, 'trace_ref': 'test'}
    evaluation.evaluate(args, cfg, runtime)
    baseline = runtime.call.call_args.kwargs['review']
    assert 'direct_context' not in baseline
    args.direct_context = True
    runtime.call.reset_mock()
    with pytest.raises(AssertionError, match='different configs'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()
    args.output = args.output.parent / 'direct'
    (args.output / 'traces').mkdir(parents=True)
    (args.output / 'traces/test.json').write_text('{"operations":[]}')
    evaluation.evaluate(args, cfg, runtime)
    assert runtime.call.call_args.kwargs['review'] == dict(baseline, direct_context=True)
    assert json.loads((args.output / 'config.json').read_text())['direct_context'] is True


@pytest.mark.parametrize('extra', [[], ['--sequential'], ['--trace-requests'],
    ['--sequential', '--trace-requests', '--prefetch-context'],
    ['--sequential', '--trace-requests', '--context-prompt', 'other.md'],
    ['--sequential', '--trace-requests', '--compare-context-prompt', 'other.md']])
def test_direct_context_rejects_conflicting_cli_modes_before_runtime(monkeypatch, tmp_path, extra):
    factory = Mock()
    monkeypatch.setattr(evaluation, '_load_runtime', factory)
    monkeypatch.setattr('sys.argv', ['evaluate_reply_guard', '--cases', str(tmp_path / 'cases.jsonl'),
        '--output', str(tmp_path / 'out'), '--split', 'dev', '--direct-context', *extra])
    with pytest.raises(SystemExit):
        evaluation.main()
    factory.assert_not_called()
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('conflict', ['untraced', 'parallel', 'prefetch_context', 'context_prompt',
                                     'compare_context_prompt', 'invalid_type'])
def test_direct_context_rejects_conflicting_evaluate_modes_before_requests(experiment, conflict):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.trace_requests = True
    args.direct_context = True
    if conflict == 'untraced':
        args.trace_requests = False
    elif conflict == 'parallel':
        args.sequential = False
    elif conflict == 'invalid_type':
        args.direct_context = 1
    else:
        setattr(args, conflict, True)
    with pytest.raises(ValueError, match='direct context requires'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


def test_tracing_is_opt_in_and_keeps_review_payload(experiment):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.trace_requests = True
    args.case_id = ['case-0']
    (args.output / 'traces').mkdir()
    (args.output / 'traces' / 'test.json').write_text('{"operations":[]}')
    runtime.call.return_value = {**runtime.call.return_value, 'trace_ref': 'test'}
    evaluation.evaluate(args, cfg, runtime)
    call = runtime.call.call_args
    assert call.kwargs['trace_dir'] == str(args.output.resolve())
    assert call.kwargs['trace_case_id'] == 'case-0:current'
    assert call.kwargs['review']['replies'] == ['收到']
    assert results(args)[0]['profile']['network_attempts'] == 0
    assert json.loads((args.output / 'config.json').read_text())['trace_requests'] is True


@pytest.mark.parametrize('extra', [[], ['--sequential', '--prefetch-context']])
def test_request_tracing_rejects_unmeasured_parallel_modes_before_runtime(monkeypatch, tmp_path, extra):
    factory = Mock()
    monkeypatch.setattr(evaluation, '_load_runtime', factory)
    monkeypatch.setattr('sys.argv', ['evaluate_reply_guard', '--cases', str(tmp_path / 'cases.jsonl'),
                                    '--output', str(tmp_path / 'out'), '--split', 'dev',
                                    '--trace-requests', *extra])
    with pytest.raises(SystemExit):
        evaluation.main()
    factory.assert_not_called()


@pytest.mark.parametrize('config_key', ['review_model_config', 'context_model_config'])
def test_model_override_is_frozen_and_preserves_the_full_review(experiment, config_key):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.trace_requests = True
    args.case_id = ['case-0']
    (args.output / 'traces').mkdir()
    (args.output / 'traces' / 'test.json').write_text('{"operations":[]}')
    runtime.call.return_value = {**runtime.call.return_value, 'trace_ref': 'test'}
    evaluation.evaluate(args, cfg, runtime)
    baseline = runtime.call.call_args.kwargs['review']
    assert 'review_model_config' not in runtime.call.call_args.kwargs
    assert 'context_model_config' not in runtime.call.call_args.kwargs
    model_path = args.output.parent / 'model.json'
    setattr(args, config_key, model_path)
    model_config = dict(model='test', thinking=dict(type='disabled'))
    model_path.write_text(json.dumps(model_config))
    runtime.call.reset_mock()
    with pytest.raises(AssertionError, match='different configs'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()
    args.output = args.output.parent / 'candidate'
    (args.output / 'traces').mkdir(parents=True)
    (args.output / 'traces' / 'test.json').write_text('{"operations":[]}')
    evaluation.evaluate(args, cfg, runtime)
    assert runtime.call.call_args.kwargs['review'] == baseline
    assert runtime.call.call_args.kwargs[config_key] == model_config
    assert json.loads((args.output / 'config.json').read_text())[config_key] == model_config
    model_path.write_text(json.dumps({**model_config, 'temperature': 0.5}))
    runtime.call.reset_mock()
    with pytest.raises(AssertionError, match='different configs'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


@pytest.mark.parametrize('extra', [[], ['--sequential'], ['--trace-requests'],
    ['--sequential', '--trace-requests', '--prefetch-context'],
    ['--sequential', '--trace-requests', '--compare-context-prompt', 'other.md'],
    ['--sequential', '--trace-requests', '--review-model-config', 'other.json'],
    ['--sequential', '--trace-requests', '--direct-context']])
def test_context_model_override_rejects_conflicts_before_runtime(monkeypatch, tmp_path, extra):
    factory = Mock()
    monkeypatch.setattr(evaluation, '_load_runtime', factory)
    monkeypatch.setattr('sys.argv', ['evaluate_reply_guard', '--cases', str(tmp_path / 'cases.jsonl'),
        '--output', str(tmp_path / 'out'), '--split', 'dev',
        '--context-model-config', str(tmp_path / 'model.json'), *extra])
    with pytest.raises(SystemExit):
        evaluation.main()
    factory.assert_not_called()
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('extra', [[], ['--sequential'], ['--trace-requests'],
    ['--sequential', '--trace-requests', '--prefetch-context'],
    ['--sequential', '--trace-requests', '--compare-context-prompt', 'other.md']])
def test_model_override_rejects_uncontrolled_modes_before_runtime(monkeypatch, tmp_path, extra):
    factory = Mock()
    monkeypatch.setattr(evaluation, '_load_runtime', factory)
    monkeypatch.setattr('sys.argv', ['evaluate_reply_guard', '--cases', str(tmp_path / 'cases.jsonl'),
                                    '--output', str(tmp_path / 'out'), '--split', 'dev',
                                    '--review-model-config', str(tmp_path / 'model.json'), *extra])
    with pytest.raises(SystemExit):
        evaluation.main()
    factory.assert_not_called()
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('override_keys', [(), ('primary',), ('verify',), ('primary', 'verify'),
                                         ('context',), ('primary', 'context', 'verify')])
def test_indexed_context_binds_nonadjacent_targets_without_changing_source_or_other_inputs(experiment, override_keys):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.trace_requests = True
    args.indexed_context = True
    args.review_model_config = args.output / 'model.json'
    args.review_model_config.write_text('{"model":"test","thinking":{"type":"disabled"}}')
    for key in override_keys:
        path = args.output.parent / (key + '.md')
        path.write_text('历史玩笑不能豁免当前身份风险')
        setattr(args, key + '_prompt', path)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    for case in cases:
        case['conversation']['context'] = [dict(sender='甲', text='请看看', is_self=False),
                                           dict(sender='本人', text='已买好', is_self=True),
                                           dict(sender='甲', text='记得设闹钟', is_self=False)]
        case['target_indices'] = [2, 0]
    args.cases.write_text(''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases))
    source_bytes = args.cases.read_bytes()
    (args.output / 'traces').mkdir()
    (args.output / 'traces' / 'test.json').write_text('{"operations":[]}')
    runtime.call.return_value = {**runtime.call.return_value, 'trace_ref': 'test'}
    evaluation.evaluate(args, cfg, runtime)
    for case, call in zip(cases, runtime.call.call_args_list):
        review = call.kwargs['review']
        indexed = review['online_input']['context']
        assert [(indexed[i]['index'], indexed[i]['text']) for i in review['target_indices']] == [
            (2, '记得设闹钟'), (0, '请看看')]
        unindexed = [{k: v for k, v in message.items() if k != 'index'} for message in indexed]
        assert {**review['online_input'], 'context': unindexed} == case['conversation']
        assert review['replies'] == case['replies']
        assert review['generation_messages'] == case['generation_material']
        overrides = {key: getattr(args, key + '_prompt').read_text() for key in override_keys}
        assert review['prompt_overrides'] == overrides and review['parallel'] is False
        assert call.kwargs['review_model_config'] == json.loads(args.review_model_config.read_text())
    assert args.cases.read_bytes() == source_bytes
    assert json.loads((args.output / 'config.json').read_text())['indexed_context'] is True
    for key in override_keys:
        filename = {'primary': 'reply_review.md', 'context': 'reply_review_context.md',
                    'verify': 'reply_review_verify.md'}[key]
        path = getattr(args, key + '_prompt')
        assert (args.output / filename).read_text() == path.read_text()
        path.write_text('不同的身份风险边界')
        runtime.call.reset_mock()
        with pytest.raises(AssertionError, match='different configs'):
            evaluation.evaluate(args, cfg, runtime)
        runtime.call.assert_not_called()
        path.write_text('历史玩笑不能豁免当前身份风险')
    args.indexed_context = False
    runtime.call.reset_mock()
    with pytest.raises(AssertionError, match='different configs'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()


@pytest.mark.parametrize('extra', [[], ['--sequential'], ['--trace-requests'],
    ['--sequential', '--trace-requests', '--prefetch-context'],
    ['--sequential', '--trace-requests', '--compare-context-prompt', 'other.md']])
def test_indexed_context_rejects_uncontrolled_modes_before_runtime(monkeypatch, tmp_path, extra):
    factory = Mock()
    monkeypatch.setattr(evaluation, '_load_runtime', factory)
    monkeypatch.setattr('sys.argv', ['evaluate_reply_guard', '--cases', str(tmp_path / 'cases.jsonl'),
                                    '--output', str(tmp_path / 'out'), '--split', 'dev',
                                    '--indexed-context', *extra])
    with pytest.raises(SystemExit):
        evaluation.main()
    factory.assert_not_called()
    assert not (tmp_path / 'out').exists()


def test_cli_accepts_prompt_overrides_with_indexed_model_trace(experiment, monkeypatch):
    args, cfg, runtime = experiment
    runtime.close = Mock()
    factory = Mock(return_value=runtime)
    runner = Mock()
    monkeypatch.setattr(evaluation, 'load_digital_human_config', lambda: cfg)
    monkeypatch.setattr(evaluation, '_load_runtime', lambda _: factory)
    monkeypatch.setattr(evaluation, 'evaluate', runner)
    monkeypatch.setattr('sys.argv', ['evaluate_reply_guard', '--cases', str(args.cases),
        '--output', str(args.output), '--split', 'dev', '--sequential', '--trace-requests',
        '--indexed-context', '--review-model-config', str(args.output.parent / 'model.json'),
        '--primary-prompt', str(args.output.parent / 'primary.md'),
        '--context-prompt', str(args.output.parent / 'context.md'),
        '--verify-prompt', str(args.output.parent / 'verify.md')])
    evaluation.main()
    parsed, passed_cfg, passed_runtime = runner.call_args.args
    assert parsed.primary_prompt == args.output.parent / 'primary.md'
    assert parsed.context_prompt == args.output.parent / 'context.md'
    assert parsed.verify_prompt == args.output.parent / 'verify.md'
    assert parsed.review_model_config == args.output.parent / 'model.json'
    assert parsed.sequential and parsed.trace_requests and parsed.indexed_context
    assert passed_cfg == cfg and passed_runtime is runtime
    runtime.close.assert_called_once()


@pytest.mark.parametrize('mode', ['indexed_context', 'review_model_config'])
@pytest.mark.parametrize('uncontrolled', ['compare_context_prompt', 'prefetch_context', 'parallel', 'untraced'])
def test_evaluate_rejects_uncontrolled_modes_with_context_override(experiment, mode, uncontrolled):
    args, cfg, runtime = experiment
    args.compare_context_prompt = None
    args.trace_requests = True
    if mode == 'review_model_config':
        args.review_model_config = args.output.parent / 'model.json'
        args.review_model_config.write_text('{"model":"test"}')
    else:
        args.indexed_context = True
    args.context_prompt = args.output.parent / 'other.md'
    args.context_prompt.write_text('仅梳理与待回复消息相关的事件')
    if uncontrolled == 'compare_context_prompt':
        args.compare_context_prompt = args.context_prompt
    elif uncontrolled == 'parallel':
        args.sequential = False
    elif uncontrolled == 'untraced':
        args.trace_requests = False
    else:
        args.prefetch_context = True
    with pytest.raises(ValueError, match='isolated sequential request tracing'):
        evaluation.evaluate(args, cfg, runtime)
    runtime.call.assert_not_called()
