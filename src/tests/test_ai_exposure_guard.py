import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.models.base import ChatMessage, SenderType
from src.reply.digital_human import DigitalHumanError, DigitalHumanReplyGenerator, load_digital_human_config
from src.reply.exposure_guard import ContextPrefetch, capture_generation, review_replies


def input_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()



def issue(category='hallucination', quote='已经转好了', reply_index=0):
    return dict(category=category, quote=quote, reply_index=reply_index, reason='上下文没有操作完成的依据')


def model_with(result):
    model = Mock(model_name='test-review-model')
    def respond(messages, **kwargs):
        if '独立梳理上下文' in messages[0]['content']:
            return '{"events":[],"targets":[]}'
        return json.dumps(result, ensure_ascii=False)
    model.chat.side_effect = respond
    return model


@pytest.mark.parametrize('category', [
    'ai_identity', 'hallucination', 'identity_confusion', 'repetition', 'no_information', 'other_ai_trait',
])
def test_every_model_issue_blocks_with_evidence(category):
    verdict = dict(decision='block', issues=[issue(category)])
    model = model_with(verdict)
    result = review_replies(model, ['已经转好了'], {'context': []}, [0], [{'role': 'system', 'content': '资料'}])
    assert result['decision'] == 'block' and result['issues'] == verdict['issues']
    assert result['model'] == 'test-review-model' and len(result['prompt_sha256']) == 64
    assert model.chat.call_args.kwargs == dict(json_mode=True)


def test_review_receives_context_targets_all_replies_and_actual_evidence():
    model = model_with(dict(decision='allow', issues=[]))
    context = dict(chat_type='group', context=[
        dict(sender='小李', is_self=False, text='我今天去上海'),
        dict(sender='本人', is_self=True, text='收到'),
        dict(sender='小王', is_self=False, text='小李你几点到'),
    ])
    material = [dict(role='system', content='本人叫小钱，不是小李'),
                dict(role='user', content='Wiki 原始证据：小钱在北京')]
    replies = ['这工具是AI助手', '有点意思']
    result = review_replies(model, replies, context, [2], material)
    assert result['decision'] == 'allow'
    payload = json.loads(model.chat.call_args.args[0][1]['content'])
    assert payload == dict(conversation=context, target_indices=[2], generation_material=material, replies=replies,
                           context_analysis=dict(events=[], targets=[]))
    assert model.chat.call_count == 3 and result['context_review']['analysis'] == dict(events=[], targets=[])


@pytest.mark.parametrize('decision', ['allow', 'block'])
def test_allow_requires_independent_verification_without_previous_verdict(decision):
    model = model_with(None)
    final = dict(decision=decision, issues=[issue()] if decision == 'block' else [])
    model.chat.side_effect = lambda messages, **kwargs: (
        '{"events":[],"targets":[]}' if '独立梳理上下文' in messages[0]['content'] else
        json.dumps(final) if '独立质检员' in messages[0]['content'] else '{"decision":"allow","issues":[]}')
    result = review_replies(model, ['已经转好了'], {}, [0], [{'role': 'system', 'content': '依据'}])
    assert result['decision'] == decision and result['issues'] == final['issues']
    primary, verification = sorted(model.chat.call_args_list[1:], key=lambda call: '独立质检员' in call.args[0][0]['content'])
    assert primary.args[0][1] == verification.args[0][1]
    assert primary.args[0][0] != verification.args[0][0]
    assert [r['decision'] for r in result['reviews']] == ['allow', decision]


def test_reference_prefix_only_changes_payload_order_and_preserves_independent_checks():
    model = model_with(dict(decision='allow', issues=[]))
    conversation = dict(context=[dict(sender='甲', text='问题')])
    material = [dict(role='system', content='本人资料'), dict(role='user', content='历史和当前原文')]
    review_replies(model, ['回复'], conversation, [0], material)
    baseline = [call.args[0] for call in model.chat.call_args_list]
    model.reset_mock()
    review_replies(model, ['回复'], conversation, [0], material, reference_prefix=True)
    candidate = [call.args[0] for call in model.chat.call_args_list]
    assert len(baseline) == len(candidate) == 3
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        assert left[0] == right[0] and left[1]['role'] == right[1]['role'] == 'user'
        before, after = [json.loads(messages[1]['content']) for messages in (left, right)]
        assert before == after
        assert list(after) == (['generation_material', 'conversation', 'target_indices'] +
                              (['replies', 'context_analysis'] if index else []))
    assert 'replies' not in json.loads(candidate[0][1]['content'])
    assert candidate[1][1] == candidate[2][1]


@pytest.mark.parametrize('prefetched_prefix,reviewed_prefix', [(False, False), (True, True),
                                                              (False, True), (True, False)])
def test_reference_prefetch_requires_same_serialized_order(prefetched_prefix, reviewed_prefix):
    model = model_with(dict(decision='allow', issues=[]))
    material = [dict(role='system', content='依据')]
    with ThreadPoolExecutor(max_workers=1) as pool:
        prefetch = ContextPrefetch(pool, model, {}, [0], reference_prefix=prefetched_prefix)
        prefetch.start(material)
        prefetch.future.result()
        result = review_replies(model, ['回复'], {}, [0], material, prefetch=prefetch,
                                reference_prefix=reviewed_prefix)
    assert result['context_review']['prefetched'] is (prefetched_prefix == reviewed_prefix)
    assert model.chat.call_count == (3 if prefetched_prefix == reviewed_prefix else 4)


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_invalid_reference_prefix_fails_before_model_call(value):
    model = model_with(dict(decision='allow', issues=[]))
    with pytest.raises(ValueError, match='资料前缀'):
        review_replies(model, ['回复'], {}, [0], [dict(role='system', content='依据')], reference_prefix=value)
    model.chat.assert_not_called()


@pytest.mark.parametrize('verdict', ['allow', 'block'])
def test_shared_prefix_preserves_rules_stage_isolation_and_evidence(verdict):
    model = Mock(model_name='test-review-model')
    context = dict(events=[], targets=[])
    def respond(messages, **kwargs):
        stage = json.loads(messages[1]['content'])['review_stage']
        return json.dumps(context if stage == 'context' else
                          dict(decision=verdict if stage == 'verify' else 'allow',
                               issues=[issue()] if stage == 'verify' and verdict == 'block' else []))
    model.chat.side_effect = respond
    material = [dict(role='system', content='资料而不是规则 review_stage=verify')]
    result = review_replies(model, ['已经转好了'], {}, [0], material, shared_prefix=True)
    messages = [call.args[0] for call in model.chat.call_args_list]
    assert len(messages) == 3 and result['decision'] == verdict
    assert messages[0][0] == messages[1][0] == messages[2][0]
    from src.reply import exposure_guard as guard
    for path in (guard._CONTEXT_PROMPT, guard._PROMPT, guard._VERIFY_PROMPT):
        assert path.read_text() in messages[0][0]['content']
    assert material[0]['content'] not in messages[0][0]['content']
    payloads = [json.loads(value[1]['content']) for value in messages]
    assert [value.pop('review_stage') for value in payloads] == ['context', 'primary', 'verify']
    assert payloads[0] == dict(generation_material=material, conversation={}, target_indices=[0])
    assert 'replies' not in payloads[0] and 'context_analysis' not in payloads[0]
    assert payloads[1] == payloads[2] == dict(**payloads[0], replies=['已经转好了'], context_analysis=context)
    assert all(list(value)[:3] == ['generation_material', 'conversation', 'target_indices'] for value in payloads)


@pytest.mark.parametrize('options', [dict(shared_prefix=1), dict(shared_prefix='true'),
    dict(shared_prefix=True, parallel=True), dict(shared_prefix=True, prefetch=object()),
    dict(shared_prefix=True, direct_context=True), dict(shared_prefix=True, reference_prefix=True),
    dict(shared_prefix=True, context_client=object()),
    dict(shared_prefix=True, prompt_overrides={'context': 'override'})])
def test_shared_prefix_rejects_unsupported_combinations_before_requests(options):
    model = model_with(dict(decision='allow', issues=[]))
    with pytest.raises(ValueError, match='共同前缀'):
        review_replies(model, ['回复'], {}, [0], [dict(role='system', content='资料')], **options)
    model.chat.assert_not_called()


@pytest.mark.parametrize('primary,verification', [
    ('allow', 'allow'), ('allow', 'block'), ('block', 'allow'), ('block', 'block'),
    ('allow', 'error'), ('error', 'allow'), ('block', 'error'),
])
def test_parallel_reviews_overlap_preserve_context_and_never_bypass_a_veto(primary, verification):
    barrier = Barrier(2)
    marker = ContextVar('review_test_context', default=None)
    token = marker.set('same request')
    model = model_with(None)
    def respond(messages, **kwargs):
        if '独立梳理上下文' in messages[0]['content']:
            return '{"events":[],"targets":[]}'
        assert marker.get() == 'same request'
        barrier.wait(timeout=2)
        decision = verification if '独立质检员' in messages[0]['content'] else primary
        if decision == 'error':
            raise TimeoutError('test reviewer failed')
        return json.dumps(dict(decision=decision, issues=[issue()] if decision == 'block' else []))
    model.chat.side_effect = respond
    try:
        if primary == 'error' or (primary == 'allow' and verification == 'error'):
            with pytest.raises(TimeoutError):
                review_replies(model, ['已经转好了'], {}, [0], [dict(role='system', content='依据')], parallel=True)
        else:
            result = review_replies(model, ['已经转好了'], {}, [0], [dict(role='system', content='依据')], parallel=True)
            assert result['decision'] == ('allow' if primary == verification == 'allow' else 'block')
            assert result['parallel'] is True
    finally:
        marker.reset(token)
    assert model.chat.call_count == 3


def test_verification_failure_never_releases_primary_allow():
    model = model_with(None)
    def respond(messages, **kwargs):
        prompt = messages[0]['content']
        if '独立质检员' in prompt:
            raise TimeoutError('verification timed out')
        return '{"events":[],"targets":[]}' if '独立梳理上下文' in prompt else '{"decision":"allow","issues":[]}'
    model.chat.side_effect = respond
    with pytest.raises(TimeoutError):
        review_replies(model, ['草稿'], {}, [0], [{'role': 'system', 'content': '依据'}])


def test_experimental_prompts_are_isolated_and_keep_independent_checks():
    model = model_with(dict(decision='allow', issues=[]))
    overrides = dict(primary='试验主审', context='试验：独立梳理上下文', verify='试验复核')
    result = review_replies(model, ['草稿'], {}, [0], [dict(role='system', content='依据')],
                            prompt_overrides=overrides)
    assert [call.args[0][0]['content'] for call in model.chat.call_args_list] == [
        overrides['context'], overrides['primary'], overrides['verify']]
    assert 'replies' not in json.loads(model.chat.call_args_list[0].args[0][1]['content'])
    assert model.chat.call_args_list[1].args[0][1] == model.chat.call_args_list[2].args[0][1]
    assert result['context_review']['prompt_sha256'] == hashlib.sha256(overrides['context'].encode()).hexdigest()
    assert [r['prompt_sha256'] for r in result['reviews']] == [
        hashlib.sha256(overrides[key].encode()).hexdigest() for key in ('primary', 'verify')]
    model.reset_mock()
    review_replies(model, ['草稿'], {}, [0], [dict(role='system', content='依据')])
    assert all(call.args[0][0]['content'] not in overrides.values() for call in model.chat.call_args_list)


@pytest.mark.parametrize('overrides', [[], {'typo': '提示'}, {'primary': ''}, {'context': None}])
def test_invalid_prompt_overrides_fail_before_model_call(overrides):
    model = model_with(dict(decision='allow', issues=[]))
    with pytest.raises(ValueError, match='提示覆盖'):
        review_replies(model, ['草稿'], {}, [0], [dict(role='system', content='依据')],
                        prompt_overrides=overrides)
    model.chat.assert_not_called()


def test_context_analysis_never_receives_draft_and_is_preserved_for_review():
    model = model_with(None)
    context = dict(events=[dict(actor='本人', event='借款')], targets=[])
    model.chat.side_effect = lambda messages, **kwargs: json.dumps(
        context if '独立梳理上下文' in messages[0]['content'] else dict(decision='block', issues=[issue()]))
    result = review_replies(model, ['已经转好了'], {}, [0], [{'role': 'system', 'content': '依据'}])
    assert result['decision'] == 'block'
    first = json.loads(model.chat.call_args_list[0].args[0][1]['content'])
    second = json.loads(model.chat.call_args_list[1].args[0][1]['content'])
    assert 'replies' not in first and '已经转好了' not in json.dumps(first, ensure_ascii=False)
    assert second['context_analysis'] == context and second['replies'] == ['已经转好了']
    assert model.chat.call_args_list[0].args[0][0] != model.chat.call_args_list[1].args[0][0]


@pytest.mark.parametrize('primary,verification', [('allow', 'allow'), ('allow', 'block'), ('block', 'allow')])
def test_isolated_context_client_preserves_full_input_and_review_checks(primary, verification):
    model = model_with(None)
    model.chat.side_effect = [json.dumps(dict(decision=value, issues=[issue()] if value == 'block' else []))
                              for value in (primary, verification)]
    context_model = Mock(model_name='context-only')
    analysis = dict(events=[dict(actor='甲', event='提醒本人')], targets=[0])
    context_model.chat.return_value = json.dumps(analysis)
    conversation = dict(context=[dict(sender='甲', text='别忘了', is_self=False)])
    material = [dict(role='user', content='完整历史和资料')]
    result = review_replies(model, ['已经转好了'], conversation, [0], material, context_client=context_model)
    context_model.chat.assert_called_once()
    assert json.loads(context_model.chat.call_args.args[0][1]['content']) == dict(
        conversation=conversation, target_indices=[0], generation_material=material)
    assert model.chat.call_count == (2 if primary == 'allow' else 1)
    for call in model.chat.call_args_list:
        assert json.loads(call.args[0][1]['content']) == dict(conversation=conversation, target_indices=[0],
            generation_material=material, replies=['已经转好了'], context_analysis=analysis)
    assert result['decision'] == (verification if primary == 'allow' else primary)
    assert result['context_review']['model'] == 'context-only'


@pytest.mark.parametrize('failure', [TimeoutError('context timed out'), 'not JSON', '{"events":[]}'])
def test_isolated_context_failure_never_reaches_review(failure):
    model = model_with(dict(decision='allow', issues=[]))
    context_model = Mock(model_name='context-only')
    context_model.chat.side_effect = [failure]
    with pytest.raises((ValueError, TimeoutError)):
        review_replies(model, ['草稿'], {}, [0], [dict(role='user', content='依据')], context_client=context_model)
    model.chat.assert_not_called()


@pytest.mark.parametrize('options', [dict(parallel=True), dict(prefetch=object()), dict(direct_context=True)])
def test_isolated_context_rejects_conflicting_modes_before_chat(options):
    model, context_model = Mock(), Mock()
    with pytest.raises(ValueError, match='独立上下文模型'):
        review_replies(model, ['草稿'], {}, [0], [dict(role='user', content='依据')],
                       context_client=context_model, **options)
    model.chat.assert_not_called()
    context_model.chat.assert_not_called()


@pytest.mark.parametrize('second', [TimeoutError('review timed out'), 'not JSON', '{"decision":"block","issues":[]}'])
def test_review_failure_after_context_analysis_never_releases(second):
    model = model_with(None)
    def respond(messages, **kwargs):
        if '独立梳理上下文' in messages[0]['content']:
            return '{"events":[],"targets":[]}'
        if isinstance(second, Exception):
            raise second
        return second
    model.chat.side_effect = respond
    with pytest.raises((ValueError, TimeoutError)):
        review_replies(model, ['草稿'], {}, [0], [{'role': 'system', 'content': '依据'}])


def test_sequential_benchmark_retains_original_short_circuit():
    model = model_with(dict(decision='block', issues=[issue()]))
    result = review_replies(model, ['已经转好了'], {}, [0], [dict(role='system', content='依据')], parallel=False)
    assert result['decision'] == 'block' and result['parallel'] is False
    assert model.chat.call_count == 2


@pytest.mark.parametrize('primary,verification', [('allow', 'allow'), ('allow', 'block'), ('block', 'allow')])
def test_direct_context_keeps_full_source_and_independent_verification(primary, verification):
    model = model_with(None)
    verdicts = [dict(decision=value, issues=[issue()] if value == 'block' else [])
                for value in (primary, verification)]
    model.chat.side_effect = [json.dumps(value) for value in verdicts]
    context = dict(context=[dict(sender='甲', text='提醒本人设闹钟', is_self=False),
                            dict(sender='本人', text='收到', is_self=True)])
    material = [dict(role='user', content='完整原始资料')]
    result = review_replies(model, ['已经转好了'], context, [0], material, direct_context=True)
    assert result['decision'] == (verification if primary == 'allow' else primary)
    assert model.chat.call_count == (2 if primary == 'allow' else 1)
    payload = dict(conversation=context, target_indices=[0], generation_material=material, replies=['已经转好了'])
    for call in model.chat.call_args_list:
        assert json.loads(call.args[0][1]['content']) == payload
        assert '独立梳理上下文' not in call.args[0][0]['content']
    if primary == 'allow':
        assert '独立质检员' in model.chat.call_args_list[1].args[0][0]['content']
    assert result['context_review'] == dict(skipped=True, seconds=0.0, prefetched=False, wait_seconds=0.0)


@pytest.mark.parametrize('failure', [TimeoutError('review timed out'), 'not JSON',
                                   '{"decision":"allow","issues":[{}]}'])
@pytest.mark.parametrize('stage', ['primary', 'verify'])
def test_direct_context_review_failure_never_releases(stage, failure):
    model = model_with(None)
    model.chat.side_effect = ([json.dumps(dict(decision='allow', issues=[]))] if stage == 'verify' else []) + [failure]
    with pytest.raises((ValueError, TimeoutError)):
        review_replies(model, ['草稿'], {}, [0], [dict(role='system', content='依据')], direct_context=True)


@pytest.mark.parametrize('options', [dict(direct_context=1), dict(direct_context=None),
    dict(direct_context=True, parallel=True), dict(direct_context=True, prefetch=object()),
    dict(direct_context=True, prompt_overrides=dict(context='摘要'))])
def test_direct_context_rejects_conflicting_modes_before_model_call(options):
    model = model_with(dict(decision='allow', issues=[]))
    with pytest.raises(ValueError, match='直接原文审查'):
        review_replies(model, ['草稿'], {}, [0], [dict(role='system', content='依据')], **options)
    model.chat.assert_not_called()


@pytest.mark.parametrize('result', [
    None, [], {}, {'decision': 'allow'}, {'decision': 'unknown', 'issues': []},
    {'decision': 'block', 'issues': []}, {'decision': 'allow', 'issues': [issue()]},
    {'decision': 'block', 'issues': [None]},
    {'decision': 'block', 'issues': [issue('unsupported')]},
    {'decision': 'block', 'issues': [issue(quote='没出现在草稿中')]},
    {'decision': 'block', 'issues': [issue(reply_index=True)]},
    {'decision': 'block', 'issues': [issue(reply_index=-1)]},
    {'decision': 'block', 'issues': [issue(reply_index=1)]},
    {'decision': 'block', 'issues': [dict(issue(), reason='')]},
])
def test_invalid_model_verdict_is_never_released(result):
    with pytest.raises(ValueError):
        review_replies(model_with(result), ['已经转好了'], {}, [0], [{'role': 'system', 'content': '依据'}])


@pytest.mark.parametrize('raw', ['', 'not JSON', '```json\n{"decision":"allow","issues":[]}\n```'])
def test_unparseable_review_fails_closed(raw):
    model = model_with(None)
    model.chat.side_effect = None
    model.chat.return_value = raw
    with pytest.raises(ValueError):
        review_replies(model, ['草稿'], {}, [0], [{'role': 'system', 'content': '依据'}])


def test_generation_capture_preserves_final_tool_material_and_restores_clients_on_error():
    client = model_with(dict(replies=['草稿']))
    clients = {'private': client, 'group': client}
    first = [dict(role='system', content='身份和人设')]
    final = first + [dict(role='user', content='Wiki 查询结果：本人在北京')]
    with pytest.raises(RuntimeError):
        with capture_generation(clients) as captured:
            clients['group'].chat(first, json_mode=True)
            clients['group'].chat(final, json_mode=True)
            final.append(dict(role='user', content='后续修改'))
            assert captured == final[:-1]
            assert clients['group'].model_name == client.model_name
            raise RuntimeError('generation failed')
    assert clients == {'private': client, 'group': client}


def test_context_prefetch_overlaps_generation_without_inheriting_generation_context():
    barrier = Barrier(2)
    marker = ContextVar('generation_validator', default='outside generation')
    context = dict(chat_type='group', context=[dict(text='问题')])
    material = [dict(role='system', content='原始依据')]
    review = model_with(None)
    def analyze(messages, **kwargs):
        assert marker.get() == 'outside generation'
        barrier.wait(timeout=2)
        return '{"events":[],"targets":[]}'
    review.chat.side_effect = analyze
    generation = model_with(None)
    generation.chat.side_effect = lambda *args, **kwargs: barrier.wait(timeout=2)
    clients = dict(group=generation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        prefetch = ContextPrefetch(pool, review, context, [0])
        token = marker.set('generation only')
        try:
            with capture_generation(clients, prefetch.start) as captured:
                clients['group'].chat(material)
        finally:
            marker.reset(token)
        assert prefetch.future.result()['analysis'] == dict(events=[], targets=[])
        review.chat.side_effect = lambda *args, **kwargs: '{"decision":"allow","issues":[]}'
        result = review_replies(review, ['回复'], context, [0], captured, prefetch=prefetch)
    assert result['context_review']['prefetched'] is True and result['parallel'] is False
    assert review.chat.call_count == 3


@pytest.mark.parametrize('changed', ['material', 'conversation', 'targets', 'client', 'prompt'])
def test_prefetch_is_discarded_when_any_review_input_changes(changed, tmp_path, monkeypatch):
    model = model_with(dict(decision='allow', issues=[]))
    context, indices = dict(context=[dict(text='原问题')]), [0]
    material = [dict(role='system', content='原始依据')]
    with ThreadPoolExecutor(max_workers=1) as pool:
        prefetch = ContextPrefetch(pool, model, context, indices)
        prefetch.start(material)
        prefetch.future.result()
        if changed == 'material':
            material.append(dict(role='user', content='新增 Wiki 结果'))
        elif changed == 'conversation':
            context['context'][0]['text'] = '新问题'
        elif changed == 'targets':
            indices.append(1)
        elif changed == 'client':
            model = model_with(dict(decision='allow', issues=[]))
        else:
            path = tmp_path / 'prompt.md'
            path.write_text('新的上下文分析提示词：独立梳理上下文')
            monkeypatch.setattr('src.reply.exposure_guard._CONTEXT_PROMPT', path)
        before = model.chat.call_count
        result = review_replies(model, ['回复'], context, indices, material, prefetch=prefetch)
    assert result['context_review']['prefetched'] is False
    assert model.chat.call_count - before == 3
    payload = json.loads(model.chat.call_args_list[before].args[0][1]['content'])
    assert payload == dict(conversation=context, target_indices=indices, generation_material=material)


@pytest.mark.parametrize('raw', [TimeoutError('prefetch failed'), 'invalid', '{"events":[]}'])
def test_prefetch_failure_cannot_release_a_draft(raw):
    model = model_with(None)
    model.chat.side_effect = raw if isinstance(raw, Exception) else None
    model.chat.return_value = raw
    with ThreadPoolExecutor(max_workers=1) as pool:
        prefetch = ContextPrefetch(pool, model, {}, [0])
        material = [dict(role='system', content='依据')]
        prefetch.start(material)
        with pytest.raises((ValueError, TimeoutError)):
            review_replies(model, ['草稿'], {}, [0], material, prefetch=prefetch)
    assert model.chat.call_count == 1


class ExposureRuntime:
    def __init__(self, instance, **config):
        self.binding = dict(instance=instance, generator='g-test', data='d-test',
                            config_sha256='a' * 64, sources_sha256='b' * 64)
        self.replies = ['好的', '已经转好了']
        self.model = model_with(dict(decision='block', issues=[issue(reply_index=1)]))
        self.generator = SimpleNamespace(_llm={'private': self.model, 'group': self.model})
        self.closed = False

    def draft(self, request):
        value = request['input']
        if self.replies:
            self.generator._llm[value['chat_type']].chat([
                dict(role='system', content='本人是小钱，尚未进行转账'),
                dict(role='user', content=json.dumps(value, ensure_ascii=False))], json_mode=True)
        return dict(request_id=request['request_id'], binding=dict(self.binding), draft_only=True, sent=False,
                    status='draft' if self.replies else 'silence', replies=list(self.replies), input_sha256='c' * 64,
                    input_payload_sha256=input_digest(value))

    def close(self):
        self.closed = True


def message(**kwargs):
    return replace(ChatMessage('请帮我转账', 'friend', SenderType.OTHER, 'example',
                               chatroom_id='source-test', create_time=100, local_id=17, server_id='19'), **kwargs)


@pytest.fixture
def generator(tmp_path, monkeypatch):
    runtime = ExposureRuntime('test')
    monkeypatch.setattr('src.reply.digital_human._load_runtime', lambda repository: lambda *args, **kwargs: runtime)
    generator = DigitalHumanReplyGenerator(tmp_path, 'test')
    yield generator, runtime
    generator.close()


def test_generator_blocks_entire_turn_from_actual_model_review(generator):
    generator, runtime = generator
    target = message()
    assert generator.generate([target], [target], tick_id=7) == []
    assert generator.last_generation_failed is False
    assert generator.last_exposure_block['rule'] == 'hallucination'
    assert generator.last_exposure_block['replies'] == runtime.replies
    review = generator.last_generation_trace[-1]
    assert review['type'] == 'reply_review' and review['decision'] == 'block'
    assert review['model'] == 'test-review-model'
    assert runtime.model.chat.call_count == 3
    payload = json.loads(runtime.model.chat.call_args.args[0][1]['content'])
    assert payload['target_indices'] == [0]
    assert payload['generation_material'][0]['content'] == '本人是小钱，尚未进行转账'


@pytest.mark.parametrize('fault', ['timeout', 'invalid', 'missing_evidence'])
def test_review_failure_keeps_unread_and_draft_trace_without_closing_runtime(generator, fault):
    generator, runtime = generator
    if fault == 'timeout':
        runtime.model.chat.side_effect = ['draft', TimeoutError('review timed out')]
    elif fault == 'invalid':
        runtime.model.chat.side_effect = None
        runtime.model.chat.return_value = '{"decision":"allow","issues":[{}]}'
    else:
        original_draft = runtime.draft
        def without_evidence(request):
            clients = runtime.generator._llm
            runtime.generator._llm = {'private': runtime.model, 'group': runtime.model}
            try:
                return original_draft(request)
            finally:
                runtime.generator._llm = clients
        runtime.draft = without_evidence
    target = message()
    assert generator.generate([target], [target]) == []
    assert generator.last_generation_failed is True and generator.last_exposure_block is None
    assert generator.last_generation_trace[-1]['type'] == 'reply_review_error'
    assert json.loads(generator.last_raw_response)['replies'] == runtime.replies
    assert not runtime.closed


def test_silence_skips_model_review(generator):
    generator, runtime = generator
    runtime.replies = []
    target = message()
    assert generator.generate([target], [target]) == []
    runtime.model.chat.assert_not_called()
    assert generator.last_generation_failed is False and generator.last_exposure_block is None


def test_targets_are_relative_to_truncated_context_and_later_observed_self_is_preserved(generator):
    generator, runtime = generator
    runtime.model.chat.return_value = '{"decision":"allow","issues":[]}'
    runtime.model.chat.side_effect = lambda messages, **kw: ('{"events":[],"targets":[]}'
        if '独立梳理上下文' in messages[0]['content'] else '{"decision":"allow","issues":[]}')
    prior = [message(text=f'旧消息{i}', observed_at=i + 1) for i in range(203)]
    target = message(observed_at=204)
    later = message(text='已经回复过了', sender_type=SenderType.SELF, observed_at=205)
    assert generator.generate([target], prior + [target, later]) == runtime.replies
    payload = json.loads(runtime.model.chat.call_args.args[0][1]['content'])
    assert len(payload['conversation']['context']) == 200 and payload['target_indices'] == [198]
    assert payload['conversation']['context'][-1]['is_self'] is True
    assert payload['conversation']['context'][-1]['text'] == '已经回复过了'


@pytest.mark.parametrize('chat_type', ['private', 'group'])
def test_selected_review_client_preserves_generation_route_and_evidence(tmp_path, monkeypatch, chat_type):
    runtime = ExposureRuntime('test')
    private = model_with(dict(replies=runtime.replies))
    group = model_with(dict(decision='block', issues=[issue(reply_index=1)]))
    runtime.generator._llm = {'private': private, 'group': group}
    monkeypatch.setattr('src.reply.digital_human._load_runtime', lambda repository: lambda *args, **kwargs: runtime)
    config_path = tmp_path / 'digital_human.json'
    config_path.write_text(json.dumps(dict(repository=str(tmp_path), instance='test',
                                         timeout_seconds=900, review_client='group')))
    with_context = load_digital_human_config(config_path)
    generator = DigitalHumanReplyGenerator(**with_context)
    target = message()
    try:
        assert generator.generate([target], [target], is_group=chat_type == 'group') == []
        assert not generator.last_generation_failed
        assert generator.last_exposure_block['rule'] == 'hallucination'
        assert private.chat.call_count == (1 if chat_type == 'private' else 0)
        assert group.chat.call_count == (2 if chat_type == 'private' else 3)
        generation_call = runtime.generator._llm[chat_type].chat.call_args_list[0]
        payload = json.loads(group.chat.call_args.args[0][1]['content'])
        assert payload['generation_material'] == generation_call.args[0]
        assert payload['conversation']['chat_type'] == chat_type
        assert generator.runtime_config['review_client'] == 'group'
    finally:
        generator.close()


@pytest.mark.parametrize('review_client', ['', 'unknown', True, [], {}])
def test_invalid_review_client_is_rejected_before_runtime_load(tmp_path, monkeypatch, review_client):
    load = Mock()
    monkeypatch.setattr('src.reply.digital_human._load_runtime', load)
    config_path = tmp_path / 'digital_human.json'
    config_path.write_text(json.dumps(dict(repository=str(tmp_path), instance='test',
                                         timeout_seconds=900, review_client=review_client)))
    with pytest.raises(DigitalHumanError, match='审查客户端'):
        DigitalHumanReplyGenerator(**load_digital_human_config(config_path))
    load.assert_not_called()


def test_issue_quote_with_whitespace_difference_is_kept():
    """引用与草稿仅有空白差异时不得判为无效依据。"""
    verdict = dict(decision='block', issues=[issue(quote=' 已经转好了 ')])
    model = model_with(verdict)
    result = review_replies(model, ['已经转好了'], {'context': []}, [0], [{'role': 'system', 'content': '资料'}])
    assert result['decision'] == 'block' and len(result['issues']) == 1


def test_issue_quote_mismatch_still_fails_closed():
    """引用真实不符时维持 fail-closed：整个判定被拒绝。"""
    verdict = dict(decision='block', issues=[issue(quote='草稿里根本没有这句话')])
    model = model_with(verdict)
    with pytest.raises(ValueError):
        review_replies(model, ['已经转好了'], {'context': []}, [0], [{'role': 'system', 'content': '资料'}])
