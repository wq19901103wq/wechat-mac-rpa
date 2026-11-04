import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.models.base import ChatMessage, SenderType
from src.reply.digital_human import DigitalHumanError, DigitalHumanReplyGenerator, input_digest, load_digital_human_config
from src.reply.exposure_guard import ContextPrefetch, capture_generation, review_replies


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
        load_digital_human_config(config_path)
    with pytest.raises(DigitalHumanError, match='审查客户端'):
        DigitalHumanReplyGenerator(tmp_path, 'test', review_client=review_client)
    load.assert_not_called()
