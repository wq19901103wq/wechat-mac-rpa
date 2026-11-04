import hashlib
import json
import importlib.util
import io
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.models.base import ChatMessage, PerceptionResult, SenderType
from src.reply.digital_human import (
    DigitalHumanDraftClient, DigitalHumanError, DigitalHumanReplyGenerator,
    input_digest, load_digital_human_config, messages_input,
)


class FakeRuntime:
    def __init__(self, instance, **config):
        self.binding = dict(instance=instance, generator="g-test", data="d-test",
                            config_sha256="a" * 64, sources_sha256="b" * 64)
        self.config, self.closed = config, False
        model = Mock(model_name='review-test')
        model.chat.side_effect = lambda messages, **kwargs: (
            '{"events":[],"targets":[]}' if '独立梳理上下文' in messages[0]['content']
            else '{"decision":"allow","issues":[]}'
        )
        self.generator = SimpleNamespace(_llm={'private': model, 'group': model})
        if instance == "bad_ready":
            self.binding["instance"] = "wrong"

    def draft(self, request):
        assert not self.closed
        value = request['input']
        fault = value.get('fault')
        if fault == 'exception':
            raise RuntimeError('unavailable')
        replies = [] if fault == 'silence' else ['synthetic draft']
        if replies:
            self.generator._llm[value.get('chat_type', 'private')].chat([
                dict(role='system', content='Test persona and verified evidence'),
                dict(role='user', content=json.dumps(value))], json_mode=True)
        response = dict(request_id=request['request_id'], binding=dict(self.binding), draft_only=True, sent=False,
            status='draft' if replies else 'silence', replies=replies, input_sha256='c'*64,
            input_payload_sha256=input_digest(value), pid=os.getpid())
        changes = {'binding': ('binding', dict(self.binding, generator='wrong')),
            'request_id': ('request_id', 'wrong'), 'digest': ('input_payload_sha256', 'd'*64),
            'case_hash': ('input_sha256', 'z'*64), 'sent': ('sent', True), 'draft_only': ('draft_only', False),
            'empty_reply': ('replies', ['']), 'status': ('status', 'silence'), 'failure': ('status', 'error')}
        if fault in changes:
            key, changed = changes[fault]
            response[key] = changed
        return [] if fault == 'array' else response

    def close(self):
        self.closed = True


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr('src.reply.digital_human._load_runtime', lambda repository: FakeRuntime)
    return tmp_path


def message(**kwargs):
    return replace(ChatMessage('hello', 'friend', SenderType.OTHER, 'example',
                               chatroom_id='source-test', create_time=100, local_id=17, server_id='19'), **kwargs)


def test_messages_preserve_original_time_and_explicit_roles():
    value = messages_input([message(), message(sender='self', sender_type=SenderType.SELF, create_time=101)],
                           source_chat_id='source-test', is_group=True)
    assert value['chat_type'] == 'group'
    assert value['context'] == [dict(text='hello', sender='friend', is_self=False, timestamp=100),
                                dict(text='hello', sender='self', is_self=True, timestamp=101)]
    assert all('message_id' not in row for row in value['context'])


@pytest.mark.parametrize('is_group', [False, True])
def test_screenshot_adapter_uses_observation_time_without_source_ids(endpoint, is_group):
    own = message(sender='自己', sender_type=SenderType.SELF, chatroom_id=None,
                  create_time=None, observed_at=150)
    target = message(chatroom_id=None, create_time=None, observed_at=150)
    later = message(text='我下来了，马上到楼下', sender_type=SenderType.SELF, chatroom_id=None,
                    create_time=None, observed_at=160)
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    try:
        assert generator.generate([target], [own, target, later], is_group=is_group) == ['synthetic draft']
        value = json.loads(generator.last_user_prompt)
        assert value['kind'] == 'observed'
        assert 'source_chat_id' not in value
        assert value['context'] == [
            dict(text='hello', sender='自己', is_self=True, timestamp=150, time_basis='first_observed'),
            dict(text='hello', sender='friend', is_self=False, timestamp=150, time_basis='first_observed'),
            dict(text='我下来了，马上到楼下', sender='friend', is_self=True, timestamp=160,
                 time_basis='first_observed')]
    finally:
        generator.close()


@pytest.mark.parametrize('kwargs', [dict(sender_type=SenderType.UNKNOWN), dict(sender_type=SenderType.SYSTEM),
    dict(create_time=None), dict(create_time=True), dict(create_time=0), dict(create_time=float('nan')),
    dict(message_type='unknown'), dict(chatroom_id='other'), dict(text=' '), dict(sender='')])
def test_unsupported_message_boundaries_are_rejected(kwargs):
    with pytest.raises(DigitalHumanError):
        messages_input([message(**kwargs)], source_chat_id='source-test', is_group=False)


@pytest.mark.parametrize('message_type,label', [('image', '图片'), ('sticker', '表情包'),
    ('mixed', '图文'), ('link_card', '链接卡片'), ('video', '视频')])
def test_media_adapter_preserves_recognized_content_and_quotes(message_type, label):
    target = message(message_type=message_type, text='正文', image_description='画面描述',
                     image_text='画面文字', quoted_text='被引用的原话', observed_at=150)
    value = messages_input([target], is_group=False)
    assert value['context'] == [dict(
        text=f'「引用：被引用的原话」\n[{label}]\n描述：画面描述\n正文\n识别文字：画面文字',
        sender='friend', is_self=False, timestamp=150, time_basis='first_observed')]
    assert target.text == '正文' and target.quoted_text == '被引用的原话'
    target.text = target.image_text = '相同文字'
    assert messages_input([target], is_group=False)['context'][0]['text'].count('相同文字') == 1


@pytest.mark.parametrize('is_group', [False, True])
def test_bot_adapter_generates_with_media_and_quoted_reply(endpoint, is_group):
    media = message(text='', message_type='image', observed_at=150)
    target = message(text='明天带给你', quoted_text='这个给我留着', observed_at=151)
    later = message(text='已经发送的回复', sender_type=SenderType.SELF, observed_at=152)
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    try:
        assert generator.generate([target], [media, target, later], is_group=is_group) == ['synthetic draft']
        value = json.loads(generator.last_user_prompt)
        assert [row['text'] for row in value['context']] == [
            '[图片]', '「引用：这个给我留着」\n明天带给你', '已经发送的回复']
        assert [row['timestamp'] for row in value['context']] == [150, 151, 152]
        assert not generator.last_generation_failed
        assert json.loads(generator.last_raw_response)['sent'] is False
    finally:
        generator.close()


def test_recorded_replay_excludes_future_reply(endpoint):
    target = message(text='到了吗', create_time=100)
    later = message(text='我下来了，马上到楼下', sender_type=SenderType.SELF, create_time=101)
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    try:
        assert generator.generate([target], [target, later]) == ['synthetic draft']
        value = json.loads(generator.last_user_prompt)
        assert value['kind'] == 'messages'
        assert [row['text'] for row in value['context']] == ['到了吗']
    finally:
        generator.close()


@pytest.mark.parametrize('observed', [False, True])
def test_old_context_is_excluded_relative_to_target_time(endpoint, observed):
    def timed(text, timestamp, **kwargs):
        return message(text=text, **({'observed_at': timestamp} if observed else {'create_time': timestamp}), **kwargs)
    target = timed('突然忙起来了', 10_000_000)
    history = [timed('几个月前的玩笑', 1_000_000), timed('昨天太早的消息', 9_913_599),
               timed('24小时边界', 9_913_600), timed('近期话题', 9_999_900), target,
               timed('已回复', 10_000_001, sender_type=SenderType.SELF)]
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    try:
        assert generator.generate([target], history) == ['synthetic draft']
        value = json.loads(generator.last_user_prompt)
        assert [row['text'] for row in value['context']] == (
            ['24小时边界', '近期话题', '突然忙起来了'] + (['已回复'] if observed else []))
        review = generator._client._runtime.generator._llm['private'].chat.call_args.args[0]
        assert json.loads(review[1]['content'])['target_indices'] == [2]
        assert value['context'][-1].get('time_basis') == ('first_observed' if observed else None)
        assert len(history) == 6
    finally:
        generator.close()


def test_time_window_preserves_all_targets_and_later_observed_self(endpoint):
    old = message(text='无关旧消息', create_time=1_000_000)
    first = message(text='第一条待回复', observed_at=10_000_000)
    second = message(text='第二条待回复', observed_at=10_100_000)
    later = message(text='已经回复', sender_type=SenderType.SELF, observed_at=10_100_001)
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    try:
        assert generator.generate([first, second], [old, first, second, later]) == ['synthetic draft']
        value = json.loads(generator.last_user_prompt)
        assert [row['text'] for row in value['context']] == ['第一条待回复', '第二条待回复', '已经回复']
        review = generator._client._runtime.generator._llm['private'].chat.call_args.args[0]
        assert json.loads(review[1]['content'])['target_indices'] == [0, 1]
        assert value['context'][-1]['is_self'] is True
    finally:
        generator.close()


def test_observed_target_outside_latest_context_window_is_rejected(endpoint):
    target = message(observed_at=150)
    history = [target] + [message(sender_type=SenderType.SELF, observed_at=151 + i) for i in range(200)]
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    try:
        assert generator.generate([target], history) == []
        assert generator.last_generation_failed
        assert '200 条上下文限制' in json.loads(generator.last_raw_response)['error']
    finally:
        generator.close()


def test_mixed_chats_and_empty_inputs_are_rejected():
    with pytest.raises(DigitalHumanError):
        messages_input([message(), message(chat_name='different')], source_chat_id='source-test', is_group=False)
    with pytest.raises(DigitalHumanError):
        messages_input([], source_chat_id='source-test', is_group=False)


def test_client_environment_and_multiple_drafts(endpoint, monkeypatch):
    monkeypatch.setenv('DH_INSTANCE', 'must-not-inherit')
    environment = dict(os.environ)
    with DigitalHumanDraftClient(endpoint, 'test') as client:
        runtime = client._runtime
        first = client.draft({'text': 'first'})
        second = client.draft({'text': 'second', 'fault': 'silence'})
        assert first['replies'] == ['synthetic draft'] and first['sent'] is False
        assert second['status'] == 'silence' and second['replies'] == []
        assert runtime is client._runtime and not runtime.closed
        assert first['pid'] == os.getpid()
        assert runtime.config['instances_root'] == endpoint / 'instances'
        assert first['rpa_timing_ms']['startup'] == second['rpa_timing_ms']['startup'] == client.startup_ms
        assert all(result['rpa_timing_ms']['draft'] > 0 for result in (first, second))
    assert runtime.closed and os.environ == environment


def test_stream_cli_reuses_runtime_and_stops_on_invalid_input(endpoint, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('draft_cli', root / 'scripts/digital_human_draft.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = [{'text': 'first'}, {'text': 'second', 'fault': 'silence'}]
    monkeypatch.setattr(sys, 'argv', ['digital_human_draft.py', '--repository', str(endpoint),
                                    '--instance', 'test', '--stream'])
    monkeypatch.setattr(sys, 'stdin', io.StringIO('\n'.join(
        [*(json.dumps(value) for value in values), 'invalid', json.dumps(values[0])]) + '\n'))
    assert module.main() == 1
    captured = capsys.readouterr()
    responses = [json.loads(line) for line in captured.out.splitlines()]
    assert len(responses) == 2
    assert responses[0]['pid'] == responses[1]['pid'] == os.getpid()
    assert [response['status'] for response in responses] == ['draft', 'silence']
    assert all(response['sent'] is False for response in responses)
    assert '初始化' in captured.err and '未发送' in captured.err


@pytest.mark.parametrize('is_group', [False, True])
def test_screenshot_cli_saves_observed_input_and_outputs_only_draft_json(endpoint, monkeypatch, capsys, is_group):
    from scripts import digital_human_draft as cli

    target = message(chatroom_id=None, create_time=None, observed_at=150,
                     message_type='image', text='', image_description='一只猫', quoted_text='给我看看')
    def perceive():
        print('perception diagnostics')
        return PerceptionResult('example', [target], [], '', is_group=is_group)
    pipeline = Mock()
    pipeline.perceive.side_effect = perceive
    monkeypatch.setattr('src.perception.smart_pipeline.SmartPerceptionPipeline', Mock(return_value=pipeline))
    saved = endpoint / 'observed.json'
    monkeypatch.setattr(sys, 'argv', ['digital_human_draft.py', '--repository', str(endpoint),
        '--instance', 'test', '--screenshot', '--save-input', str(saved)])

    assert cli.main() == 0
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    value = json.loads(saved.read_text())
    assert response['replies'] == ['synthetic draft'] and response['sent'] is False
    assert response['input_payload_sha256'] == input_digest(value)
    assert value == dict(kind='observed', chat_type='group' if is_group else 'private',
        chat_name='example', context=[dict(text='「引用：给我看看」\n[图片]\n描述：一只猫', sender='friend', is_self=False,
                                         timestamp=150, time_basis='first_observed')])
    assert 'perception diagnostics' in captured.err
    pipeline.perceive.assert_called_once_with()


@pytest.mark.parametrize('error_name', ['CaptureValidationError', 'WeChatNotReadyError', 'WindowNotFoundError'])
def test_screenshot_cli_capture_failure_does_not_start_dh(endpoint, monkeypatch, capsys, error_name):
    from scripts import digital_human_draft as cli
    from src.capture import window_capture

    pipeline = Mock()
    pipeline.perceive.side_effect = getattr(window_capture, error_name)('微信窗口未就绪')
    monkeypatch.setattr('src.perception.smart_pipeline.SmartPerceptionPipeline', Mock(return_value=pipeline))
    client = Mock()
    monkeypatch.setattr(cli, 'DigitalHumanDraftClient', client)
    saved = endpoint / 'observed.json'
    monkeypatch.setattr(sys, 'argv', ['digital_human_draft.py', '--repository', str(endpoint),
        '--instance', 'test', '--screenshot', '--save-input', str(saved)])

    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ''
    assert '草稿失败（未发送）：微信窗口未就绪' in captured.err
    assert not saved.exists()
    client.assert_not_called()


@pytest.mark.parametrize('fault', ['binding', 'request_id', 'digest', 'case_hash', 'sent', 'draft_only',
                                 'empty_reply', 'status', 'failure', 'array', 'exception'])
def test_invalid_results_close_runtime_without_fallback(endpoint, fault):
    client = DigitalHumanDraftClient(endpoint, 'test')
    runtime = client._runtime
    with pytest.raises(DigitalHumanError):
        client.draft({'text': 'hello', 'fault': fault})
    assert client._runtime is None and runtime.closed
    with pytest.raises(DigitalHumanError, match='关闭'):
        client.draft({'text': 'hello'})


def test_input_limit(endpoint):
    client = DigitalHumanDraftClient(endpoint, 'test')
    with pytest.raises(DigitalHumanError, match='输入超过'):
        client.draft({'text': 'x' * (4 * 1024 * 1024)})
    assert client._runtime is None


def test_invalid_binding_and_timeout(endpoint):
    with pytest.raises(DigitalHumanError, match='初始化失败'):
        DigitalHumanDraftClient(endpoint, 'bad_ready')
    with pytest.raises(DigitalHumanError):
        DigitalHumanDraftClient(endpoint, 'test', timeout=float('inf'))


def test_bot_adapter_initialization_failure_is_reported_at_construction(endpoint):
    with pytest.raises(DigitalHumanError, match='初始化失败'):
        DigitalHumanReplyGenerator(endpoint, 'bad_ready')


@pytest.fixture
def worker_repository(tmp_path):
    package = tmp_path / 'src/generator'
    package.mkdir(parents=True)
    (tmp_path / 'src/__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (tmp_path / '.env').write_text('DH_SETTINGS_FILE=worker-settings\n')
    (package / 'draft.py').write_text('''
import hashlib
import json
import os
import time
from threading import Barrier, Event

class Client:
    model_name = 'worker-review'

    def __init__(self):
        self.mode, self.calls = '', []
        self.barrier = Barrier(2)
        self.context_started, self.release = Event(), Event()
        self.contexts = []

    def chat(self, messages, **kwargs):
        prompt = messages[0]['content']
        if prompt == 'inspect':
            return json.dumps(dict(calls=self.calls, pid=os.getpid(),
                contexts=self.contexts,
                instance=os.environ.get('DH_INSTANCE'), settings=os.environ.get('DH_SETTINGS_FILE')))
        self.calls.append(prompt)
        if prompt == 'actual generation evidence':
            if self.mode == 'concurrent':
                self.barrier.wait(timeout=2)
            if self.mode == 'silence_wait':
                assert self.context_started.wait(timeout=2)
            return 'synthetic draft'
        payload = json.loads(messages[1]['content'])
        assert payload['generation_material'][0]['content'] == 'actual generation evidence'
        assert 'decision' not in payload
        if '独立梳理上下文' in prompt:
            assert 'replies' not in payload
            self.contexts.append(payload)
            if self.mode == 'context_error':
                raise TimeoutError('context failed')
            if self.mode == 'silence_wait':
                self.context_started.set()
                assert self.release.wait(timeout=2)
            if self.mode == 'concurrent':
                self.barrier.wait(timeout=2)
            return '{"events":[],"targets":[]}'
        assert payload['context_analysis'] == dict(events=[], targets=[])
        if self.mode == 'review_error':
            raise RuntimeError('review failed')
        if self.mode == 'first_block' or (self.mode == 'verify_block' and '独立质检员' in prompt):
            return json.dumps(dict(decision='block', issues=[dict(category='hallucination',
                reply_index=0, reason='test evidence', quote='synthetic')]))
        return '{"decision":"allow","issues":[]}'

class DraftRuntime:
    def __init__(self, instance):
        from types import SimpleNamespace
        self.binding = dict(instance=instance, generator='g-test', data='d-test',
            config_sha256='a'*64, sources_sha256='b'*64)
        client = Client()
        self.client = client
        self.generator = SimpleNamespace(_llm=dict(private=client, group=client))

    def draft(self, request):
        value = request['input']
        mode = value.get('text', value.get('context', [{}])[-1].get('text', ''))
        if mode == 'exit':
            os._exit(1)
        if mode == 'timeout':
            time.sleep(10)
        client = self.generator._llm['private']
        self.client.mode = mode
        if mode == 'next_after_silence':
            self.client.release.set()
        client.chat([dict(role='system', content='actual generation evidence'),
                     dict(role='user', content=json.dumps(value))])
        if mode == 'tool_changes':
            client.chat([dict(role='system', content='actual generation evidence'),
                         dict(role='user', content='final Wiki result')])
        replies = [] if mode == 'silence_wait' else ['synthetic draft']
        return dict(request_id=request['request_id'], binding=self.binding, draft_only=True,
            sent=False, status='draft' if replies else 'silence', replies=replies, input_sha256='c'*64,
            input_payload_sha256=hashlib.sha256(json.dumps(value, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(), pid=os.getpid())

    def close(self):
        pass
''')
    return tmp_path


def test_shared_worker_isolates_imports_environment_and_reuses_runtime(worker_repository, monkeypatch):
    from src import models
    original_src = sys.modules['src']
    monkeypatch.setenv('DH_INSTANCE', 'host-instance')
    monkeypatch.setenv('DH_SETTINGS_FILE', 'host-settings')
    environment = dict(os.environ)
    with DigitalHumanDraftClient(worker_repository, 'test', timeout=10) as client:
        runtime = client._runtime
        process = runtime._process
        assert Path(runtime.source) == worker_repository / 'src/generator/draft.py'
        first = client.draft(dict(text='first', chat_type='private'))
        second = client.draft(dict(text='second', chat_type='private'))
        assert first['pid'] == second['pid'] == process.pid != os.getpid()
        assert client.review(second['replies'], dict(chat_type='private'), [0])['decision'] == 'allow'
        state = json.loads(runtime.generator._llm['private'].chat([dict(content='inspect')]))
        assert state['instance'] == 'test' and state['settings'] == 'worker-settings'
        assert state['calls'].count('actual generation evidence') == 2
        assert len(state['calls']) == 5
        assert sys.modules['src'] is original_src and sys.modules['src.models'] is models
        assert os.environ == environment
    assert runtime._process is None and process.poll() is not None
    from src.reply.digital_human import _load_runtime
    with pytest.raises(DigitalHumanError, match='缺失'):
        _load_runtime(worker_repository / 'absent')


@pytest.mark.parametrize('mode,blocked,failed,review_calls', [
    ('allow', False, False, 3), ('first_block', True, False, 2),
    ('verify_block', True, False, 3), ('review_error', False, True, 2)])
def test_shared_worker_preserves_review_decisions(worker_repository, mode, blocked, failed, review_calls):
    generator = DigitalHumanReplyGenerator(worker_repository, 'test', timeout=10)
    try:
        target = message(text=mode)
        replies = generator.generate([target], [target])
        assert replies == ([] if blocked or failed else ['synthetic draft'])
        assert bool(generator.last_exposure_block) is blocked
        assert generator.last_generation_failed is failed
        if not failed:
            assert generator.last_generation_trace[-1]['context_review']['prefetched'] is False
        state = json.loads(generator._client._runtime.generator._llm['private'].chat([dict(content='inspect')]))
        assert len(state['calls']) == 1 + review_calls
    finally:
        generator.close()


@pytest.fixture
def prefetch_generator(worker_repository, monkeypatch):
    generator = DigitalHumanReplyGenerator(worker_repository, 'test', timeout=10)
    draft = generator._client.draft
    monkeypatch.setattr(generator._client, 'draft', lambda online_input: draft(online_input, target_indices=[0]))
    yield generator
    generator.close()


def test_shared_worker_reanalyzes_final_wiki_material(prefetch_generator):
    generator = prefetch_generator
    try:
        target = message(text='tool_changes')
        assert generator.generate([target], [target]) == ['synthetic draft']
        review = generator.last_generation_trace[-1]
        assert review['context_review']['prefetched'] is False
        state = json.loads(generator._client._runtime.generator._llm['private'].chat([dict(content='inspect')]))
        assert len(state['contexts']) == 2
        assert state['contexts'][-1]['generation_material'][-1]['content'] == 'final Wiki result'
    finally:
        generator.close()


def test_shared_worker_prefetch_error_keeps_draft_unsent(prefetch_generator):
    generator = prefetch_generator
    try:
        target = message(text='context_error')
        assert generator.generate([target], [target]) == []
        assert generator.last_generation_failed
        assert generator.last_generation_trace[-1]['type'] == 'reply_review_error'
        target = message(text='allow')
        assert generator.generate([target], [target]) == ['synthetic draft']
        assert generator.last_generation_trace[-1]['context_review']['prefetched'] is True
    finally:
        generator.close()


def test_shared_worker_does_not_wait_for_silent_draft_or_reuse_it_next_turn(prefetch_generator):
    generator = prefetch_generator
    try:
        target = message(text='silence_wait')
        assert generator.generate([target], [target]) == []
        assert not generator.last_generation_failed
        target = message(text='next_after_silence')
        assert generator.generate([target], [target]) == ['synthetic draft']
        assert generator.last_generation_trace[-1]['context_review']['prefetched'] is False
        target = message(text='allow')
        assert generator.generate([target], [target]) == ['synthetic draft']
        assert generator.last_generation_trace[-1]['context_review']['prefetched'] is True
    finally:
        generator.close()


@pytest.mark.parametrize('fault', ['exit', 'timeout'])
def test_shared_worker_failure_closes_process_without_fallback(worker_repository, fault):
    client = DigitalHumanDraftClient(worker_repository, 'test', timeout=10)
    runtime = client._runtime
    process = runtime._process
    runtime.timeout = 0.05
    with pytest.raises(DigitalHumanError):
        client.draft(dict(text=fault, chat_type='private'))
    assert client._runtime is None and runtime._process is None and process.poll() is not None
    with pytest.raises(DigitalHumanError, match='关闭'):
        client.draft(dict(text='after', chat_type='private'))


def test_digest_is_canonical_and_matches_dh_encoding():
    assert input_digest({'text': '中文', 'kind': 'composed'}) == input_digest({'kind': 'composed', 'text': '中文'})
    assert input_digest({'text': '中文'}) == hashlib.sha256(
        json.dumps({'text': '中文'}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@pytest.mark.parametrize('is_group', [False, True])
def test_bot_adapter_reuses_runtime_and_cuts_off_later_reply(endpoint, is_group):
    generator = DigitalHumanReplyGenerator(endpoint, 'test', timeout=2)
    client = generator._client
    assert client is not None
    runtime = client._runtime
    prior = message(sender='self', sender_type=SenderType.SELF, create_time=99)
    target = message()
    later = message(text='真人后续回复', sender_type=SenderType.SELF, create_time=101)
    try:
        assert generator.generate([target], [prior, target, later], is_group=is_group, tick_id=7) == ['synthetic draft']
        assert generator._client is client and client._runtime is runtime
        first = json.loads(generator.last_raw_response)
        assert generator.generate([target], [prior, target], is_group=is_group) == ['synthetic draft']
        assert generator._client is client and client._runtime is runtime
        assert first['pid'] == json.loads(generator.last_raw_response)['pid']
        online_input = json.loads(generator.last_user_prompt)
        assert online_input == messages_input([prior, target], source_chat_id='source-test', is_group=is_group)
        assert generator.last_generation_failed is False
        assert generator.last_active_llm == 'digital-human:test/g-test/d-test'
        assert generator.last_generation_trace[0]['capture_scope'] == 'draft_protocol'
    finally:
        generator.close()
    assert generator._client is None and runtime.closed


@pytest.mark.parametrize('boundary', ['missing_id', 'synthetic_id', 'mixed_ids', 'missing_target', 'too_many_targets',
                                     'missing_time', 'unknown_role', 'unknown_type'])
def test_bot_adapter_rejects_unsupported_input_without_drafting(endpoint, monkeypatch, boundary):
    target = message()
    context = [target]
    if boundary == 'missing_id':
        target.chatroom_id = None
    elif boundary == 'synthetic_id':
        target.chatroom_id = '_chats_example'
    elif boundary == 'mixed_ids':
        context.insert(0, message(chatroom_id='another-chat'))
    elif boundary == 'missing_target':
        context = [message()]
    elif boundary == 'too_many_targets':
        context.extend(message(create_time=101+i) for i in range(200))
    elif boundary == 'missing_time':
        target.create_time = None
    elif boundary == 'unknown_role':
        target.sender_type = SenderType.UNKNOWN
    else:
        target.message_type = 'unknown'
    factory = Mock()
    monkeypatch.setattr('src.reply.digital_human.DigitalHumanDraftClient', factory)
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    targets = [target, context[-1]] if boundary == 'too_many_targets' else [target]
    assert generator.generate(targets, context) == []
    assert generator.last_generation_failed is True
    factory.assert_called_once()
    factory.return_value.draft.assert_not_called()


def test_bot_adapter_closes_failed_client_and_only_retries_dh(endpoint, monkeypatch):
    failed = Mock()
    failed.draft.side_effect = DigitalHumanError('unavailable')
    factory = Mock(side_effect=[failed, DigitalHumanDraftClient(endpoint, 'test')])
    monkeypatch.setattr('src.reply.digital_human.DigitalHumanDraftClient', factory)
    generator = DigitalHumanReplyGenerator(endpoint, 'test')
    target = message()
    assert generator.generate([target], [target]) == []
    assert generator.last_generation_failed is True and generator._client is None
    failed.close.assert_called_once()
    try:
        assert generator.generate([target], [target]) == ['synthetic draft']
        assert factory.call_count == 2
        assert generator.last_generation_failed is False
    finally:
        generator.close()


@pytest.mark.parametrize('timeout', [0, -1, True, float('inf')])
def test_invalid_bot_config_is_rejected(tmp_path, timeout):
    path = tmp_path / 'digital_human.json'
    path.write_text(json.dumps(dict(repository='../digital-human', instance='test', timeout_seconds=timeout)))
    with pytest.raises(DigitalHumanError):
        load_digital_human_config(path)


def test_missing_bot_config_explains_local_setup(tmp_path):
    with pytest.raises(DigitalHumanError, match='digital_human.json.example'):
        load_digital_human_config(tmp_path / 'missing.json')


def test_draft_cli_help_needs_no_local_config(tmp_path, monkeypatch, capsys):
    from scripts import digital_human_draft as cli
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'load_digital_human_config', Mock(side_effect=AssertionError('must not load')))
    monkeypatch.setattr(sys, 'argv', ['digital_human_draft.py', '--help'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert '--instance' in capsys.readouterr().out


def test_draft_cli_without_config_requires_instance(tmp_path, monkeypatch, capsys):
    from scripts import digital_human_draft as cli
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['digital_human_draft.py', '--stream'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert '需要 --instance' in capsys.readouterr().err


def test_draft_cli_explicit_options_work_without_local_config(endpoint, monkeypatch, capsys):
    from scripts import digital_human_draft as cli
    monkeypatch.setattr(cli, 'ROOT', endpoint)
    monkeypatch.setattr(sys, 'argv', ['digital_human_draft.py', '--repository', str(endpoint),
        '--instance', 'test', '--text', 'hello', '--chat-type', 'private',
        '--chat-name', 'example', '--source-chat-id', 'source-test'])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['replies'] == ['synthetic draft'] and result['sent'] is False


def test_shared_worker_starts_without_optional_profiling_modules(worker_repository):
    package = worker_repository / 'src/generator'
    (package / 'generator.py').write_text('')
    with (package / 'draft.py').open('a') as output:
        output.write('\nimport src.generator.generator\n')
    with DigitalHumanDraftClient(worker_repository, 'test', timeout=10) as client:
        assert client.draft(dict(text='hello', chat_type='private'))['replies'] == ['synthetic draft']


@pytest.mark.parametrize('timestamp', ['', 'invalid', '2026-10-22 10:20:30'])
def test_ocr_preserves_missing_or_original_time(timestamp):
    from datetime import datetime
    from src.perception.smart_pipeline import SmartPerceptionPipeline

    pipeline = object.__new__(SmartPerceptionPipeline)
    converted = pipeline._convert_api_messages([dict(sender='对方', text='hello', timestamp=timestamp)], 'example')
    expected = int(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').timestamp()) if timestamp.startswith('2026') else None
    assert converted[0].create_time == expected
