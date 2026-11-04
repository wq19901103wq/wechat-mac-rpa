from unittest.mock import Mock
from pathlib import Path
import runpy

import pytest

import run_bot
from src.bot.wechat_bot import WeChatBot
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.models.base import ActionResult, ChatListItem, ChatMessage, PerceptionResult, Rect, SenderType
from src.perception.smart_pipeline import SmartPerceptionPipeline
from src.reply.digital_human import DigitalHumanError
from src.session.global_store import GlobalStore


@pytest.fixture(autouse=True)
def local_config(monkeypatch, tmp_path):
    config = dict(repository=tmp_path, instance='test', timeout=10)
    monkeypatch.setattr('src.bot.wechat_bot.load_digital_human_config', lambda: config)
    monkeypatch.setattr(run_bot, 'load_digital_human_config', lambda: config)


@pytest.fixture
def isolated_bot(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('scripts.sync_knowledge.sync', lambda: False)
    monkeypatch.setattr('src.bot.wechat_bot.MemoryEngine', Mock)
    monkeypatch.setattr('src.reply.digital_human.DigitalHumanDraftClient', Mock())
    monkeypatch.setattr('src.bot.wechat_bot.GlobalStore', lambda: GlobalStore(state_file=str(tmp_path / 'state.json')))
    sender = Mock(silent_mode=False)
    sender.send.return_value = ActionResult(success=True)
    monkeypatch.setattr('src.bot.wechat_bot.WeChatMessageSender', lambda **kwargs: sender)
    db = Mock()
    monkeypatch.setattr('src.badcase.case_db.get_db', lambda: db)
    return sender


def test_run_bot_uses_default_bot_perception(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_bot, 'SingleInstanceLock', lambda: __import__('contextlib').nullcontext())
    monkeypatch.setattr(run_bot.signal, 'signal', Mock())
    factory = Mock()
    monkeypatch.setattr(run_bot, 'WeChatBot', factory)
    run_bot.main()
    assert 'perception' not in factory.call_args.kwargs
    factory.return_value.run_auto.assert_called_once()


def test_run_bot_selects_project_python_before_importing_bot(monkeypatch, tmp_path):
    entry = tmp_path / 'run_bot.py'
    entry.write_text(Path(run_bot.__file__).read_text())
    interpreter = entry.parent / '.venv' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    monkeypatch.setattr(run_bot.sys, 'prefix', '/external-python')
    monkeypatch.setattr(run_bot.sys, 'argv', [str(entry)])
    execute = Mock(side_effect=SystemExit(0))
    monkeypatch.setattr(run_bot.os, 'execv', execute)
    with pytest.raises(SystemExit):
        runpy.run_path(str(entry), run_name='__main__')
    execute.assert_called_once_with(str(interpreter), [str(interpreter), '-u', str(entry)])


def test_run_bot_logs_startup_failure_and_releases_lock(monkeypatch, tmp_path):
    pid_file = tmp_path / 'bot.pid'
    lock_type = run_bot.SingleInstanceLock
    monkeypatch.setattr(run_bot, 'SingleInstanceLock', lambda: lock_type(str(pid_file)))
    factory = Mock(side_effect=DigitalHumanError('startup dependency failed'))
    monkeypatch.setattr(run_bot, 'WeChatBot', factory)
    logger = Mock()
    monkeypatch.setattr(run_bot, 'get_logger', lambda: logger)
    with pytest.raises(DigitalHumanError, match='startup dependency failed'):
        run_bot.main()
    logger.error.assert_called_once_with('Bot 启动失败', exc_info=True)
    assert not pid_file.exists()
    with lock_type(str(pid_file)):
        assert pid_file.read_text() == str(run_bot.os.getpid())


def test_default_entry_uses_screenshot_pipeline(isolated_bot):
    bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280)
    assert isinstance(bot.perception, SmartPerceptionPipeline)
    assert bot.generator.last_active_llm == 'digital-human'
    assert bot.generator._client is not None


def test_default_entry_stops_before_polling_when_dh_initialization_fails(isolated_bot, monkeypatch):
    perception = Mock()
    monkeypatch.setattr('src.reply.digital_human.DigitalHumanDraftClient',
                        Mock(side_effect=DigitalHumanError('DH 初始化失败')))
    with pytest.raises(DigitalHumanError, match='初始化失败'):
        WeChatBot(PROFILE_WECHAT_MAC_1760X1280, perception=perception)
    perception.perceive.assert_not_called()
    isolated_bot.send.assert_not_called()


@pytest.mark.parametrize('is_group', [False, True])
@pytest.mark.parametrize('metadata', [False, True])
def test_default_entry_preserves_digital_human_input_boundary(isolated_bot, monkeypatch, is_group, metadata):
    monkeypatch.setenv('WECHAT_GROUP_REPLY_ENABLED', '1')
    target = ChatMessage('今天可以吗', 'friend', SenderType.OTHER, 'example',
                         chatroom_id='source-test' if metadata else None,
                         create_time=100 if metadata else None,
                         observed_at=None if metadata else 150)
    factory = Mock()
    client = factory.return_value
    client.draft.return_value = dict(replies=['DH 回复'], binding=dict(instance='test', generator='g-test', data='d-test'))
    client.review.return_value = dict(decision='allow', issues=[])
    monkeypatch.setattr('src.reply.digital_human.DigitalHumanDraftClient', factory)
    bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280)
    factory.assert_called_once()
    monkeypatch.setattr(bot.perception, 'perceive', Mock(return_value=PerceptionResult(
        'example', [target], [], '', is_group=is_group)))
    bot.debug_logger = Mock(current=None)
    bot.policy.should_reply = Mock(return_value=True)

    bot.tick()
    factory.assert_called_once()

    state = bot.global_store.chats['example']
    assert state.is_group is is_group
    if metadata:
        value = client.draft.call_args.args[0]
        assert value == dict(kind='messages', chat_type='group' if is_group else 'private',
                            chat_name='example', source_chat_id='source-test', context=[
                                dict(sender='friend', text='今天可以吗', is_self=False, timestamp=100)])
    else:
        assert client.draft.call_args.args[0] == dict(
            kind='observed', chat_type='group' if is_group else 'private', chat_name='example',
            context=[dict(sender='friend', text='今天可以吗', is_self=False, timestamp=150,
                          time_basis='first_observed')])
    isolated_bot.send.assert_called_once_with('DH 回复', chat_name='example')
    assert state.messages[-1].replied is True


def test_screenshot_pipeline_records_capture_time_before_recognition(monkeypatch):
    pipeline = SmartPerceptionPipeline(PROFILE_WECHAT_MAC_1760X1280)
    pipeline.capture = Mock()
    clock = Mock(return_value=100)
    def capture():
        clock.return_value = 101
        return Mock(image_path='screenshot.png', window_rect=Rect(0, 0, 1760, 1280), scale_factor=1)
    pipeline.capture.capture.side_effect = capture
    pipeline._compute_hash = Mock(return_value='hash')
    target = ChatMessage('hello', 'friend', SenderType.OTHER, 'example')
    result = PerceptionResult('example', [target], [], '')
    pipeline._run_with_api = Mock(return_value=result)
    monkeypatch.setattr('src.perception.smart_pipeline.time.time', clock)
    assert pipeline.perceive() is result
    assert target.observed_at == 100
    assert target.create_time is None


def test_chat_switch_uses_largest_visible_unread_and_debounces(isolated_bot, monkeypatch):
    bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280)
    bot.debug_logger = Mock(current=None)
    items = [ChatListItem(name, '', count, '', Rect(0, index * 60, 200, 60)) for index, (name, count) in enumerate([
        ('example', '20'), ('文件传输助手', '30'), ('few', '2'), ('many', '9')])]
    result = PerceptionResult('example', [], items, '', window_rect=Rect(0, 0, 1000, 800))
    clicker = Mock()
    monkeypatch.setattr('src.bot.wechat_bot.ChatListClicker', Mock(return_value=clicker))

    assert bot._try_switch_to_unread_chat(result) == 'many'
    clicker.click_item.assert_called_once_with(items[-1])
    assert bot._try_switch_to_unread_chat(result) == ''
    clicker.click_item.assert_called_once()

    for item in items:
        item.unread_count = ''
    assert bot._try_switch_to_unread_chat(result) == ''
    clicker.click_item.assert_called_once()
