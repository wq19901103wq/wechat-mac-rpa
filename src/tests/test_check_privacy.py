import subprocess
import sys
from pathlib import Path

import pytest


SCANNER = Path(__file__).resolve().parents[2] / 'scripts/check_privacy.py'


@pytest.fixture
def repository(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    return tmp_path


def stage(repository, name, content):
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(['git', 'add', name], cwd=repository, check=True)
    return path


def scan(repository):
    return subprocess.run([sys.executable, str(SCANNER), '--staged'], cwd=repository,
                          capture_output=True, text=True)


@pytest.mark.parametrize('content', [
    'sk-' + 'x' * 32,
    'wxid_' + 'sensitive_test_identifier',
    '12345678' + '@chatroom',
    '-----BEGIN ' + 'PRIVATE KEY-----',
    '/Users/' + 'private-account/local',
])
def test_staged_privacy_findings_survive_worktree_cleanup(repository, content):
    path = stage(repository, 'example.txt', content)
    path.write_text('safe worktree')
    result = scan(repository)
    assert result.returncode == 1
    assert 'example.txt' in result.stderr
    assert content not in result.stderr


def test_scan_uses_index_and_allows_synthetic_identifiers(repository):
    path = stage(repository, 'example.txt', 'friend@example.com wxid_friend ' + 'a' * 32 + '138' + '12345678' + 'a' * 32)
    path.write_text('sk-' + 'x' * 32)
    assert scan(repository).returncode == 0


@pytest.mark.parametrize('name', ['.env', 'config/digital_human.json', 'data/chat.txt'])
def test_private_paths_are_rejected_without_recognizable_secrets(repository, name):
    stage(repository, name, '{}')
    result = scan(repository)
    assert result.returncode == 1
    assert 'private path is tracked' in result.stderr
