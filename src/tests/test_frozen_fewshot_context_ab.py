import unittest
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_frozen_fewshot_context_ab as runner
from scripts.run_frozen_fewshot_context_ab import strip_examples


class StaticExamplesTest(unittest.TestCase):
    def test_balanced_resume_only_schedules_missing_rows_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in runner.ARMS:
                rows = [{'sample_id': f'{arm}:{i}'} for i in range(42)]
                runner.put(root / arm / 'samples.jsonl', ''.join(map(runner.dump, rows)))
                runner.put(root / arm / 'labels/15.jsonl', runner.dump({
                    'sample_id': f'{arm}:0', 'status': 'ok'}) + runner.dump({
                    'sample_id': f'{arm}:1', 'status': 'error'}))
            with patch.object(runner, 'OUT', root):
                plan = runner.prepare_pending()
                self.assertEqual(runner.prepare_pending(), plan)
                for arm in runner.ARMS:
                    shards = [runner.read_rows(plan / arm / f'{i:02}.jsonl') for i in range(16)]
                    ids = [r['sample_id'] for rows in shards for r in rows]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {f'{arm}:{i}' for i in range(1, 42)})
                    self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)
                    self.assertEqual(runner.successful_ids(arm), {f'{arm}:0'})

    def check_routed_run(self, arm, skills, ids=('one',), expected=('one',)):
        from scripts.label_few_shot_ranker_samples import FixedFewShotRetriever
        retriever = FixedFewShotRetriever([{'id': value} for value in ids])
        generator = SimpleNamespace(last_loaded_skills=['stale_previous_route'])
        audit, events = {}, []

        def run(**kwargs):
            events.append('route')
            generator.last_loaded_skills = skills
            retriever.render_selected(retriever.retrieve(limit=len(ids)))
            events.append('generate')
            return {'trace': {'loaded_skills': skills}, 'few_shot_ids': list(ids)}

        try:
            with patch.object(runner.PersonaFewShotRetriever, 'render', return_value=('unchanged', list(ids))):
                runner.run_c0_with_audit(arm, {'generation': {'injected_candidate_ids': list(expected)}},
                    audit, run, persona_few_shot_retriever=retriever, reply_generator=generator)
        except RuntimeError:
            if not audit.get('reuse'):
                raise
        finally:
            self.assertNotIn('render_selected', retriever.__dict__)
        return audit, events

    def test_non_hit_reuses_after_real_routing_before_generation(self):
        audit, events = self.check_routed_run(runner.ARMS[0], ['group_banter'])
        self.assertEqual(events, ['route'])
        self.assertTrue(audit['reuse'])
        self.assertEqual(audit['loaded_skills'], ['group_banter'])
        self.assertEqual(audit['injected_candidate_ids'], ['one'])

    def test_static_hit_completes_generation(self):
        audit, events = self.check_routed_run(runner.ARMS[0], ['casual_chat'])
        self.assertEqual(events, ['route', 'generate'])
        self.assertTrue(audit['static_treatment_applied'])
        self.assertNotIn('reuse', audit)

    def test_top6_does_not_skip_generation(self):
        audit, events = self.check_routed_run(runner.ARMS[1], ['group_banter'])
        self.assertEqual(events, ['route', 'generate'])
        self.assertFalse(audit['static_treatment_applied'])
        self.assertNotIn('reuse', audit)

    def test_non_hit_still_rejects_wrong_bound_ids(self):
        from scripts.label_few_shot_ranker_samples import FewShotBindingError
        with self.assertRaises(FewShotBindingError):
            self.check_routed_run(runner.ARMS[0], ['group_banter'], expected=('wrong',))

    def test_lazy_exports_match_original_and_cache_only_one_chat(self):
        from scripts.label_few_shot_ranker_samples import load_export_payloads
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('群聊_alpha', '私聊_beta', '曾经的好友_gamma'):
                (root / f'{name}.json').write_text(json.dumps({'messages': [name]}))
            expected, _ = load_export_payloads((root,))
            payloads = runner.ExportPayloads((root,))
            self.assertIsNone(payloads.cached_payload)
            for chat_id, payload in expected.items():
                self.assertEqual(payloads.get(chat_id), payload)
                self.assertEqual(payloads.cached_id, chat_id)
                self.assertIs(payloads.get(chat_id), payloads.cached_payload)
            self.assertIsNone(payloads.get('missing'))
            self.assertIsNone(payloads.cached_payload)

    def test_lazy_exports_reject_conflicting_duplicates(self):
        from scripts.label_few_shot_ranker_samples import load_export_payloads
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '群聊_alpha.json').write_text('{}')
            expected, _ = load_export_payloads((root,))
            (root / '私聊_alpha.json').write_text('{"different": true}')
            with self.assertRaises(ValueError):
                runner.ExportPayloads((root,)).get(next(iter(expected)))

    def test_only_examples_are_removed(self):
        before = '<skill>\n<rule>unchanged</rule>\n'
        after = '</skill>\n'
        examples = '  <examples trust="style_only">\n' + '<example>x</example>\n' * 3 + '</examples>\n'
        self.assertEqual(strip_examples(before + examples + after), before + after)

    def test_unexpected_source_fails_closed(self):
        for text in ('', '<examples><example>x</example></examples>',
                     '  <examples><example>x</example></examples>\n'):
            with self.assertRaises(ValueError):
                strip_examples(text)

    def test_shard_is_locked_while_worker_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / runner.ARMS[0]).mkdir()

            def check_lock(arm, shard):
                with (root / arm / f'{shard:02}.lock').open('a') as handle:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            with patch.object(runner, 'OUT', root), patch.object(runner, '_worker', side_effect=check_lock) as run:
                runner.worker(runner.ARMS[0], 8)
                run.assert_called_once_with(runner.ARMS[0], 8)
            with (root / runner.ARMS[0] / '08.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == '__main__':
    unittest.main()
