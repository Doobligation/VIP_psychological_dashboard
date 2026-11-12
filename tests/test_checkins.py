"""Closed-loop check-ins (guide steps 5-8): when to ask, answers, expiry, agreement and the worker's files."""
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from pipeline import checkins as rules
from serving.server import ClosedLoopService
from tests.test_feedback_service import Request

SCENARIO = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'demo_scenario.json').read_text())
MINUTE = 60_000


def scripted(session, number, timestamp, pattern, source='team3-dashboard', participant='worker-1', mode='demo'):
    """One prediction like the dashboard's scripted demo: a % change pattern from config/demo_scenario.json."""
    from pipeline.config import load_config
    policy = load_config(None, 'rules.json')['featurePolicy']
    baseline, changes = SCENARIO['baseline'], SCENARIO['patterns'][pattern]
    snapshot = {k: baseline[k] * (1 + changes[k] / 100) for k in baseline}
    snapshot.update(edaFresh=True, baseline=dict(baseline), baselineState='ready', fresh=True, rrCount=max(60, policy['minRr']),
                    windowSeconds=policy['windowSeconds'], requiredWindowSeconds=policy['windowSeconds'],
                    featurePolicy=dict(policy), postureCalibrated=True, posture=dict(level='good', pitch=0, roll=0))
    raw = {'source': source} if source else {}
    return dict(session_id=session, participant=participant, mode=mode, sample_id=str(number), timestamp_ms=timestamp,
                snapshot=snapshot, raw=raw)


class CheckinServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'records.sqlite3'
        self.service = ClosedLoopService(self.db)
        self.now = time.time() * 1000

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def feed(self, session, patterns, start=None, first=0, **kwargs):
        start = self.now if start is None else start
        return [self.service.process_sample(scripted(session, first + i, start + i * 1000, p, **kwargs))
                for i, p in enumerate(patterns)]

    def files(self, participant='worker-1'):
        folder = Path(self.temp.name) / 'checkins' / rules.folder_name(participant)
        rows = list(csv.DictReader((folder / 'checkins.csv').read_text(encoding='utf-8-sig').splitlines()))
        return rows, json.loads((folder / 'checkins.json').read_text())

    def answer(self, session, checkin_id, frustration='yes', fatigue='no', **extra):
        return self.service.checkin_action(dict(session_id=session, kind='answer', checkin_id=checkin_id,
                                                answers=dict(frustration=frustration, fatigue=fatigue), **extra))

    def test_a_high_prediction_opens_one_quick_check_about_that_moment(self):
        results = self.feed('s', ['rest', 'frustrationMedium', 'frustrationHigh', 'frustrationHigh', 'frustrationHigh'])
        view = self.service.checkins('s')
        question = view['open']
        self.assertEqual(question['trigger'], ['frustration'])
        self.assertEqual(question['moment_ms'], results[2]['timestamp_ms'])
        self.assertEqual(question['reading_id'], results[2]['reading_id'])
        self.assertEqual(question['predicted']['frustration']['level'], 'high')
        self.assertEqual(question['predicted']['frustration']['source'], 'Scripted demo · DBR rules')
        self.assertEqual(question['predictedCause']['key'], results[2]['states']['frustration']['cause'])
        self.assertEqual(view['session']['counts'], dict(asked=1, answered=0, noAnswer=0, dismissed=0, open=1, selfReports=0))
        # Only one question is open at a time, so the next Highs do not ask again.
        self.assertEqual(len(view['checkins']), 1)
        rows, saved = self.files()
        self.assertEqual([r['status'] for r in rows], ['Open'])
        self.assertEqual(rows[0]['frustration_predicted'], 'high')
        change = {k: v for k, v in results[2]['features'].items() if k.endswith('ChangePct')}
        self.assertTrue(change)
        self.assertEqual(saved['checkins'][0]['features']['changePct'], change)

    def test_questions_keep_a_gap_and_ask_only_about_recent_moments(self):
        self.feed('s', ['frustrationHigh'])
        first = self.service.checkins('s')['open']
        self.answer('s', first['id'])
        self.feed('s', ['frustrationHigh'] * 3, start=self.now + 1000, first=1)
        self.assertEqual(len(self.service.checkins('s')['checkins']), 1, 'asked again inside the 15-minute gap')
        later = time.time() + 16 * 60
        with patch('serving.server.time.time', return_value=later):
            self.feed('s', ['frustrationHigh'], start=later * 1000, first=10)
            self.assertEqual(len(self.service.checkins('s')['checkins']), 2)
        # A moment more than 30 minutes before asking is not asked about.
        self.feed('old', ['frustrationHigh'], start=self.now - 31 * MINUTE)
        self.assertIsNone(self.service.checkins('old')['open'])
        self.feed('recent', ['frustrationHigh'], start=self.now - 29 * MINUTE)
        self.assertIsNotNone(self.service.checkins('recent')['open'])

    def test_high_fatigue_also_asks_and_only_configured_sources_ask(self):
        self.feed('fatigue', ['fatigueHigh'])
        self.assertEqual(self.service.checkins('fatigue')['open']['trigger'], ['fatigue'])
        self.feed('medium', ['frustrationMedium', 'fatigueMedium'])
        self.assertIsNone(self.service.checkins('medium')['open'])
        # The Polar H10 page sends no raw.source; the guide keeps heart-only signals research-only.
        results = self.feed('polar', ['frustrationHigh', 'fatigueHigh'], source=None, mode='live')
        self.assertEqual(results[1]['states']['fatigue']['level'], 'high')
        self.assertIsNone(self.service.checkins('polar')['open'])
        self.assertEqual(self.service.checkins('polar')['levels']['device'], 'Polar H10')

    def test_an_unanswered_question_is_saved_as_no_answer(self):
        self.feed('s', ['frustrationHigh'])
        question = self.service.checkins('s')['open']
        with patch('serving.server.time.time', return_value=time.time() + 11 * 60):
            view = self.service.checkins('s')
            self.assertIsNone(view['open'])
            self.assertEqual(view['checkins'][0]['status'], 'no-answer')
            self.assertEqual(view['session']['counts']['noAnswer'], 1)
            with self.assertRaisesRegex(ValueError, 'already closed'):
                self.answer('s', question['id'])
        rows, _ = self.files()
        self.assertEqual(rows[0]['status'], 'No answer')

    def test_answers_are_validated_saved_and_scored_against_the_prediction(self):
        self.feed('s', ['frustrationHigh'])
        self.feed('other', ['frustrationHigh'], participant='worker-2')
        question = self.service.checkins('s')['open']
        with self.assertRaisesRegex(ValueError, 'answers must give'):
            self.service.checkin_action(dict(session_id='s', kind='answer', checkin_id=question['id'], answers=dict(frustration='yes')))
        with self.assertRaisesRegex(ValueError, 'answers must give'):
            self.answer('s', question['id'], frustration='maybe')
        with self.assertRaisesRegex(ValueError, 'cause must be'):
            self.answer('s', question['id'], cause='aliens')
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            self.answer('other', question['id'])
        saved = self.answer('s', question['id'], frustration='no', fatigue='no', cause='schedule-pressure', note='Crane was late')
        self.assertEqual(saved['status'], 'answered')
        self.assertEqual(saved['agrees'], dict(frustration=False, fatigue=True))
        self.assertEqual(saved['reportedCause'], dict(key='schedule-pressure', label='Schedule Pressure'))
        row = self.service.checkins('s')['all']['agreement'][0]
        self.assertEqual((row['state'], row['predictedHigh'], row['no'], row['confirmedRate']), ('frustration', 1, 1, 0.0))
        rows, saved_json = self.files()
        self.assertEqual((rows[0]['felt_frustrated'], rows[0]['what_was_going_on'], rows[0]['note'], rows[0]['frustration_agrees']),
                         ('No', 'Schedule Pressure', 'Crane was late', 'no'))
        self.assertEqual(saved_json['checkins'][0]['answers'], dict(frustration='no', fatigue='no'))
        with self.assertRaisesRegex(ValueError, 'already closed'):
            self.answer('s', question['id'])

    def test_not_now_and_self_reports(self):
        self.feed('s', ['frustrationHigh'])
        question = self.service.checkins('s')['open']
        dismissed = self.service.checkin_action(dict(session_id='s', kind='dismiss', checkin_id=question['id']))
        self.assertEqual(dismissed['status'], 'dismissed')
        report = dict(session_id='s', kind='self-report', answers=dict(frustration='a-little', fatigue='yes'), cause='not-sure')
        saved = self.service.checkin_action(report)
        self.assertEqual((saved['kind'], saved['status'], saved['predicted']['frustration']['level']), ('self-report', 'answered', 'high'))
        self.assertIsNotNone(saved['reading_id'])
        self.assertEqual(saved['agrees']['frustration'], True)  # "A little" counts as felt by default
        with self.assertRaisesRegex(ValueError, 'just saved'):
            self.service.checkin_action(report)
        with self.assertRaisesRegex(ValueError, 'Unknown session'):
            self.service.checkin_action(dict(report, session_id='nobody'))
        # A delayed session has no reading near "now" yet: saved, matched to features at retraining.
        self.feed('late', ['frustrationHigh'], start=self.now - 40 * MINUTE)
        late = self.service.checkin_action(dict(report, session_id='late'))
        self.assertIsNone(late['reading_id'])
        self.assertEqual(late['predicted']['frustration']['level'], 'unavailable')
        self.assertIsNone(late['agrees']['frustration'])
        counts = self.service.checkins('s')['session']['counts']
        self.assertEqual((counts['asked'], counts['dismissed'], counts['selfReports']), (1, 1, 1))

    def test_predictions_by_level_and_the_medium_or_higher_run(self):
        patterns = ['rest', 'frustrationMedium', 'frustrationHigh', 'frustrationHigh', 'rest', 'frustrationMedium']
        results = self.feed('s', patterns)
        levels = self.service.checkins('s')['levels']
        counted = {level: sum(r['states']['frustration']['level'] == level for r in results) for level in ('low', 'medium', 'high')}
        self.assertEqual({k: levels['frustration'][k] for k in counted}, counted)
        self.assertEqual((levels['frustration']['run'], levels['frustration']['longestRun']), (1, 3))
        self.feed('s', ['frustrationHigh'], start=self.now + 10_000, first=50)
        self.assertEqual(self.service.checkins('s')['levels']['frustration']['run'], 2)
        # A fresh service counts the same from the database.
        self.service.close()
        self.service = ClosedLoopService(self.db)
        self.assertEqual(self.service.checkins('s')['levels']['frustration']['run'], 2)

    def test_checkins_survive_a_restart_and_replay_does_not_ask_again(self):
        self.feed('s', ['frustrationHigh'] * 3)
        before = self.service.checkins('s')['checkins']
        self.service.close()
        self.service = ClosedLoopService(self.db)
        after = self.service.checkins('s')['checkins']
        self.assertEqual([c['id'] for c in after], [c['id'] for c in before])
        self.assertEqual(self.service.export('s')['checkins'][0]['id'], before[0]['id'])

    def test_http_routes(self):
        self.feed('s', ['frustrationHigh'])
        get = Request(self.service, 'GET', '/checkins?session_id=s')
        self.assertTrue(get.handle_feedback_api())
        self.assertEqual(get.status, 200)
        question = get.payload()['open']
        body = json.dumps(dict(session_id='s', kind='answer', checkin_id=question['id'],
                               answers=dict(frustration='yes', fatigue='a-little'))).encode()
        post = Request(self.service, 'POST', '/checkins', body)
        self.assertTrue(post.handle_feedback_api())
        self.assertEqual((post.status, post.payload()['status']), (200, 'answered'))
        bad = Request(self.service, 'POST', '/checkins', json.dumps(dict(session_id='s', kind='guess')).encode())
        bad.handle_feedback_api()
        self.assertEqual(bad.status, 400)
        wrong = Request(self.service, 'DELETE', '/checkins?session_id=s')
        wrong.handle_feedback_api()
        self.assertEqual(wrong.status, 405)
        foreign = Request(self.service, 'POST', '/checkins', body, {'Origin': 'http://evil.example'})
        foreign.handle_feedback_api()
        self.assertEqual(foreign.status, 403)


class CheckinRuleTests(unittest.TestCase):
    def test_agreement_follows_the_little_setting_and_skips_missing_sides(self):
        record = dict(kind='self-report', status='answered', mode='live',
                      predicted=dict(frustration=dict(level='high', source='Empatica · ML model'),
                                     fatigue=dict(level='unavailable', source='Empatica · DBR rules')),
                      answers=dict(frustration='a-little', fatigue='yes'))
        self.assertEqual(rules.agreement(record, 'yes'), dict(frustration=True, fatigue=None))
        self.assertEqual(rules.agreement(record, 'no'), dict(frustration=False, fatigue=None))
        summary = rules.summarize([record], 'no')
        self.assertEqual(len(summary['agreement']), 1)
        self.assertEqual((summary['agreement'][0]['aLittle'], summary['agreement'][0]['confirmed']), (1, 0))

    def test_worker_folders_never_collide_or_escape(self):
        self.assertEqual(rules.folder_name('empatica-XYZ'), 'empatica-XYZ')
        self.assertNotEqual(rules.folder_name('a/b'), rules.folder_name('a_b'))
        self.assertFalse(rules.folder_name('../x').startswith('.'))
        self.assertNotIn('/', rules.folder_name('../../etc/passwd'))

    def test_settings_are_validated(self):
        good = rules.load_settings()
        self.assertEqual((good['minGapMinutes'], good['maxAgeMinutes'], good['expireMinutes']), (15, 30, 10))
        for change in ({'littleCountsAs': 'maybe'}, {'minGapMinutes': 0}, {'askLevels': {'stress': ['high']}},
                       {'askLevels': {'frustration': ['extreme']}}, {'questions': {}}):
            with self.assertRaises(ValueError):
                rules.load_settings(good | change)


if __name__ == '__main__':
    unittest.main()
