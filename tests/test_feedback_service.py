"""Durability, session boundaries and wire-level API validation without sockets."""
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from serving.server import ClosedLoopService, FeedbackAPIMixin, private_path
from pipeline.config import load_config


def sample(session='one', number=1, timestamp=None, severe=False):
    policy = load_config(None, 'rules.json')['featurePolicy']
    return {'session_id': session, 'participant': 'person-' + session, 'mode': 'live',
            'sample_id': str(number), 'timestamp_ms': timestamp or int(time.time() * 1000),
            'snapshot': {'hr': 95 if severe else 60, 'rmssd': 20 if severe else 50,
                         'sdnn': 20 if severe else 50, 'fresh': True,
                         'baselineState': 'ready', 'baseline': {'hr': 60, 'rmssd': 50, 'sdnn': 50},
                         'rrCount': policy['minRr'], 'windowSeconds': policy['windowSeconds'],
                         'requiredWindowSeconds': policy['windowSeconds'], 'featurePolicy': policy,
                         'postureCalibrated': True,
                         'posture': {'level': 'critical' if severe else 'poor', 'pitch': 50 if severe else 30, 'roll': 0}}}


class Request(FeedbackAPIMixin):
    """Exercise the mixin's actual parser, headers, serialization and dispatch."""
    def __init__(self, service, method, path, body=b'', headers=None):
        self.feedback_service = service
        self.command, self.path = method, path
        self.headers = {'Host': '127.0.0.1:8000', **(headers or {})}
        if method == 'POST' and 'Content-Length' not in self.headers:
            self.headers['Content-Length'] = str(len(body))
        self.rfile, self.wfile = io.BytesIO(body), io.BytesIO()
        self.status, self.response_headers = None, {}
        self.close_connection = False
    def send_response(self, code): self.status = code
    def send_header(self, key, value): self.response_headers[key] = value
    def end_headers(self): pass
    def payload(self): return json.loads(self.wfile.getvalue())


class FeedbackServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'records.sqlite3'
        self.service = ClosedLoopService(self.path)
    def tearDown(self):
        self.service.close()
        self.temp.cleanup()
    def request(self, method, path, payload=None, headers=None, raw=None):
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else b''
        request = Request(self.service, method, path, body, headers)
        self.assertTrue(request.handle_feedback_api())
        return request
    def test_retry_is_deduplicated_before_pipeline_and_survives_restart(self):
        reading = self.service.process_sample(sample())
        with patch.object(self.service.pipeline, 'process', side_effect=AssertionError('retry advanced pipeline')):
            self.assertEqual(self.service.process_sample(sample()), reading)
        self.service.close()
        self.service = ClosedLoopService(self.path)
        self.assertEqual(self.service.latest('one')['reading_id'], reading['reading_id'])
        self.assertEqual(self.service.summary('one')['readings'], 1)
    def test_concurrent_retries_produce_one_durable_reading(self):
        data=sample()
        with ThreadPoolExecutor(max_workers=6) as pool:
            results=list(pool.map(self.service.process_sample,[data]*12))
        self.assertEqual(len({r['reading_id'] for r in results}),1)
        self.assertEqual(self.service.summary('one')['readings'],1)
    def test_latest_uses_ingestion_order_without_reversing_observation_time(self):
        now = int(time.time() * 1000)
        newest = self.service.process_sample(sample(timestamp=now))
        self.service.process_sample(sample(number=2, timestamp=now-10000))
        self.assertEqual(self.service.latest('one')['reading_id'], newest['reading_id'])
        self.assertEqual(self.service.summary('one')['readings'], 2)
    def test_missing_data_is_persisted_but_cannot_be_rated(self):
        data = sample()
        data['snapshot'] = {'fresh': False}
        result = self.service.process_sample(data)
        self.assertEqual(self.service.summary('one')['readings'], 1)
        with self.assertRaises(ValueError):
            self.service.feedback({'session_id':'one','reading_id':result['reading_id'],'kind':'rating','level':'high'})
    def test_stale_polling_changes_availability_without_pipeline_calls(self):
        when = int(time.time() * 1000)
        self.service.process_sample(sample(timestamp=when))
        with patch('serving.server.time.time', return_value=when/1000 + 6), patch.object(self.service.pipeline, 'process', side_effect=AssertionError('GET advanced timers')):
            result = self.service.latest('one')
            self.assertTrue(result['stale'])
            self.assertFalse(result['states']['fatigue']['available'])
            self.assertEqual(result['alerts'], [])
            self.assertFalse(result['features']['fresh'])
            self.assertFalse(result['features']['fatigueEligible'])
            self.assertNotIn('hrChangePct',result['features'])
            self.assertNotIn('absPitch',result['features'])
    def test_stale_latest_invalidates_quick_test_score_without_rewriting_history(self):
        when = int(time.time()*1000)
        data = sample(timestamp=when)
        data['snapshot']['quickFatigue'] = dict(available=True, score=64, source='hr-only-debug', provisional=True)
        self.service.process_sample(data)
        with patch('serving.server.time.time', return_value=when/1000 + 6):
            quick = self.service.latest('one')['features']['snapshot']['quickFatigue']
            self.assertFalse(quick['available'])
            self.assertIsNone(quick['score'])
        self.assertEqual(self.service.export('one')['readings'][0]['sample']['snapshot']['quickFatigue']['score'], 64)

    def test_quick_test_score_is_saved_but_cannot_enable_full_rule_feedback(self):
        data = sample()
        data['snapshot'] = dict(fresh=False, quickFatigue=dict(available=True, score=99, source='hr-only-debug', provisional=True))
        result = self.service.process_sample(data)
        self.assertFalse(result['states']['fatigue']['available'])
        self.assertEqual(result['alerts'], [])
        self.assertEqual(result['features']['snapshot']['quickFatigue']['score'], 99)
    def test_independent_rating_export_and_context_are_durable(self):
        reading = self.service.process_sample(sample())
        self.service.set_context({'session_id':'one','participant':'person-one','mode':'live','context':{'task':'packing'}})
        self.service.feedback({'session_id':'one','reading_id':reading['reading_id'],'kind':'rating','level':'low','task':'packing'})
        self.service.close()
        self.service = ClosedLoopService(self.path)
        exported = self.service.export('one')
        self.assertEqual(exported['schemaVersion'], 2)
        self.assertEqual(exported['participant'], 'person-one')
        self.assertEqual(exported['fatigueRatings'][0]['level'], 'low')
        self.assertEqual(exported['fatigueRatings'][0]['source'], 'participant-self-report')
        self.assertEqual(exported['fatigueRatings'][0]['snapshot']['hr'], 60)
        self.assertEqual(self.service.context('one')['context'], {'task':'packing'})
    def test_rating_observation_times_survive_variable_network_latency(self):
        start = int(time.time()*1000)
        for number, observed, received in ((1, start+500, start+1000), (2, start+60500, start+60600)):
            reading_at = start+(number-1)*60000
            with patch('serving.server.time.time', return_value=(reading_at+50)/1000):
                reading = self.service.process_sample(sample(number=number,timestamp=reading_at))
            with patch('serving.server.time.time', return_value=received/1000):
                response = self.request('POST','/feedback',{'session_id':'one','reading_id':reading['reading_id'],
                    'kind':'rating','level':'low','observed_at_ms':observed})
            self.assertEqual(response.status,200,response.payload())
        ratings = self.service.export('one')['fatigueRatings']
        times = [datetime.fromisoformat(r['time'].replace('Z','+00:00')).timestamp() for r in ratings]
        self.assertEqual(times[1]-times[0],60)
        self.assertEqual(ratings[1]['observed_at_ms']-ratings[0]['observed_at_ms'],60000)
        self.assertAlmostEqual(ratings[1]['received_at_ms']-ratings[0]['received_at_ms'],59600,places=3)
        from train_polar_fatigue import load_sessions
        features, labels, _ = load_sessions([self.service.export('one')])
        self.assertEqual(features.shape,(2,3))
        self.assertEqual(labels.tolist(),['low','low'])
    def test_rating_without_observation_time_uses_linked_reading_time(self):
        now=int(time.time()*1000)
        reading=self.service.process_sample(sample(timestamp=now))
        with patch('serving.server.time.time',return_value=(now+900)/1000):
            rating=self.service.feedback({'session_id':'one','reading_id':reading['reading_id'],'kind':'rating','level':'low'})
        self.assertEqual(rating['observed_at_ms'],now)
        self.assertEqual(datetime.fromisoformat(rating['time'].replace('Z','+00:00')).timestamp(),now/1000)
    def test_rating_observation_time_rejects_invalid_stale_or_pre_reading_values(self):
        now=int(time.time()*1000)
        reading=self.service.process_sample(sample(timestamp=now))
        for observed in (True,'123',None,float('nan'),float('inf'),now-1,now+5001):
            with self.subTest(observed=observed), patch('serving.server.time.time',return_value=(now+1000)/1000):
                response=self.request('POST','/feedback',{'session_id':'one','reading_id':reading['reading_id'],
                    'kind':'rating','level':'low','observed_at_ms':observed})
                self.assertEqual(response.status,400)
        self.assertEqual(self.service.summary('one')['ratings'],0)
        with patch('serving.server.time.time',return_value=now/1000):
            future_reading=self.service.process_sample(sample('two',timestamp=now+4000))
            response=self.request('POST','/feedback',{'session_id':'two','reading_id':future_reading['reading_id'],
                'kind':'rating','level':'low','observed_at_ms':now+6000})
        self.assertEqual(response.status,400)
    def test_rating_spacing_is_enforced_across_restart_using_configured_window(self):
        now=int(time.time()*1000)
        reading=self.service.process_sample(sample(timestamp=now))
        self.service.pipeline.rules['featurePolicy']['windowSeconds']=2
        first={'session_id':'one','reading_id':reading['reading_id'],'kind':'rating','level':'low','observed_at_ms':now}
        self.service.feedback(first)
        self.service.close()
        self.service=ClosedLoopService(self.path)
        self.service.pipeline.rules['featurePolicy']['windowSeconds']=2
        with patch('serving.server.time.time',return_value=(now+1999)/1000):
            with self.assertRaisesRegex(ValueError,'apart'):
                self.service.feedback({**first,'observed_at_ms':now+1999})
        with patch('serving.server.time.time',return_value=(now+2000)/1000):
            self.service.feedback({**first,'observed_at_ms':now+2000})
        self.assertEqual(self.service.summary('one')['ratings'],2)
    def test_reading_references_cannot_cross_sessions(self):
        reading = self.service.process_sample(sample())
        self.service.process_sample(sample('two'))
        with self.assertRaises(ValueError):
            self.service.feedback({'session_id':'two','reading_id':reading['reading_id'],'kind':'rating','level':'low'})
        self.assertEqual(self.service.summary('two')['ratings'], 0)
    def test_session_identity_cannot_change_participant_or_mode(self):
        self.service.process_sample(sample())
        for changes in ({'participant':'different'}, {'mode':'demo'}):
            invalid = {**sample(number=2), **changes}
            with self.assertRaises(ValueError): self.service.process_sample(invalid)
        self.assertEqual(self.service.summary('one')['readings'], 1)
    def test_future_demo_observations_are_allowed_but_live_future_is_invalid(self):
        future = sample(timestamp=int(time.time()*1000)+60000)
        with self.assertRaises(ValueError): self.service.process_sample(future)
        future['mode'] = 'demo'
        result = self.service.process_sample(future)
        self.assertEqual(result['mode'], 'demo')
        self.assertFalse(self.service.latest('one')['stale'])
        with self.assertRaises(ValueError):
            self.service.feedback({'session_id':'one','reading_id':result['reading_id'],'kind':'rating','level':'high'})
    def test_alert_response_outcome_references_are_durable_and_scoped(self):
        when = int(time.time()*1000) - 65000
        for number in range(14):
            observed = when+number*5000
            with patch('serving.server.time.time', return_value=observed/1000):
                self.service.process_sample(sample(number=number, timestamp=observed, severe=True))
        exported = self.service.export('one')
        self.assertTrue(exported['alerts'])
        alert = exported['alerts'][0]['id']
        self.service.feedback({'session_id':'one','alert_id':alert,'kind':'response','response':'break-taken'})
        self.service.feedback({'session_id':'one','alert_id':alert,'kind':'outcome','outcome':'better'})
        self.service.process_sample(sample('two'))
        with self.assertRaises(ValueError):
            self.service.feedback({'session_id':'two','alert_id':alert,'kind':'outcome','outcome':'same'})
        self.service.close()
        self.service = ClosedLoopService(self.path)
        self.assertEqual(self.service.summary('one')['responses'], 1)
        self.assertEqual(self.service.summary('one')['outcomes'], 1)
    def test_delayed_live_evidence_is_logged_without_alerts_and_stays_ineligible_after_restart(self):
        when = int(time.time()*1000)-65000
        for number in range(8):
            result = self.service.process_sample(sample(number=number, timestamp=when+number*5000, severe=True))
            self.assertFalse(result['states']['fatigue']['available'])
            self.assertFalse(result['states']['posture']['available'])
        self.assertEqual(self.service.summary('one')['alerts'], 0)
        exported = self.service.export('one')
        self.assertTrue(exported['readings'][0]['sample']['snapshot']['fresh'])
        self.assertTrue(exported['readings'][0]['stale_on_arrival'])
        self.service.close()
        self.service = ClosedLoopService(self.path)
        self.assertEqual(self.service.summary('one')['alerts'], 0)
        self.assertFalse(self.service.latest('one')['states']['fatigue']['available'])
    def test_stale_raw_input_cannot_be_used_and_invalid_original_fields_are_rejected(self):
        data = sample(timestamp=int(time.time()*1000)-10000)
        data.pop('snapshot')
        data['raw'] = {'hr':90,'rr_intervals_ms':[700,710,690]}
        result = self.service.process_sample(data)
        self.assertFalse(result['states']['fatigue']['available'])
        malformed = sample(number=2,timestamp=int(time.time()*1000)-20000)
        malformed['snapshot']['posture']['pitch'] = 'invalid-number'
        with self.assertRaises(ValueError): self.service.process_sample(malformed)
    def test_actual_alert_cooldown_survives_restart_with_changed_config(self):
        root = Path(__file__).resolve().parents[1]
        config_dir = Path(self.temp.name) / 'config'
        config_dir.mkdir()
        for filename in ('rules.json','feedback_table.json'):
            (config_dir / filename).write_bytes((root / 'config' / filename).read_bytes())
        start = int(time.time()*1000)
        with patch('pipeline.config.CONFIG_DIR', config_dir):
            self.service.close()
            self.service = ClosedLoopService(self.path)
            for number in range(3):
                data = sample(number=number,timestamp=start+number*5000,severe=True)
                data['mode']='demo'
                self.service.process_sample(data)
            self.assertEqual(self.service.summary('one')['alerts'], 1)
            # Replay under a stricter count would never have alerted; the saved alert still counts.
            changed = json.loads((config_dir/'feedback_table.json').read_text())
            changed['decision']['consecutiveHighPredictions']=5
            (config_dir/'feedback_table.json').write_text(json.dumps(changed))
            self.service.close()
            self.service = ClosedLoopService(self.path)
            for number in range(3,10):
                data=sample(number=number,timestamp=start+number*5000,severe=True)
                data['mode']='demo'
                result=self.service.process_sample(data)
            self.assertEqual(result['states']['posture']['suppression'],'cooldown')
            self.assertEqual(self.service.summary('one')['alerts'],1)

    def test_real_static_handler_blocks_get_and_head_to_private_files(self):
        with patch('live_dashboard_backend.LiveDashboardBridge.start'):
            from serve_dashboard import Handler
        for method in ('GET','HEAD'):
            request=Request(self.service,method,'/.local/closed_loop.sqlite3')
            getattr(Handler,'do_'+method)(request)
            self.assertEqual(request.status,403)
    def test_real_handler_rejects_unsupported_api_methods_with_json(self):
        with patch('live_dashboard_backend.LiveDashboardBridge.start'):
            from serve_dashboard import Handler
        for method in ('PUT','DELETE','PATCH','OPTIONS','HEAD'):
            request=Request(self.service,method,'/samples')
            getattr(Handler,'do_'+method)(request)
            self.assertEqual(request.status,405)
            self.assertIn('application/json',request.response_headers['Content-Type'])
            if method=='HEAD': self.assertEqual(request.wfile.getvalue(),b'')
    def test_routes_return_json_and_preserve_historical_routes(self):
        result = self.request('POST','/samples',sample()).payload()
        self.assertIsInstance(result['reading_id'], str)
        for path in ('/latest','/decision','/context','/summary','/export'):
            response = self.request('GET',path+'?session_id=one')
            self.assertEqual(response.status, 200)
            self.assertIn('application/json', response.response_headers['Content-Type'])
        self.assertFalse(Request(self.service,'GET','/api/dashboard-data').handle_feedback_api())
    def test_cross_origin_and_dns_rebinding_mutations_are_rejected(self):
        for headers in ({'Origin':'https://evil.invalid'}, {'Host':'evil.invalid:8000'}, {'Origin':'http://127.0.0.1:7000'}):
            response = self.request('POST','/samples',sample(),headers)
            self.assertEqual(response.status, 403)
        self.assertEqual(self.service.summary('one')['readings'], 0)
        self.assertEqual(self.request('POST','/samples',sample(), {'Origin':'http://127.0.0.1:8000'}).status, 200)
    def test_bad_json_size_and_invalid_numbers_are_errors(self):
        for raw in (b'{', b'[]', b'{"timestamp_ms":NaN}', b'{"a":1,"a":2}'):
            self.assertEqual(self.request('POST','/samples',raw=raw).status, 400)
        self.assertEqual(self.request('POST','/samples',raw=b'', headers={'Content-Length':'1000001'}).status, 413)
        self.assertEqual(self.request('POST','/samples',raw=b'', headers={'Content-Length':'-1'}).status, 400)
        self.assertEqual(self.request('GET','/latest').status, 400)
    def test_private_paths_include_encoded_traversal_and_runtime_symlinks(self):
        root = Path(self.temp.name)
        (root / '.local').mkdir()
        (root / '.local' / 'private.sqlite3').touch()
        (root / 'alias').symlink_to(root / '.local', target_is_directory=True)
        for path in ('/.local/private.sqlite3','/%2elocal/private.sqlite3','/x/../.local/private.sqlite3','/alias/private.sqlite3','/.git/config','/secret.sqlite3'):
            self.assertTrue(private_path(path,root),path)
        self.assertFalse(private_path('/config/rules.json',root))

    def test_failed_persistence_returns_json_error_and_does_not_consume_retry(self):
        with patch.object(self.service.store, 'save_reading', side_effect=OSError('disk unavailable')):
            response = self.request('POST','/samples',sample())
        self.assertEqual(response.status, 503)
        self.assertEqual(self.service.summary('one')['readings'], 0)
        self.assertIsInstance(self.request('POST','/samples',sample()).payload()['reading_id'], str)

    def test_current_store_is_private_even_with_nonstandard_filename(self):
        self.assertTrue(private_path('/records.sqlite3', Path(self.temp.name)))
        self.service.close()
        self.path = Path(self.temp.name) / 'private-records.data'
        self.service = ClosedLoopService(self.path)
        self.assertTrue(private_path('/private-records.data', Path(self.temp.name)))

    def test_server_direct_script_help_runs_without_starting_server(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, 'serving/server.py', '--help'], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--db', result.stdout)


if __name__ == '__main__': unittest.main()
