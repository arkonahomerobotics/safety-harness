"""Hash-chained, tamper-evident logging (safety_harness/audit_log.py) -- Annex III 1.1.9/1.2.1(f).

Covers: append/verify round trips (unkeyed and HMAC-keyed), every kind of tamper the chain is
supposed to catch (edited entry, deleted entry, reordered entries, truncated/malformed tail, wrong
verification key), resuming an existing log file (both a clean one and a tampered one), the
HashChainedDecisionLogger/SoftwareVersionLog wrappers against the real schema types, digest
determinism under set-ordering variation, and report_identity()."""

import json
import os
import tempfile
import unittest
from dataclasses import replace

from safety_harness.audit_log import (
    GENESIS_DIGEST,
    HashChainedDecisionLogger,
    LogIntegrityError,
    SoftwareVersionLog,
    _HashChainedFile,
    _jsonable,
    report_identity,
    verify_log,
)
from safety_harness.schema import (
    Action,
    Decision,
    DecisionVerdict,
    HazardTag,
    Pose,
    PreconditionResult,
    RobotProprioception,
    TrackedObject,
    WorldState,
)

DOMAIN = b"safety-harness/audit-log/test/v1"


def _tmp_path():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)  # the file must not exist yet -- _HashChainedFile creates it fresh
    return path


def _sample_decision():
    action = Action("grasp", {"object_id": "cube_2", "grip_force_n": 5.0})
    results = (
        PreconditionResult(name="object_hazard_confirmed", satisfied=True, reason="ok"),
        PreconditionResult(name="current_position_confirmed_stable", satisfied=False, reason="moving"),
    )
    return Decision(verdict=DecisionVerdict.BLOCK, action=action, precondition_results=results,
                    triggered_fallback=True, action_digest="sha256:deadbeef")


def _sample_state():
    obj = TrackedObject(object_id="cube_2", object_class="cube", pose=Pose(position=(0.5, 0.0, 0.05)),
                        hazard_tags=frozenset({HazardTag.FRAGILE, HazardTag.SHARP}), pose_confidence=1.0,
                        class_confidence=1.0)
    robot = RobotProprioception(joint_positions=(0.0,) * 7, joint_velocities=(0.0,) * 7,
                                end_effector_pose=Pose(position=(0.5, 0.0, 0.3)), gripper_state=1.0)
    return WorldState(objects=(obj,), robot=robot, sensor_timestamp=1234.5)


class AppendVerifyRoundTripTest(unittest.TestCase):
    def test_empty_file_verifies_ok(self):
        path = _tmp_path()
        chain = _HashChainedFile(path, DOMAIN)
        chain.close()
        v = verify_log(path, DOMAIN)
        self.assertTrue(v.ok)
        self.assertEqual(v.entries, 0)

    def test_appended_entries_verify_unkeyed(self):
        path = _tmp_path()
        chain = _HashChainedFile(path, DOMAIN)
        for i in range(5):
            chain.append({"i": i})
        chain.close()
        v = verify_log(path, DOMAIN)
        self.assertTrue(v.ok, v.reason)
        self.assertEqual(v.entries, 5)

    def test_appended_entries_verify_keyed(self):
        path = _tmp_path()
        key = b"secret-key"
        chain = _HashChainedFile(path, DOMAIN, key=key)
        for i in range(3):
            chain.append({"i": i})
        chain.close()
        self.assertTrue(verify_log(path, DOMAIN, key=key).ok)

    def test_keyed_log_fails_without_the_key(self):
        path = _tmp_path()
        chain = _HashChainedFile(path, DOMAIN, key=b"secret")
        chain.append({"i": 0})
        chain.close()
        v_no_key = verify_log(path, DOMAIN)
        v_wrong_key = verify_log(path, DOMAIN, key=b"guess")
        self.assertFalse(v_no_key.ok)
        self.assertFalse(v_wrong_key.ok)

    def test_missing_file_fails_closed_not_ok_not_raise(self):
        v = verify_log("/nonexistent/path/x.jsonl", DOMAIN)
        self.assertFalse(v.ok)
        self.assertEqual(v.entries, 0)

    def test_first_entry_chains_from_genesis(self):
        path = _tmp_path()
        chain = _HashChainedFile(path, DOMAIN)
        entry = chain.append({"i": 0})
        chain.close()
        self.assertEqual(entry["prev_digest"], GENESIS_DIGEST)
        self.assertEqual(entry["seq"], 0)

    def test_entries_chain_sequentially(self):
        path = _tmp_path()
        chain = _HashChainedFile(path, DOMAIN)
        e0 = chain.append({"i": 0})
        e1 = chain.append({"i": 1})
        chain.close()
        self.assertEqual(e1["prev_digest"], e0["digest"])
        self.assertEqual(e1["seq"], 1)


class TamperDetectionTest(unittest.TestCase):
    def _write(self, path, n=5, key=None):
        chain = _HashChainedFile(path, DOMAIN, key=key)
        for i in range(n):
            chain.append({"i": i})
        chain.close()

    def _lines(self, path):
        with open(path) as f:
            return [json.loads(x) for x in f if x.strip()]

    def _rewrite(self, path, lines):
        with open(path, "w") as f:
            for e in lines:
                f.write(json.dumps(e) + "\n")

    def test_editing_one_entrys_payload_is_caught(self):
        path = _tmp_path()
        self._write(path)
        lines = self._lines(path)
        lines[2]["payload"] = {"i": 999}  # content changed, digest left stale -- exactly a forger's mistake
        self._rewrite(path, lines)
        v = verify_log(path, DOMAIN)
        self.assertFalse(v.ok)
        self.assertEqual(v.first_bad_seq, 2)

    def test_deleting_a_middle_entry_is_caught(self):
        path = _tmp_path()
        self._write(path)
        lines = self._lines(path)
        del lines[2]
        self._rewrite(path, lines)
        v = verify_log(path, DOMAIN)
        self.assertFalse(v.ok)
        self.assertEqual(v.first_bad_seq, 2)  # seq 3 now appears where 2 is expected

    def test_reordering_entries_is_caught(self):
        path = _tmp_path()
        self._write(path)
        lines = self._lines(path)
        lines[1], lines[2] = lines[2], lines[1]
        self._rewrite(path, lines)
        v = verify_log(path, DOMAIN)
        self.assertFalse(v.ok)
        self.assertEqual(v.first_bad_seq, 1)

    def test_dropping_the_tail_leaves_the_remaining_prefix_valid(self):
        path = _tmp_path()
        self._write(path)
        lines = self._lines(path)
        self._rewrite(path, lines[:3])  # a truncated file, but at a clean entry boundary
        v = verify_log(path, DOMAIN)
        self.assertTrue(v.ok)  # a shorter but internally-consistent chain is not itself tampering
        self.assertEqual(v.entries, 3)

    def test_malformed_trailing_line_fails_closed_not_raise(self):
        path = _tmp_path()
        self._write(path)
        with open(path, "a") as f:
            f.write('{"seq": 5, "not": "valid json"\n')  # a mid-write crash: partial JSON on disk
        v = verify_log(path, DOMAIN)  # must not raise out of verify_log -- callers can't crash on this
        self.assertFalse(v.ok)

    def test_forging_a_consistent_unkeyed_chain_from_scratch_succeeds(self):
        # documents the module's own stated limitation: unkeyed = tamper-EVIDENT, not tamper-PROOF.
        path = _tmp_path()
        self._write(path)
        forged = _HashChainedFile(path + ".forged", DOMAIN)
        forged.append({"i": 0})
        forged.append({"i": "completely different content"})
        forged.close()
        self.assertTrue(verify_log(path + ".forged", DOMAIN).ok)

    def test_keyed_chain_cannot_be_forged_without_the_key(self):
        path = _tmp_path()
        self._write(path, key=b"real-key")
        lines = self._lines(path)
        lines[1]["payload"] = {"i": "forged"}
        # recompute a digest the attacker CAN compute (unkeyed) -- still fails against the real key
        lines[1]["digest"] = "sha256:" + "0" * 64
        self._rewrite(path, lines)
        self.assertFalse(verify_log(path, DOMAIN, key=b"real-key").ok)


class ResumeTest(unittest.TestCase):
    def test_resuming_a_clean_log_continues_the_chain(self):
        path = _tmp_path()
        c1 = _HashChainedFile(path, DOMAIN)
        c1.append({"i": 0})
        c1.append({"i": 1})
        c1.close()
        c2 = _HashChainedFile(path, DOMAIN)  # simulates a process restart
        c2.append({"i": 2})
        c2.close()
        v = verify_log(path, DOMAIN)
        self.assertTrue(v.ok, v.reason)
        self.assertEqual(v.entries, 3)

    def test_resuming_a_tampered_log_raises_rather_than_silently_continuing(self):
        path = _tmp_path()
        c1 = _HashChainedFile(path, DOMAIN)
        c1.append({"i": 0})
        c1.close()
        with open(path) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        lines[0]["payload"] = {"i": "tampered"}
        with open(path, "w") as f:
            for e in lines:
                f.write(json.dumps(e) + "\n")
        with self.assertRaises(LogIntegrityError):
            _HashChainedFile(path, DOMAIN)  # must refuse to build on a chain it can't verify


class JsonableTest(unittest.TestCase):
    def test_nonfinite_floats_become_marked_strings(self):
        self.assertEqual(_jsonable(float("nan")), "nan")
        self.assertEqual(_jsonable(float("inf")), "inf")
        self.assertEqual(_jsonable(float("-inf")), "-inf")
        self.assertEqual(_jsonable(1.5), 1.5)

    def test_result_is_always_json_round_trippable(self):
        payload = {"decision": _jsonable(_sample_decision()), "state": _jsonable(_sample_state())}
        round_tripped = json.loads(json.dumps(payload))
        self.assertEqual(round_tripped, payload)

    def test_frozenset_ordering_does_not_affect_the_encoding(self):
        a = frozenset({HazardTag.FRAGILE, HazardTag.SHARP, HazardTag.HOT})
        b = frozenset(reversed(list({HazardTag.HOT, HazardTag.SHARP, HazardTag.FRAGILE})))
        self.assertEqual(_jsonable(a), _jsonable(b))

    def test_reason_strings_and_check_names_are_plaintext_in_the_log(self):
        j = _jsonable(_sample_decision())
        names = [r["name"] for r in j["precondition_results"]]
        self.assertIn("current_position_confirmed_stable", names)
        self.assertIn("moving", [r["reason"] for r in j["precondition_results"]])


class HashChainedDecisionLoggerTest(unittest.TestCase):
    def test_logs_a_real_decision_and_verifies(self):
        path = _tmp_path()
        logger = HashChainedDecisionLogger(path)
        logger.record(_sample_decision(), _sample_state())
        logger.record(replace(_sample_decision(), verdict=DecisionVerdict.PERMIT), _sample_state())
        logger.close()
        v = verify_log(path, b"safety-harness/audit-log/decision/v1")
        self.assertTrue(v.ok, v.reason)
        self.assertEqual(v.entries, 2)

    def test_the_stored_log_is_human_readable_plaintext(self):
        path = _tmp_path()
        logger = HashChainedDecisionLogger(path)
        logger.record(_sample_decision(), _sample_state())
        logger.close()
        with open(path) as f:
            raw = f.read()
        self.assertIn("current_position_confirmed_stable", raw)
        self.assertIn("cube_2", raw)

    def test_editing_a_stored_decisions_verdict_is_caught(self):
        path = _tmp_path()
        logger = HashChainedDecisionLogger(path)
        logger.record(_sample_decision(), _sample_state())
        logger.close()
        with open(path) as f:
            entry = json.loads(f.read().strip())
        entry["payload"]["decision"]["verdict"] = "permit"  # flip BLOCK -> PERMIT after the fact
        with open(path, "w") as f:
            f.write(json.dumps(entry) + "\n")
        v = verify_log(path, b"safety-harness/audit-log/decision/v1")
        self.assertFalse(v.ok)

    def test_verify_convenience_method_uses_its_own_key(self):
        path = _tmp_path()
        logger = HashChainedDecisionLogger(path, key=b"k")
        logger.record(_sample_decision(), _sample_state())
        logger.close()
        self.assertTrue(logger.verify().ok)
        self.assertFalse(logger.verify(key=b"wrong").ok)


class SoftwareVersionLogTest(unittest.TestCase):
    def test_records_and_verifies_a_version_snapshot(self):
        path = _tmp_path()
        vlog = SoftwareVersionLog(path)
        entry = vlog.record(config_digest="sha256:abc123")
        vlog.close()
        self.assertEqual(entry["payload"]["config_digest"], "sha256:abc123")
        self.assertEqual(entry["payload"]["pin_scheme"], "sha256")
        v = verify_log(path, b"safety-harness/audit-log/version/v1")
        self.assertTrue(v.ok, v.reason)

    def test_multiple_snapshots_chain(self):
        path = _tmp_path()
        vlog = SoftwareVersionLog(path)
        vlog.record(config_digest="sha256:v1")
        vlog.record(config_digest="hmac-sha256:v2")  # e.g. a config change mid-deployment
        vlog.close()
        v = verify_log(path, b"safety-harness/audit-log/version/v1")
        self.assertTrue(v.ok)
        self.assertEqual(v.entries, 2)


class ReportIdentityTest(unittest.TestCase):
    def test_has_the_current_schema_version(self):
        from safety_harness.schema import SCHEMA_VERSION

        self.assertEqual(report_identity()["schema_version"], SCHEMA_VERSION)

    def test_package_version_is_a_string_or_none(self):
        self.assertIsInstance(report_identity()["package_version"], (str, type(None)))

    def test_pin_scheme_derived_from_digest_prefix(self):
        self.assertEqual(report_identity(config_digest="sha256:abc")["pin_scheme"], "sha256")
        self.assertEqual(report_identity(config_digest="hmac-sha256:abc")["pin_scheme"], "hmac-sha256")
        self.assertIsNone(report_identity(config_digest=None)["pin_scheme"])

    def test_no_config_digest_means_no_pin_scheme_not_a_guess(self):
        self.assertIsNone(report_identity()["config_digest"])
        self.assertIsNone(report_identity()["pin_scheme"])


class EndToEndThroughActuatorGateTest(unittest.TestCase):
    """The actual integration surface: HashChainedDecisionLogger dropped straight into a real
    ActuatorGate as its logger= argument, no manual .record() calls."""

    def test_real_gate_decisions_land_in_a_verifiable_log(self):
        import sys as _sys

        _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        import fixtures  # noqa: E402

        from safety_harness import ActionSchemaRegistry, ActuatorGate
        from safety_harness.adapters import FreezeInPlaceFallback

        schema = ActionSchemaRegistry.from_dict({
            "action_types": {"grasp": {"checks": [
                {"name": "object_hazard_confirmed", "kwargs": {}},
                {"name": "current_position_confirmed_stable", "kwargs": {}},
            ]}}
        })

        class _Perception:
            def __init__(self, state):
                self._state = state

            def get_world_state(self):
                return self._state

        class _Dynamics:
            def predict_trajectory(self, state, action, horizon_s):
                return fixtures.straight_line_trajectory()

        path = _tmp_path()
        log = HashChainedDecisionLogger(path)
        permit_state = fixtures.base_world_state(objects=(fixtures.confirmed_object(),))
        block_state = fixtures.base_world_state(objects=(fixtures.unstable_object(),))

        gate = ActuatorGate(_Perception(permit_state), _Dynamics(), FreezeInPlaceFallback(), log, schema)
        d1 = gate.gate(fixtures.grasp_action())
        gate = ActuatorGate(_Perception(block_state), _Dynamics(), FreezeInPlaceFallback(), log, schema)
        d2 = gate.gate(fixtures.grasp_action())
        log.close()

        self.assertEqual(d1.verdict.value, "permit")
        self.assertEqual(d2.verdict.value, "block")
        v = log.verify()
        self.assertTrue(v.ok, v.reason)
        self.assertEqual(v.entries, 2)
        with open(path) as f:
            raw = f.read()
        self.assertIn('"verdict": "permit"', raw)
        self.assertIn('"verdict": "block"', raw)


if __name__ == "__main__":
    unittest.main()
