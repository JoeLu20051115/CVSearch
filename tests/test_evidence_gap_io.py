import fcntl
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvsearch.evidence_gap.io import JsonlCheckpointWriter


class _CloseThenRaise:
    def __init__(self, handle, error):
        self._handle = handle
        self._error = error

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def close(self):
        self._handle.close()
        raise self._error


class JsonlCheckpointWriterTest(unittest.TestCase):
    def assert_closed_and_reopenable(self, writer, expected_ordinals):
        self.assertTrue(writer._closed)
        self.assertIsNone(writer._handle)
        self.assertIsNone(writer._lock_handle)
        reopened = JsonlCheckpointWriter(
            writer.final_path,
            expected_ordinals,
            resume=True,
            run_fingerprint=writer.run_fingerprint,
        )
        reopened.close()

    def test_crash_close_resume_and_finalize_preserve_expected_order(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            first = JsonlCheckpointWriter(final_path, (3, 7, 9))
            first.write(3, {"output": "a"})
            first.write(7, {"output": "b"})
            first.close()

            resumed = JsonlCheckpointWriter(final_path, (3, 7, 9), resume=True)
            self.assertEqual(resumed.completed, frozenset({3, 7}))
            with self.assertRaisesRegex(ValueError, "duplicate ordinal"):
                resumed.write(7, {"output": "duplicate"})
            resumed.write(9, {"output": "c"})
            resumed.finalize()

            self.assertFalse(Path(f"{final_path}.partial").exists())
            rows = [json.loads(line) for line in final_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["_eg_ordinal"] for row in rows], [3, 7, 9])
            self.assertEqual([row["output"] for row in rows], ["a", "b", "c"])

    def test_writer_rejects_bad_or_out_of_order_ordinals_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            for expected in ((True,), (-1,), (1.5,), (1, 1), ()):
                with self.subTest(expected=expected):
                    with self.assertRaises((TypeError, ValueError)):
                        JsonlCheckpointWriter(final_path, expected)

            writer = JsonlCheckpointWriter(final_path, (2, 4))
            for ordinal in (True, -1, 1.5, 5, 4):
                with self.subTest(ordinal=ordinal):
                    with self.assertRaises((TypeError, ValueError)):
                        writer.write(ordinal, {"output": "bad"})
            self.assertEqual(Path(f"{final_path}.partial").read_text(encoding="utf-8"), "")
            writer.close()

    def test_strict_json_and_reserved_ordinal_fail_without_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            for record in (
                {"value": math.nan},
                {"value": object()},
                {"_eg_ordinal": 0},
                {"_eg_run_fingerprint": "caller-owned"},
                {1: "numeric key", "1": "colliding string key"},
                ["not", "a", "mapping"],
            ):
                with self.subTest(record=record):
                    with self.assertRaises((TypeError, ValueError)):
                        writer.write(0, record)
            self.assertEqual(Path(f"{final_path}.partial").read_bytes(), b"")
            writer.write(0, {"nested": [1, 2]})
            writer.finalize()

    def test_finalize_requires_exact_completion_and_closed_writer_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0, 1))
            writer.write(0, {"output": 0})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                writer.finalize()
            writer.close()
            writer.close()
            with self.assertRaises(ValueError):
                writer.write(1, {"output": 1})
            self.assertFalse(final_path.exists())
            self.assertTrue(Path(f"{final_path}.partial").exists())

    def test_existing_final_or_partial_requires_explicit_resume_or_cli_force(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            final_path.write_text('{"_eg_ordinal":0}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                JsonlCheckpointWriter(final_path, (0,))
            final_path.unlink()

            partial_path = Path(f"{final_path}.partial")
            partial_path.write_text('{"_eg_ordinal":0}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                JsonlCheckpointWriter(final_path, (0,))

    def test_resume_truncates_only_a_non_newline_torn_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            partial_path = Path(f"{final_path}.partial")
            committed = (
                '{"output":"ok","_eg_ordinal":0,'
                '"_eg_run_fingerprint":"run-a"}\n'
            ).encode("utf-8")
            partial_path.write_bytes(committed + b'{"output":"torn","_eg_ordinal":1')

            writer = JsonlCheckpointWriter(
                final_path, (0, 1), resume=True, run_fingerprint="run-a"
            )
            self.assertEqual(writer.completed, frozenset({0}))
            self.assertEqual(partial_path.read_bytes(), committed)
            writer.write(1, {"output": "recomputed"})
            writer.finalize()

    def test_resume_rejects_newline_terminated_corruption_and_stale_fingerprint(self):
        corrupt_payloads = (
            b'not-json\n',
            b'{"_eg_ordinal":NaN,"_eg_run_fingerprint":"run-a"}\n',
            b'{"value":1e999,"_eg_ordinal":0,"_eg_run_fingerprint":"run-a"}\n',
            b'[0]\n',
            b'{"output":"missing ordinal","_eg_run_fingerprint":"run-a"}\n',
            b'{"_eg_ordinal":0}\n',
            b'{"_eg_ordinal":0,"_eg_run_fingerprint":"stale"}\n',
            b'{"_eg_ordinal":1,"_eg_run_fingerprint":"run-a"}\n',
            (
                b'{"_eg_ordinal":0,"_eg_run_fingerprint":"run-a"}\n'
                b'{"_eg_ordinal":0,"_eg_run_fingerprint":"run-a"}\n'
            ),
            (
                b'{"_eg_ordinal":0,"_eg_run_fingerprint":"run-a"}\n'
                b'{"_eg_ordinal":2,"_eg_run_fingerprint":"run-a"}\n'
            ),
        )
        for payload in corrupt_payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                final_path = Path(directory) / "answers.jsonl"
                Path(f"{final_path}.partial").write_bytes(payload)
                with self.assertRaises((TypeError, ValueError)):
                    JsonlCheckpointWriter(
                        final_path, (0, 1, 2), resume=True, run_fingerprint="run-a"
                    )

    def test_resume_rejects_ambiguous_final_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            final_path.write_text("{}\n", encoding="utf-8")
            Path(f"{final_path}.partial").write_text("", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                JsonlCheckpointWriter(final_path, (0,), resume=True)

    def test_matching_complete_final_is_an_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            with JsonlCheckpointWriter(
                final_path, (2,), run_fingerprint="run-a"
            ) as writer:
                writer.write(2, {"output": "done"})
                writer.finalize()
            before = final_path.read_bytes()

            resumed = JsonlCheckpointWriter(
                final_path, (2,), resume=True, run_fingerprint="run-a"
            )
            self.assertEqual(resumed.completed, frozenset({2}))
            resumed.finalize()
            with self.assertRaises(ValueError):
                resumed.write(2, {"output": "must not append"})
            self.assertEqual(final_path.read_bytes(), before)

    def test_explicit_replace_keeps_old_final_until_atomic_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            old_bytes = b'{"old":"complete"}\n'
            final_path.write_bytes(old_bytes)

            writer = JsonlCheckpointWriter(
                final_path,
                (0,),
                run_fingerprint="new-run",
                allow_replace=True,
            )
            writer.write(0, {"output": "new"})
            self.assertEqual(final_path.read_bytes(), old_bytes)
            writer.finalize()

            row = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertEqual(row["output"], "new")
            self.assertEqual(row["_eg_run_fingerprint"], "new-run")

    def test_explicit_replace_never_discards_a_partial_or_combines_with_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            partial_path = Path(f"{final_path}.partial")
            partial_path.write_bytes(b"pending\n")
            with self.assertRaises(FileExistsError):
                JsonlCheckpointWriter(final_path, (0,), allow_replace=True)
            self.assertEqual(partial_path.read_bytes(), b"pending\n")

            partial_path.unlink()
            for allow_replace in (1, "yes"):
                with self.subTest(allow_replace=allow_replace):
                    with self.assertRaises(TypeError):
                        JsonlCheckpointWriter(
                            final_path, (0,), allow_replace=allow_replace
                        )
            with self.assertRaises(ValueError):
                JsonlCheckpointWriter(
                    final_path, (0,), resume=True, allow_replace=True
                )

    def test_concurrent_writer_for_same_target_fails_nonblocking(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            first = JsonlCheckpointWriter(final_path, (0,))
            try:
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    JsonlCheckpointWriter(final_path, (0,), resume=True)
            finally:
                first.close()

    def test_write_and_finalize_use_durable_ordering(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            events = []

            def note_fsync(_fd):
                events.append("fsync")

            real_replace = __import__("os").replace

            def note_replace(source, destination):
                events.append("replace")
                real_replace(source, destination)

            with mock.patch("cvsearch.evidence_gap.io.os.fsync", side_effect=note_fsync), mock.patch(
                "cvsearch.evidence_gap.io.os.replace", side_effect=note_replace
            ):
                writer = JsonlCheckpointWriter(final_path, (0,))
                writer.write(0, {"output": "ok"})
                writer.finalize()

            self.assertEqual(events, ["fsync", "replace", "fsync"])

    def test_close_failure_still_closes_state_and_releases_checkpoint_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            primary = OSError("data close failed")
            writer._handle = _CloseThenRaise(writer._handle, primary)

            with self.assertRaises(OSError) as caught:
                writer.close()

            self.assertIs(caught.exception, primary)
            self.assertFalse(final_path.exists())
            self.assertTrue(writer.partial_path.exists())
            self.assert_closed_and_reopenable(writer, (0,))

    def test_data_close_failure_wins_over_raw_descriptor_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            primary = OSError("data close failed")
            cleanup = RuntimeError("raw descriptor cleanup failed")
            writer._handle = _CloseThenRaise(writer._handle, primary)

            with mock.patch("cvsearch.evidence_gap.io.os.close", side_effect=cleanup):
                with self.assertRaises(OSError) as caught:
                    writer.close()

            self.assertIs(caught.exception, primary)
            self.assertTrue(any("raw descriptor cleanup failed" in note for note in primary.__notes__))
            self.assert_closed_and_reopenable(writer, (0,))

    def test_context_cleanup_failure_does_not_mask_body_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            cleanup = OSError("data close failed")
            primary = RuntimeError("body failed")
            writer._handle = _CloseThenRaise(writer._handle, cleanup)

            with self.assertRaises(RuntimeError) as caught:
                with writer:
                    raise primary

            self.assertIs(caught.exception, primary)
            self.assertTrue(any("data close failed" in note for note in primary.__notes__))
            self.assertFalse(final_path.exists())
            self.assertTrue(writer.partial_path.exists())
            self.assert_closed_and_reopenable(writer, (0,))

    def test_finalize_data_close_failure_keeps_partial_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            writer.write(0, {"output": "ready"})
            primary = OSError("data close failed")
            writer._handle = _CloseThenRaise(writer._handle, primary)

            with self.assertRaises(OSError) as caught:
                writer.finalize()

            self.assertIs(caught.exception, primary)
            self.assertFalse(final_path.exists())
            self.assertTrue(writer.partial_path.exists())
            self.assertFalse(writer._finalized)
            self.assert_closed_and_reopenable(writer, (0,))

    def test_replace_failure_wins_over_lock_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            writer.write(0, {"output": "ready"})
            primary = OSError("replace failed")
            cleanup = OSError("unlock failed")
            real_flock = fcntl.flock

            def fail_unlock(fd, operation):
                if operation == fcntl.LOCK_UN:
                    raise cleanup
                return real_flock(fd, operation)

            with mock.patch("cvsearch.evidence_gap.io.os.replace", side_effect=primary), mock.patch(
                "cvsearch.evidence_gap.io.fcntl.flock", side_effect=fail_unlock
            ):
                with self.assertRaises(OSError) as caught:
                    writer.finalize()

            self.assertIs(caught.exception, primary)
            self.assertTrue(any("unlock failed" in note for note in primary.__notes__))
            self.assertFalse(final_path.exists())
            self.assertTrue(writer.partial_path.exists())
            self.assertFalse(writer._finalized)
            self.assert_closed_and_reopenable(writer, (0,))

    def test_directory_open_failure_wins_over_lock_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            writer.write(0, {"output": "ready"})
            primary = OSError("directory open failed")
            cleanup = OSError("unlock failed")
            real_flock = fcntl.flock

            def fail_unlock(fd, operation):
                if operation == fcntl.LOCK_UN:
                    raise cleanup
                return real_flock(fd, operation)

            with mock.patch("cvsearch.evidence_gap.io.os.open", side_effect=primary), mock.patch(
                "cvsearch.evidence_gap.io.fcntl.flock", side_effect=fail_unlock
            ):
                with self.assertRaises(OSError) as caught:
                    writer.finalize()

            self.assertIs(caught.exception, primary)
            self.assertTrue(any("unlock failed" in note for note in primary.__notes__))
            self.assertTrue(final_path.exists())
            self.assertFalse(writer.partial_path.exists())
            self.assertTrue(writer._finalized)
            self.assert_closed_and_reopenable(writer, (0,))

    def test_directory_fsync_failure_wins_over_directory_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            writer.write(0, {"output": "ready"})
            primary = OSError("directory fsync failed")
            cleanup = OSError("directory close failed")
            real_close = os.close

            def close_then_fail(fd):
                real_close(fd)
                raise cleanup

            with mock.patch("cvsearch.evidence_gap.io.os.fsync", side_effect=primary), mock.patch(
                "cvsearch.evidence_gap.io.os.close", side_effect=close_then_fail
            ):
                with self.assertRaises(OSError) as caught:
                    writer.finalize()

            self.assertIs(caught.exception, primary)
            self.assertTrue(any("directory close failed" in note for note in primary.__notes__))
            self.assertTrue(final_path.exists())
            self.assertFalse(writer.partial_path.exists())
            self.assertTrue(writer._finalized)
            self.assert_closed_and_reopenable(writer, (0,))

    def test_directory_close_failure_wins_over_lock_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            writer.write(0, {"output": "ready"})
            primary = OSError("directory close failed")
            cleanup = OSError("unlock failed")
            real_close = os.close
            real_flock = fcntl.flock

            def close_then_fail(fd):
                real_close(fd)
                raise primary

            def fail_unlock(fd, operation):
                if operation == fcntl.LOCK_UN:
                    raise cleanup
                return real_flock(fd, operation)

            with mock.patch("cvsearch.evidence_gap.io.os.close", side_effect=close_then_fail), mock.patch(
                "cvsearch.evidence_gap.io.fcntl.flock", side_effect=fail_unlock
            ):
                with self.assertRaises(OSError) as caught:
                    writer.finalize()

            self.assertIs(caught.exception, primary)
            self.assertTrue(any("unlock failed" in note for note in primary.__notes__))
            self.assertTrue(final_path.exists())
            self.assertFalse(writer.partial_path.exists())
            self.assertTrue(writer._finalized)
            self.assert_closed_and_reopenable(writer, (0,))

    def test_unlock_failure_wins_over_lock_handle_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            writer = JsonlCheckpointWriter(final_path, (0,))
            writer.write(0, {"output": "ready"})
            primary = OSError("unlock failed")
            cleanup = OSError("lock handle close failed")
            real_flock = fcntl.flock
            writer._lock_handle = _CloseThenRaise(writer._lock_handle, cleanup)

            def fail_unlock(fd, operation):
                if operation == fcntl.LOCK_UN:
                    raise primary
                return real_flock(fd, operation)

            with mock.patch("cvsearch.evidence_gap.io.fcntl.flock", side_effect=fail_unlock):
                with self.assertRaises(OSError) as caught:
                    writer.finalize()

            self.assertIs(caught.exception, primary)
            self.assertTrue(any("lock handle close failed" in note for note in primary.__notes__))
            self.assertTrue(final_path.exists())
            self.assertFalse(writer.partial_path.exists())
            self.assertTrue(writer._finalized)
            self.assert_closed_and_reopenable(writer, (0,))

    def test_fingerprint_validation_and_record_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "answers.jsonl"
            for fingerprint in ("", True, 1):
                with self.subTest(fingerprint=fingerprint):
                    with self.assertRaises((TypeError, ValueError)):
                        JsonlCheckpointWriter(final_path, (0,), run_fingerprint=fingerprint)

            writer = JsonlCheckpointWriter(
                final_path, (0,), run_fingerprint="reproducible-run"
            )
            writer.write(0, {"output": "ok"})
            writer.finalize()
            row = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertEqual(row["_eg_run_fingerprint"], "reproducible-run")

    def test_resume_rejects_every_corrupt_or_nonprefix_partial(self):
        corrupt_payloads = (
            b'{"_eg_ordinal":0}\nnot-json\n',
            b'{"_eg_ordinal":NaN}\n',
            b'[0]\n',
            b'{"output":"missing"}\n',
            b'{"_eg_ordinal":1}\n',
            b'{"_eg_ordinal":0}\n{"_eg_ordinal":0}\n',
            b'{"_eg_ordinal":0}\n{"_eg_ordinal":2}\n',
        )
        for payload in corrupt_payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                final_path = Path(directory) / "answers.jsonl"
                Path(f"{final_path}.partial").write_bytes(payload)
                with self.assertRaises((TypeError, ValueError)):
                    JsonlCheckpointWriter(final_path, (0, 1, 2), resume=True)


if __name__ == "__main__":
    unittest.main()
