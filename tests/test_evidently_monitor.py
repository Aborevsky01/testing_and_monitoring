import asyncio
import unittest

from ml_service.evidently_monitor import EvidentlyMonitor


class FakeReport:
    def __init__(self) -> None:
        self.calls = []

    def run(self, *, reference_data, current_data):
        self.calls.append((reference_data, current_data))
        return {'status': 'ok'}


class FakeWorkspace:
    def __init__(self) -> None:
        self.runs = []

    def add_run(self, project_id, result) -> None:
        self.runs.append((project_id, result))


class EvidentlyMonitorTests(unittest.TestCase):
    def test_records_reference_then_current_rows(self) -> None:
        monitor = EvidentlyMonitor(
            workspace_url='http://example.com',
            project_id='project-id',
            report_interval_seconds=60,
            reference_size=2,
            current_size=2,
        )

        monitor.record(features={'race': 'White'}, prediction=0, probability=0.2)
        monitor.record(features={'race': 'Black'}, prediction=1, probability=0.8)
        monitor.record(features={'race': 'Asian-Pac-Islander'}, prediction=1, probability=0.9)

        self.assertEqual(len(monitor.reference_rows), 2)
        self.assertEqual(len(monitor.current_rows), 1)

    def test_build_and_send_report_uploads_snapshot_and_clears_current(self) -> None:
        report = FakeReport()
        workspace = FakeWorkspace()
        monitor = EvidentlyMonitor(
            workspace_url='http://example.com',
            project_id='project-id',
            report_interval_seconds=60,
            reference_size=2,
            current_size=2,
            report_factory=lambda: report,
            workspace_factory=lambda _: workspace,
        )

        monitor.record(features={'race': 'White'}, prediction=0, probability=0.2)
        monitor.record(features={'race': 'Black'}, prediction=1, probability=0.8)
        monitor.record(features={'race': 'White'}, prediction=1, probability=0.9)
        monitor.record(features={'race': 'Black'}, prediction=0, probability=0.1)

        sent = monitor.build_and_send_report()

        self.assertTrue(sent)
        self.assertEqual(len(report.calls), 1)
        self.assertEqual(len(workspace.runs), 1)
        self.assertEqual(len(monitor.current_rows), 0)

    def test_build_and_send_report_returns_false_without_enough_data(self) -> None:
        monitor = EvidentlyMonitor(
            workspace_url='http://example.com',
            project_id='project-id',
            report_interval_seconds=60,
            reference_size=2,
            current_size=2,
        )

        monitor.record(features={'race': 'White'}, prediction=0, probability=0.2)

        self.assertFalse(monitor.build_and_send_report())

    def test_run_forever_attempts_report_generation(self) -> None:
        monitor = EvidentlyMonitor(
            workspace_url='http://example.com',
            project_id='project-id',
            report_interval_seconds=1,
            reference_size=1,
            current_size=1,
        )

        calls = []

        def fake_build_and_send_report() -> bool:
            calls.append('called')
            return False

        monitor.build_and_send_report = fake_build_and_send_report  # type: ignore[method-assign]

        async def exercise() -> None:
            stop_event = asyncio.Event()
            task = asyncio.create_task(monitor.run_forever(stop_event))
            await asyncio.sleep(0.05)
            stop_event.set()
            await task

        asyncio.run(exercise())

        self.assertGreaterEqual(len(calls), 1)
