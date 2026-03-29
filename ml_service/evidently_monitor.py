from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from collections import deque
from collections.abc import Callable

import pandas as pd


LOGGER = logging.getLogger(__name__)


class EvidentlyMonitor:
    """
    In-memory buffer for drift monitoring and periodic Evidently report uploads.
    """

    def __init__(
        self,
        *,
        workspace_url: str,
        project_id: str,
        report_interval_seconds: int,
        reference_size: int,
        current_size: int,
        workspace_factory: Callable[[str], object] | None = None,
        report_factory: Callable[[], object] | None = None,
    ) -> None:
        self.workspace_url = workspace_url
        self.project_id = project_id
        self.report_interval_seconds = report_interval_seconds
        self.reference_size = reference_size
        self.current_size = current_size
        self.workspace_factory = workspace_factory
        self.report_factory = report_factory
        self.lock = threading.RLock()
        self.reference_rows: deque[dict[str, object]] = deque(maxlen=reference_size)
        self.current_rows: deque[dict[str, object]] = deque(maxlen=current_size)

    def reset(self) -> None:
        with self.lock:
            self.reference_rows.clear()
            self.current_rows.clear()

    def record(
        self,
        *,
        features: dict[str, object],
        prediction: int,
        probability: float,
    ) -> None:
        row = dict(features)
        row['prediction'] = prediction
        row['probability'] = probability

        with self.lock:
            if len(self.reference_rows) < self.reference_size:
                self.reference_rows.append(row)
            else:
                self.current_rows.append(row)

    def has_enough_data(self) -> bool:
        with self.lock:
            return (
                len(self.reference_rows) >= self.reference_size
                and len(self.current_rows) >= self.current_size
            )

    def _take_snapshot(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        with self.lock:
            if not self.has_enough_data():
                return None

            reference = pd.DataFrame(list(self.reference_rows))
            current = pd.DataFrame(list(self.current_rows))
            return reference, current

    def build_and_send_report(self) -> bool:
        snapshot = self._take_snapshot()
        if snapshot is None:
            return False

        reference_data, current_data = snapshot

        workspace_factory = self.workspace_factory or _default_workspace_factory
        report_factory = self.report_factory or _default_report_factory

        report = report_factory()
        # Evidently may trigger scipy runtime warnings on tiny or near-constant
        # windows; the report is still produced successfully, so we keep logs clean.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='divide by zero encountered in divide',
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message='invalid value encountered in divide',
                category=RuntimeWarning,
            )
            result = report.run(
                reference_data=reference_data,
                current_data=current_data,
            )

        workspace = workspace_factory(self.workspace_url)
        workspace.add_run(self.project_id, result)
        with self.lock:
            self.current_rows.clear()
        return True

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                self.build_and_send_report()
            except Exception:
                LOGGER.exception('Failed to build or send Evidently drift report')

            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.report_interval_seconds,
                )
            except TimeoutError:
                continue


def _default_report_factory():
    from evidently import Report
    from evidently.presets import DataDriftPreset

    return Report(metrics=[DataDriftPreset()])


def _default_workspace_factory(workspace_url: str):
    from evidently.ui.workspace import RemoteWorkspace

    return RemoteWorkspace(workspace_url)
