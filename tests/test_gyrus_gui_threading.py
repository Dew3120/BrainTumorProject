import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread

from gyrus.gui.main_window import GyrusMainWindow
from gyrus.gui.state import AppState
from gyrus.gui.work_controller import WorkerBridge

ROOT = Path(__file__).resolve().parents[1]


def test_qt_bridge_runs_real_worker_off_ui_thread(qtbot):
    bridge = WorkerBridge(ROOT / "runs/final_stack", "cpu")
    main_thread = QThread.currentThread()
    try:
        with qtbot.waitSignal(bridge.verified, timeout=60_000) as ready_signal:
            bridge.start()
        assert bridge.ready
        assert bridge.controller_thread is not main_thread
        assert bridge.controller_thread.isRunning()
        assert ready_signal.args[0].device == "cpu"
        assert ready_signal.args[0].artifact_count == 31
        window = GyrusMainWindow()
        qtbot.addWidget(window)
        assert window.on_worker_ready(ready_signal.args[0]) is True
        assert window.machine.state == AppState.READY_EMPTY

        with qtbot.waitSignal(bridge.job_finished, timeout=60_000) as job_signal:
            bridge.submit_image(
                ROOT / "runs/final_stack/verification/fixtures/elliptic.png"
            )
        response = job_signal.args[0]
        assert response.result["classification"]["predicted_category"] == "meningioma"
        assert response.events[-1].disclosure_allowed is True
        assert all(not event.disclosure_allowed for event in response.events[:-1])
    finally:
        bridge.close()
