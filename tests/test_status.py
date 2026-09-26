from stepcount import _status


def test_timed_status_reports_elapsed_time(monkeypatch, capsys):
    times = iter([10.0, 11.234])
    monkeypatch.setattr(_status.time, "perf_counter", lambda: next(times))

    with _status.timed_status("Loading model", verbose=True):
        pass

    assert capsys.readouterr().out == (
        "Loading model...\r"
        "Loading model... Done! (1.23s)\n"
    )


def test_timed_status_quiet_mode_suppresses_output(capsys):
    with _status.timed_status("Loading model", verbose=False):
        pass

    assert capsys.readouterr().out == ""
