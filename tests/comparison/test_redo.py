"""Tests for `pfs.drp.qa.comparison.redo` that need neither the opdb nor a Butler."""

import datetime
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from pfs.drp.qa.comparison.redo import fetch, fetchSummary


def testFetch(periods, visit, listing, FakeOpDB):
    answer = listing(
        visit(1, "2026-01-03 18:00", "scienceArc"),
        visit(2, "2026-01-03 20:00", "scienceObject", design=10),
    )
    opdb = FakeOpDB(
        {
            "FROM pfs_visit": answer,
            "obslog_visit_set_note": pd.DataFrame({"iic_sequence_id": [1], "note": ["fine"]}),
            "obslog_visit_note": pd.DataFrame(
                columns=["pfs_visit_id", "camera", "data_flag", "note", "source"]
            ),
            "FROM tel_status": pd.DataFrame(
                {"pfs_visit_id": [2], "n_status": [9], "focus_offset_max": [3.2]}
            ),
            "pfs_design_fiber": pd.DataFrame(
                {"pfs_design_id": [10], "proposal_id": ["S25A-123QF"], "n_fibers": [5]}
            ),
        }
    )
    period = periods["run1"]
    now = datetime.datetime(2026, 1, 4, 9)  # during the run: read up to now
    frames, readUntil = fetch(opdb, period, now=now)

    assert opdb.calls[0].params == {"start": period.start, "end": now}
    telCall = next(call for call in opdb.calls if "tel_status" in call.sql)
    assert telCall.params == {"visits": [2]}  # sky visits only
    assert set(frames) == {"listing", "notes", "telStatus", "designs"}
    assert frames["listing"]["pfs_visit_id"].tolist() == [1, 2]
    assert frames["designs"]["category"].tolist() == ["QF"]
    assert "proposal_id" not in frames["designs"]  # proposal IDs stay out of what the package returns
    assert readUntil == str(now)


def testFetchSummary(periods, visit, listing, FakeOpDB):
    answer = listing(
        visit(1, "2026-01-03 18:00", "scienceArc", sequence=10),
        visit(2, "2026-01-03 20:00", "scienceObject", design=10),
        visit(3, "2026-01-04 15:00", "darks", sequence=11),
    )
    opdb = FakeOpDB(
        {
            "FROM pfs_visit": answer,
            "obslog_visit_set_note": pd.DataFrame({"iic_sequence_id": [10], "note": ["fine"]}),
            "obslog_visit_note": pd.DataFrame(
                columns=["pfs_visit_id", "camera", "data_flag", "note", "source"]
            ),
            "FROM tel_status": pd.DataFrame(
                {"pfs_visit_id": [2], "n_status": [9], "focus_offset_max": [3.2]}
            ),
            "pfs_design_fiber": pd.DataFrame(
                {"pfs_design_id": [10], "proposal_id": ["S25A-123QF"], "n_fibers": [5]}
            ),
        }
    )
    frames, readUntil = fetch(opdb, periods["run1"], now=datetime.datetime(2026, 2, 1))
    text = fetchSummary(frames, readUntil, periods["run1"], periods)
    assert "3 visits in 3 sequences, on 2 nights" in text
    assert (
        "1 gated" in text
        and "1 unvalidated" in text
        and "1  calibration  scienceArc single" in text
        and "1  science      scienceObject" in text
    )
    assert "1  no method for darks" in text
    assert "sky: 1 visits (1 with telescope status), 1 science" in text
    assert "notes: 1 (1 on sequences)" in text


def testClassifiedVisits(periods, visit, listing, FakeOpDB):
    from pfs.drp.qa.comparison.redo import classifiedVisits

    opdb = FakeOpDB(
        {
            "FROM pfs_visit": listing(visit(1, "2026-01-03 18:00", "scienceArc", sequence=10)),
            "obslog_visit_set_note": pd.DataFrame(columns=["iic_sequence_id", "note"]),
            "obslog_visit_note": pd.DataFrame(
                columns=["pfs_visit_id", "camera", "data_flag", "note", "source"]
            ),
            "FROM tel_status": pd.DataFrame(columns=["pfs_visit_id", "n_status", "focus_offset_max"]),
            "pfs_design_fiber": pd.DataFrame(columns=["pfs_design_id", "proposal_id", "n_fibers"]),
        }
    )
    frames, _ = fetch(opdb, periods["run1"], now=datetime.datetime(2026, 2, 1))
    visits = classifiedVisits(frames, periods["run1"], periods)
    assert visits["pfs_visit_id"].tolist() == [1]
    assert (visits["period"] == "run1").all()


def testStageSelection():
    from pfs.drp.qa.comparison.redo import STAGES, _parser, _stages

    args = _parser().parse_args(["run31", "--only", "process", "--only", "load,judge"])
    assert _stages(args.only) == {"process", "load", "judge"}
    assert _stages([]) == set(STAGES)  # default: every stage
    with pytest.raises(SystemExit, match="unknown stage"):
        _stages(["bogus"])


def testRunIsPositionalAndPrefixDefaultsToRedo():
    from pfs.drp.qa.comparison.redo import _parser

    args = _parser().parse_args(["run31"])
    assert args.run == "run31"
    assert args.prefix.endswith("/redo")
    assert args.previous_collection == "qaActor/reductions"
    # Negative control: the old fetch/plan/run subcommands are gone.
    with pytest.raises(SystemExit):
        _parser().parse_args(["fetch", "--period", "run31"])


def testArmAndSpectrographSelection():
    from pfs.drp.qa.comparison.redo import _parser

    args = _parser().parse_args(["run31", "--arm", "b", "r", "m", "--spectrograph", "1", "2"])
    assert args.arm == ["b", "r", "m"]
    assert args.spectrograph == [1, 2]
    assert _parser().parse_args(["run31"]).arm is None  # every arm by default
    # Negative control: an unknown arm is refused, not silently processed.
    with pytest.raises(SystemExit):
        _parser().parse_args(["run31", "--arm", "x"])


def testCalibRangesAreTheSetCadenceGatedVisits():
    from pfs.drp.qa.comparison.redo import _calibRanges

    visits = pd.DataFrame(
        {
            "pfs_visit_id": [10, 11, 12, 20, 30],
            "validated": [True, True, True, True, False],
            "cadence": ["set", "set", "set", "daily", "set"],
        }
    )
    # The set-cadence gated visits, as contiguous ranges; the daily and the unvalidated are left out.
    assert _calibRanges(SimpleNamespace(calib_visits=None), visits) == [(10, 12)]
    # --calib-visits overrides.
    assert _calibRanges(SimpleNamespace(calib_visits=[5, 9]), visits) == [(5, 9)]


def testQgraphFromRun():
    from pfs.drp.qa.comparison.redo import _qgraph

    command = [
        "pipetask",
        "--long-log",
        "run",
        "-j",
        "8",
        "--register-dataset-types",
        "-b",
        "/repo",
        "-d",
        "v",
    ]
    # qgraph does not build or write, so it takes neither -j nor --register-dataset-types.
    assert _qgraph(command) == ["pipetask", "--long-log", "qgraph", "-b", "/repo", "-d", "v"]


def testPipetaskFailures():
    from pfs.drp.qa.comparison.redo import pipetaskFailures

    log = "\n".join(
        [
            "INFO 2026-10-08T07:53:20 lsst.ctrl.mpexec ... Executing 40 quanta",
            "ERROR 2026-10-08T07:54:01 lsst.ctrl.mpexec ... Task <reduceExposure dataId={visit: 148144}> failed",
            "ERROR 2026-10-08T07:54:01 lsst.ctrl.mpexec ... Task <reduceExposure dataId={visit: 148144}> failed",
            "INFO 2026-10-08T07:54:09 lsst.ctrl.mpexec ... Executed 30 quanta successfully, 1 failed and 9 remain"
            " out of total 40 quanta.",
        ]
    )
    summary, lines = pipetaskFailures(log)
    assert summary == "Executed 30 quanta successfully, 1 failed and 9 remain out of total 40 quanta."
    assert len(lines) == 1 and "148144" in lines[0]  # repeated lines once
    traceback = (
        "ERROR ... Caught an exception\nTraceback:\n  File x.py\nValueError: Output CHAINED collection"
    )
    assert pipetaskFailures(traceback)[1][-1] == "ValueError: Output CHAINED collection"
    assert pipetaskFailures("INFO nothing to see") == (None, [])
    many = "\n".join(f"ERROR quantum {i} failed" for i in range(50))
    assert pipetaskFailures(many, limit=5)[1][0] == "ERROR quantum 45 failed"  # the last ones


def testResolveCollections():
    from pfs.drp.qa.comparison.redo import resolveCollections

    assert resolveCollections(False, "PFS/defaults", None, None) == (
        ["drpActor/reductions"],
        ["drpActor/reductions", "PFS/defaults"],
    )
    assert resolveCollections(True, "PFS/defaults", None, None) == ([], ["PFS/defaults"])
    # Negative control: --fresh with drpActor's reductions among the inputs would reuse them.
    with pytest.raises(ValueError, match="--fresh"):
        resolveCollections(True, "PFS/defaults", None, ["drpActor/reductions", "PFS/defaults"])
    with pytest.raises(ValueError, match="--fresh"):
        resolveCollections(True, "PFS/defaults", ["drpActor/reductions"], None)


def testPipelineVersion():
    from pfs.drp.qa.comparison.redo import pipelineVersion

    weekly = {"DATAMODEL": "w.2026.40", "OBS_PFS": "w.2026.40", "DRP_STELLA": "w.2026.40"}
    assert pipelineVersion(weekly) == "pfs-w.2026.40"
    mixed = weekly | {"HIERARCH OBS_PFS": "w.2026.41-2-gabc"}
    del mixed["OBS_PFS"]
    assert pipelineVersion(mixed) == "datamodel-w.2026.40+drp_stella-w.2026.40+obs_pfs-w.2026.41-2-gabc"
    with pytest.raises(RuntimeError, match="Unknown"):
        pipelineVersion(weekly | {"DRP_STELLA": "unknown"})


def testOnlyTheReductionConfigCountsTowardsTheVersion():
    from pathlib import Path

    from pfs.drp.qa.comparison.redo import RESULT_NEUTRAL_PATHS

    package = Path(__file__).parents[2] / "python" / "pfs" / "drp" / "qa" / "comparison"
    neutral = {
        Path(path).name for path in RESULT_NEUTRAL_PATHS if path.startswith("python/pfs/drp/qa/comparison/")
    }
    modules = {path.name for path in package.iterdir() if path.name != "__pycache__"}
    assert modules - neutral == {"reduction.py"}  # a new module counts until it is listed


def testUnexpectedPipeline():
    """A fresh run that would reduce everything again under a new label is called out."""
    from pfs.drp.qa.comparison.redo import unexpectedPipeline

    warning = unexpectedPipeline("pfs-w.2026.39", [], ["datamodel-w.2026.40+drp_stella-w.2026.41"], 2152)
    assert warning and "2152 detector images" in warning and "'pfs-w.2026.39'" in warning
    assert "'datamodel-w.2026.40+drp_stella-w.2026.41' has" in warning
    # Not amiss: the label already has reductions, nothing to reduce, no other label, or not fresh.
    assert unexpectedPipeline("pfs-w.2026.39", ["u/me/redo/run25/v1/pfs-w.2026.39"], ["other"], 2152) is None
    assert unexpectedPipeline("pfs-w.2026.39", [], ["other"], 0) is None
    assert unexpectedPipeline("pfs-w.2026.39", [], [], 2152) is None
    assert unexpectedPipeline("", [], ["other"], 2152) is None


@pytest.fixture
def planned(tmp_path, periods, visit, listing, FakeOpDB, monkeypatch):
    """Stand in for the opdb, the Butler and pipetask for a run of one arc sequence and one trace.

    Returns ``main``'s arguments (``--only process``, reusing drpActor's reductions so no fresh
    pipeline version is read), the holdings to fill (one row per detector image: visit, arm,
    spectrograph, raw, reduced, judged) and the pipetask commands run.
    """
    from pfs.drp.qa.comparison import butlerQueries, redo

    answer = listing(
        visit(1, "2026-01-03 18:00", "scienceArc", sequence=50, cameras="b1,r1"),
        visit(2, "2026-01-03 18:01", "scienceArc", sequence=50, cameras="b1,r1"),
        visit(3, "2026-01-03 18:05", "scienceTrace", sequence=51, cameras="b1,r1"),
    )
    opdb = FakeOpDB(
        {
            "FROM pfs_visit": answer,
            "obslog_visit_set_note": pd.DataFrame(columns=["iic_sequence_id", "note"]),
            "obslog_visit_note": pd.DataFrame(
                columns=["pfs_visit_id", "camera", "data_flag", "note", "source"]
            ),
            "FROM tel_status": pd.DataFrame(columns=["pfs_visit_id", "n_status", "focus_offset_max"]),
            "pfs_design_fiber": pd.DataFrame(columns=["pfs_design_id", "proposal_id", "n_fibers"]),
        }
    )
    monkeypatch.setattr(redo, "_opdb", lambda args: opdb)

    holdings = []
    butlerModule = ModuleType("lsst.daf.butler")
    butlerModule.Butler = lambda repo, writeable=False: SimpleNamespace(repo=repo, writeable=writeable)
    for name, module in (
        ("lsst", ModuleType("lsst")),
        ("lsst.daf", ModuleType("lsst.daf")),
        ("lsst.daf.butler", butlerModule),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(butlerQueries, "collectionExists", lambda butler, name: False)
    monkeypatch.setattr(butlerQueries, "earlierRuns", lambda *args: [])
    monkeypatch.setattr(
        butlerQueries,
        "detectorHoldings",
        lambda butler, visits, **kwargs: pd.DataFrame(
            holdings, columns=["visit", "arm", "spectrograph", "raw", "reduced", "judged"]
        ),
    )
    commands = []

    def pipetask(command, stdout=None, stderr=None):
        commands.append(command)
        stdout.write("INFO lsst.ctrl.mpexec ... Executed 4 quanta successfully, 0 failed\n")
        stdout.flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(redo.subprocess, "run", pipetask)
    arguments = [
        "run1",
        "--only",
        "process",
        "--reductions",
        "drpActor/reductions",
        "--runs",
        str(tmp_path / "runs.yaml"),
        "--work-dir",
        str(tmp_path),
        "--version",
        "v1",
        "--prefix",
        "u/me/redo",
    ]
    return SimpleNamespace(arguments=arguments, holdings=holdings, commands=commands, directory=tmp_path)


def testProcessReducesThenMeasures(planned, capsys):
    from pfs.drp.qa.comparison.redo import main

    # The arcs are reduced but not measured; the trace has only raw data.
    planned.holdings += [(visit, arm, 1, True, visit != 3, False) for visit in (1, 2, 3) for arm in "br"]
    assert main(planned.arguments) == 0
    # The reduced arcs are measured alone; the trace is reduced then measured: three pipetask commands.
    measureJudge, reduce, measure = planned.commands
    measureSubset = "drpQA.yaml#imageQualityQa,dmLineResiduals,reductionStats"
    assert measureJudge[measureJudge.index("-p") + 1].endswith(measureSubset)
    assert reduce[reduce.index("-p") + 1].endswith("reduceExposure.yaml#isr,cosmicray,reduceExposure")
    assert measure[measure.index("-p") + 1].endswith(measureSubset)
    assert measureJudge[-1] == "instrument = 'PFS' AND visit IN (1..2)"
    assert reduce[-1] == "instrument = 'PFS' AND visit IN (3)"
    assert measure[-1] == "instrument = 'PFS' AND visit IN (3)"
    assert "-C" in reduce and "-C" not in measure  # only the reduce phase configures cosmicray
    assert "--rebase" not in measureJudge and "--rebase" in reduce and "--rebase" in measure
    assert "calibration: Executed 4 quanta successfully, 0 failed" in capsys.readouterr().out
    # The collection is under the redo prefix; no "qa" sub-path and no variant when reductions are reused.
    workDir = planned.directory / "run1" / "v1"
    names = sorted(path.name for path in workDir.iterdir())
    assert [name for name in names if not name.startswith("pipetask-")] == ["cosmicray-run1-calibration.py"]
    assert len([name for name in names if name.startswith("pipetask-run1-")]) == 3
    # Negative control: no parquet, no report files are written.
    assert not [name for name in names if name.endswith((".parquet", ".html", ".csv"))]


def testProcessRestrictsToSelectedArms(planned, capsys):
    from pfs.drp.qa.comparison.redo import main

    # Visits 1-3 each exposed b and r; the arcs are reduced, the trace is not.
    planned.holdings += [(visit, arm, 1, True, visit != 3, False) for visit in (1, 2, 3) for arm in "br"]
    assert main([*planned.arguments, "--arm", "b"]) == 0
    printed = capsys.readouterr().out
    # The coverage itself is filtered to arm b: 3 b images (2 reduced, 1 not), not the 6 b+r images.
    assert "detector images: 0 judged, 2 to judge, 1 to reduce" in printed
    assert planned.commands  # something ran
    # And every pipetask command's data query is restricted to arm b, so the r (and any n) images are skipped.
    for command in planned.commands:
        assert "AND arm IN ('b')" in command[-1]


def testProcessDryRunBuildsGraphs(planned, capsys):
    from pfs.drp.qa.comparison.redo import main

    planned.holdings += [(visit, arm, 1, True, visit != 3, False) for visit in (1, 2, 3) for arm in "br"]
    assert main([*planned.arguments, "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "output collection: u/me/redo/run1/v1" in printed
    assert "# calibration-judge (scienceArc): 2 visits" in printed
    # Negative control: a dry run builds each graph (qgraph), it does not run the pipeline.
    assert planned.commands and all(
        "qgraph" in command and "run" not in command for command in planned.commands
    )


def testProcessReportsAFailedPass(planned, monkeypatch, capsys):
    from pfs.drp.qa.comparison import redo

    planned.holdings += [(visit, arm, 1, True, True, False) for visit in (1, 2, 3) for arm in "br"]
    monkeypatch.setattr(
        redo.subprocess, "run", lambda command, stdout=None, stderr=None: SimpleNamespace(returncode=1)
    )
    assert redo.main(planned.arguments) == 1
    assert "1 of 1 passes failed: calibration-judge" in capsys.readouterr().err
    # Negative control: with nothing left to do, the process stage does nothing and succeeds.
    planned.holdings[:] = [(visit, arm, 1, True, True, True) for visit in (1, 2, 3) for arm in "br"]
    assert redo.main(planned.arguments) == 0
    assert "Nothing to do" in capsys.readouterr().out


def testLaterStagesRunInOrder(planned, monkeypatch):
    """With the process stage skipped, redo loads, makes values and judges, then points at the report."""
    from pfs.drp.qa.comparison import redo

    called = []
    monkeypatch.setattr(redo, "_qadb", lambda args: "QADB")
    monkeypatch.setattr(redo, "_load", lambda *a, **k: called.append("load"))
    monkeypatch.setattr(redo, "_values", lambda *a, **k: called.append("values"))
    monkeypatch.setattr(redo, "_judge", lambda *a, **k: called.append("judge"))
    arguments = planned.arguments[:]
    arguments[arguments.index("process")] = "load,values,judge"  # skip process: no Butler needed
    planned.holdings += [(1, "b", 1, True, True, True)]
    assert redo.main(arguments) == 0
    assert called == ["load", "values", "judge"]


def testProcessFailureDoesNotBlockLaterStages(planned, monkeypatch):
    """A failed process pass (an engineering frame, say) still loads, values and judges what succeeded."""
    from pfs.drp.qa.comparison import redo

    called = []
    monkeypatch.setattr(redo, "_plan", lambda *a, **k: 1)  # a pass reported a failed quantum
    monkeypatch.setattr(redo, "_qadb", lambda args: "QADB")
    monkeypatch.setattr(redo, "_load", lambda *a, **k: called.append("load"))
    monkeypatch.setattr(redo, "_values", lambda *a, **k: called.append("values"))
    monkeypatch.setattr(redo, "_judge", lambda *a, **k: called.append("judge"))
    planned.holdings += [(1, "b", 1, True, True, True)]
    arguments = [a for a in planned.arguments if a not in ("--only", "process")]  # all stages
    # The later stages run, and the exit status still reflects the process failure.
    assert redo.main(arguments) == 1
    assert called == ["load", "values", "judge"]
