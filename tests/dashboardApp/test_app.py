"""Tests of the dashboard app (`pfs.drp.qa.dashboard.app`) over a live store fed by a stand-in for qadb."""

import holoviews as hv
import pandas as pd
import panel as pn

from pfs.drp.qa.dashboard.app import Dashboard, verdictGrid
from pfs.drp.qa.dashboard.live import QadbMetricsStore
from pfs.drp.qa.dashboard.serving import taskThresholds

# As the app's main does.
hv.extension("bokeh")


def _image(visit, obsType, decidedAt):
    return {
        "visit": visit,
        "arm": "b",
        "spectrograph": 1,
        "obsType": obsType,
        "seqName": obsType,
        "illumination": "full",
        "traceOnly": False,
        **dict.fromkeys(
            ("medFwhm", "medDxCenter", "dxCenterRms", "pctFlagged", "linesPerFiber", "nLines"), float("nan")
        ),
        "qaStatus": "PASS",
        "qaReason": "",
        "calexpRun": "",
        "decidedAt": decidedAt,
    }


def _liveStore(loads):
    """Return a store reading ``loads`` (lists of images), each image's latest load as qadb holds it."""

    def read(since):
        latest = {(row["visit"], row["arm"]): row for load in loads for row in load}
        iq = pd.DataFrame([row for row in latest.values() if since is None or row["decidedAt"] >= since])
        visits = sorted(set(iq["visit"])) if len(iq) else []
        return {
            "iqQaMetrics": iq,
            "dmLineResidualSummary": pd.DataFrame(columns=["visit", "arm", "spectrograph", "decidedAt"]),
            "dmLineResidualMetrics": pd.DataFrame(columns=["visit", "arm", "spectrograph", "decidedAt"]),
            "visitTimes": pd.DataFrame(
                {
                    "pfs_visit_id": visits,
                    "time_exp_start": [
                        pd.Timestamp("2026-11-01 19:00") + pd.Timedelta(minutes=v % 100) for v in visits
                    ],
                    "iic_sequence_id": pd.array([pd.NA] * len(visits), dtype="Int64"),
                }
            ),
        }

    return QadbMetricsStore(read, taskThresholds()[0], {}, minInterval=0)


def testRunTabFollowsLiveVisits():
    """The Run tab's visit range and exposure types take in visits loaded after it was built."""
    loads = [[_image(1001, "bias", "t1")]]
    dashboard = Dashboard(_liveStore(loads))
    runTab = dashboard.runTab()
    [visitRange] = runTab.select(pn.widgets.IntRangeSlider)
    [obsType] = [w for w in runTab.select(pn.widgets.MultiChoice) if w.name == "Exposure type"]
    # One visit: the slider still needs an end past its start.
    assert visitRange.end > visitRange.start
    assert visitRange.value == (1001, 1001)

    obsType.value = []  # the user leaves out biases
    loads.append([_image(1002, "arc", "t2")])
    dashboard.refresh()
    assert visitRange.end == 1002 and visitRange.value == (1001, 1002)
    assert obsType.options == ["arc", "bias"]
    assert obsType.value == ["arc"]
    # Negative control: the type the user left out stays out.
    assert "bias" not in obsType.value


def testVerdictGridEscapesReasons():
    reasons = 'nLines<100 & "medFwhm"'
    images = pd.DataFrame({"arm": ["b"], "spectrograph": [1], "status": ["WARN"], "reasons": [reasons]})
    grid = verdictGrid(images)
    assert 'title="nLines&lt;100 &amp; &quot;medFwhm&quot;"' in grid
    # Negative control: the raw text, which would end the attribute at its first quote, is not there.
    assert reasons not in grid
