"""The frames Run31's gate could not type, on their stored numbers: what each is, and that none is judged.

``untypedFrames-run31.parquet`` and ``untypedLines-run31.parquet`` (``data/makeUntypedFixture.py``) hold a
few engineering visits of each kind: darks, a fiber-profile sequence, a dot-roach sequence and exposures
taken with the IIC's plain expose command. The tasks stored ``obsType`` ``unknown`` for all of them; the
sequence type is the opdb's.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import gate
from pfs.drp.qa.metrics.registry import CHECKS, NOT_JUDGED

DATA = Path(__file__).parent / "data"
IMAGE = ["visit", "arm", "spectrograph"]

#: The frame type of each sequence type in the fixture.
KINDS = {
    "darks": "dark",
    "fiberProfiles": "fiberProfile",
    "dotRoach": "dotRoach",
    "dotRoachInit": "dotRoach",
    "arcs": "engineering",
    "arcs_binned": "engineering",
    "flats": "engineering",
    "flats_binned": "engineering",
}


@pytest.fixture(scope="module")
def frames() -> pd.DataFrame:
    return pd.read_parquet(DATA / "untypedFrames-run31.parquet")


@pytest.fixture(scope="module")
def lines() -> pd.DataFrame:
    return pd.read_parquet(DATA / "untypedLines-run31.parquet")


def testTheFixtureHoldsEveryKind(frames, lines):
    assert set(frames["seqType"]) == set(KINDS) and set(frames["obsType"]) == {"unknown"}
    assert len(frames) == len(lines) == 133
    # The darks are stored as lit through the engineering fibers: their pfsConfig has those fibers GOOD.
    assert set(frames.loc[frames["seqType"] == "darks", "illumination"]) == {"iis"}


def testWithoutTheSequenceTypeNoneHasAFrameType(frames, lines):
    """As stored: ``obsType`` is ``unknown``, so nothing says what these frames are."""
    verdicts = gate(iq=frames.drop(columns="seqType"), lines=lines.drop(columns="seqType"))
    assert len(verdicts) == len(frames)
    assert verdicts["frameType"].isna().all()
    assert set(verdicts["qaStatus"]) == {"UNKNOWN"} and set(verdicts["qaReason"]) == {"no frame type"}


def testWithItEachIsNamedAndNoneIsJudged(frames, lines):
    verdicts = gate(iq=frames, lines=lines).merge(frames[[*IMAGE, "seqType"]], on=IMAGE)
    assert verdicts["frameType"].tolist() == verdicts["seqType"].map(KINDS).tolist()
    assert verdicts["frameType"].isin(NOT_JUDGED).all()
    assert set(verdicts["qaStatus"]) == {"UNKNOWN"} and set(verdicts["qaDecidedBy"]) == {""}
    for kind, reason in verdicts.groupby("frameType")["qaReason"]:
        assert set(reason) == {f"{kind} frames are not judged: {NOT_JUDGED[kind]}"}
    # Negative control: the reason is the type's, not that the frame has none.
    assert not verdicts["qaReason"].str.contains("no frame type").any()


def testADarkIsNotJudgedOnATraceCentreOffset(frames):
    """A dark with a 3.6 px trace-centre offset stored is a dark: the offset check does not reach it."""
    dark = frames[(frames["visit"] == 153972) & (frames["arm"] == "n") & (frames["spectrograph"] == 4)]
    assert abs(dark["medDxCenter"].iloc[0]) > CHECKS["traceCentreOffset"].fail
    assert gate(iq=dark)["qaStatus"].tolist() == ["UNKNOWN"]
    # Negative control: typed by its stored illumination as an engineering-fiber quartz, it fails that check.
    asQuartz = gate(iq=dark.assign(seqType="scienceTrace"))
    assert asQuartz["frameType"].tolist() == ["iisQuartz"]
    assert (asQuartz["qaStatus"].iloc[0], asQuartz["qaDecidedBy"].iloc[0]) == ("FAIL", "traceCentreOffset")


def testAFiberProfileSequenceIsDitheredByDesign(lines):
    """The slit steps through -0.4 to +0.4 px between a first and a last exposure at home."""
    sequence = lines[lines["seqType"] == "fiberProfiles"].sort_values("visit")
    assert sequence["visit"].nunique() == 7
    for _, camera in sequence.groupby(["arm", "spectrograph"]):
        steps = camera["value"].to_numpy() - camera["value"].iloc[0]
        np.testing.assert_allclose(steps, [0.0, -0.4, -0.2, 0.0, 0.2, 0.4, 0.0], atol=0.03)
    span = sequence.groupby(["arm", "spectrograph"])["value"].agg(lambda values: values.max() - values.min())
    # Negative control: a dot-roach sequence is not: its offsets move by a fraction of that within it.
    roach = lines[lines["seqType"] == "dotRoach"]
    spread = roach.groupby(["arm", "spectrograph"])["value"].agg(lambda values: values.max() - values.min())
    assert spread.max() < 0.2 < 0.7 < span.min()


def testNoneOfThemEntersANormalValue(frames, lines):
    """A frame type with no relative check gives the normal values nothing."""
    from pfs.drp.qa.metrics.normalValues import candidates

    made = candidates(frames, lines)
    assert made.empty
    # Negative control: typed as quartz and arcs they would make values, from frames that are neither.
    wrong = frames.assign(
        seqType=np.where(frames["seqType"].str.startswith("arcs"), "scienceArc", "scienceTrace")
    )
    assert not candidates(wrong, lines.assign(seqType="scienceTrace")).empty
