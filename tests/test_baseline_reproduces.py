"""Task Requirement 1 ("reproduce the official baseline"), turned into a test.

SLOW: trains the FM for real (~60s per run, ~2min total). Zero API cost --
local CPU only, no Anthropic/Tavily involvement whatsoever.

Runs starter-kit/baseline_official.py, the FROZEN copy, not baseline.py --
baseline.py is what the agent edits, so asserting against it would turn every
successful experiment into a test failure. baseline_official.py is in
agent.py's PROTECTED_FILES precisely so this reference stays trustworthy.

WHY THIS TEST EXISTS
--------------------
starter-kit/baseline.py stopped holding the official baseline at commit
36443bc ("restore clean baseline"): an agent iteration's BPR rewrite was
committed over it, and that rewrite's own run_log entry scored primary=None,
i.e. it was never validated. Every later "improvement" was then measured
against a contaminated 0.6039 starting point that was really iteration 0.
This test pins the real reference so that cannot recur silently.

Works two ways:
    python3 tests/test_baseline_reproduces.py
    pytest tests/test_baseline_reproduces.py -m slow
"""
import json
import os
import re
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STARTER_KIT = os.path.join(_REPO, "starter-kit")

# Published reference: starter-kit/baseline_scores.json -> scores.fm_official.valid
_PUBLISHED = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
_PRIMARY_RANGE = (0.6013, 0.6019)   # seed 0 is deterministic; measured 0.6015
_METRIC_TOL = 0.001                 # GAUC/nDCG tolerance around the published value

_PAT = re.compile(
    r"^\s+(valid|test)\s+GAUC ([\d.]+) \| nDCG@5 ([\d.]+) \| primary ([\d.]+)", re.M)


def _run_official(seed=0):
    """Train the frozen official baseline and parse its scores."""
    proc = subprocess.run(
        [sys.executable, "baseline_official.py", "--model", "fm", "--seed", str(seed)],
        cwd=STARTER_KIT, capture_output=True, text=True, timeout=3600)
    out = proc.stdout + proc.stderr
    scores = {m.group(1): {"GAUC": float(m.group(2)), "nDCG@5": float(m.group(3)),
                           "primary": float(m.group(4))} for m in _PAT.finditer(out)}
    assert scores, f"baseline_official.py produced no parseable scores:\n{out[-2000:]}"
    return scores


def test_official_baseline_reproduces():
    """The official baseline reproduces the published valid scores."""
    valid = _run_official(seed=0)["valid"]
    lo, hi = _PRIMARY_RANGE
    assert lo <= valid["primary"] <= hi, (
        f"valid primary {valid['primary']} outside {_PRIMARY_RANGE}. "
        f"If this reads ~0.6039, baseline_official.py is NOT the official file -- "
        f"it is the contaminated BPR again.")
    for metric in ("GAUC", "nDCG@5"):
        assert abs(valid[metric] - _PUBLISHED[metric]) <= _METRIC_TOL, (
            f"valid {metric} {valid[metric]} differs from published "
            f"{_PUBLISHED[metric]} by more than {_METRIC_TOL}")


def test_baseline_is_deterministic():
    """T9: same seed twice => identical score.

    This is what makes the NO_OP detector sound: score_analyst flags an edit as
    NO_OP when its primary equals best to within 1e-9, which is only meaningful
    if training is bit-reproducible for a fixed seed.
    """
    a = _run_official(seed=0)["valid"]["primary"]
    b = _run_official(seed=0)["valid"]["primary"]
    assert a == b, f"training is not seed-deterministic: {a} != {b}"


def test_seed_variance_file_is_consistent():
    """Phase 0.4's measured variance exists and agrees with the published mean.

    The accept gate in agent.py is calibrated against this file, so a stale or
    missing one would silently mis-calibrate it.
    """
    path = os.path.join(STARTER_KIT, "seed_variance.json")
    assert os.path.exists(path), "starter-kit/seed_variance.json missing (Phase 0.4)"
    data = json.load(open(path))
    vp = data["summary"]["valid"]["primary"]
    assert abs(vp["mean"] - _PUBLISHED["primary"]) <= 0.001, (
        f"measured valid primary mean {vp['mean']} disagrees with published "
        f"{_PUBLISHED['primary']}")
    assert vp["std"] < 0.003, f"seed std {vp['std']} implausibly large"
    # the organizers' own file must not have been clobbered by Phase 0.4
    official = json.load(open(os.path.join(STARTER_KIT, "baseline_scores.json")))
    assert "std_over_5_seeds" in official["scores"]["fm_official"], (
        "starter-kit/baseline_scores.json looks overwritten -- it is the "
        "organizers' reference file and must stay untouched")


if __name__ == "__main__":
    failures = []
    for fn in (test_official_baseline_reproduces, test_baseline_is_deterministic,
               test_seed_variance_file_is_consistent):
        try:
            fn()
            print("PASS -", fn.__name__)
        except AssertionError as e:
            failures.append(fn.__name__)
            print("FAIL -", fn.__name__, "--", e)
    print("\n" + "=" * 70)
    print(f"{3 - len(failures)}/3 PASSED" + (f", {len(failures)} FAILED" if failures else ""))
    sys.exit(1 if failures else 0)
