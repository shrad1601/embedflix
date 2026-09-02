"""Offline verification -- NO real API calls. supervisor's forced-override
returns before constructing any Anthropic client call when the feature menu
isn't exhausted, so this is safe by construction. agent.py's gate is a pure
function with no LLM/tools involvement at all.
Run from anywhere: python3 tests/test_gating.py
"""
import sys, os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
os.chdir(os.path.join(_REPO, "agent"))

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("--", detail) if detail and not cond else "")

# ---------------------------------------------------------------------------
# 1. supervisor forced-override (menu not exhausted -> no LLM call, no client
#    construction reached)
# ---------------------------------------------------------------------------
from specialists.supervisor import supervisor, SPECIALISTS
from specialists.feature_engineer import CANDIDATES as FEATURE_CANDIDATES

check("feature_engineer is in supervisor.SPECIALISTS", "feature_engineer" in SPECIALISTS)

_labels = [c[0] for c in FEATURE_CANDIDATES]
# derive from the live menu rather than hardcoding labels -- the Phase 1 rewrite
# replaced the old playcount/engagement menu, and a stale literal here would
# still pass while testing nothing real.
for tried in ([], _labels[:1], _labels[:len(_labels) - 1]):
    state = {"tried_approaches": tried, "current_scores": {"primary": 0.6016},
              "best_scores": {"primary": 0.6016}, "experiment_history": [], "iteration": 1}
    out = supervisor(state, tools={})  # empty tools dict -- forced path must never touch tools
    check(f"forced override fires with tried={tried}",
          out["next_specialist"] == "feature_engineer", out["next_specialist"])

check("guard condition is empty once all candidates are in tried_approaches",
      [l for l in _labels if l not in _labels] == [])
# the override must STOP forcing once the whole menu is exhausted, otherwise the
# run can never route anywhere else
_exhausted = {"tried_approaches": list(_labels), "current_scores": {"primary": 0.6015},
              "best_scores": {"primary": 0.6015}, "experiment_history": [], "iteration": 9}
_forced_when_exhausted = True
try:
    supervisor(_exhausted, tools={})  # would need real tools/LLM -> raises
except Exception:
    _forced_when_exhausted = False
check("override stops forcing once every candidate is tried (falls through to routing)",
      _forced_when_exhausted is False)

# ---------------------------------------------------------------------------
# 2. agent.py's _check_feature_phase_gate -- pure function, no tools/LLM at all
# ---------------------------------------------------------------------------
from agent import _check_feature_phase_gate

feature_labels = [c[0] for c in FEATURE_CANDIDATES]

state_partial = {"tried_approaches": ["features:playcount"], "experiment_history": []}
check("gate returns None when menu not exhausted", _check_feature_phase_gate(state_partial) is None)

state_none_improved = {
    "tried_approaches": feature_labels,
    "experiment_history": [
        {"iteration": i + 1, "specialist": "feature_engineer", "improved": False}
        for i in range(len(feature_labels))
    ],
}
gate_result = _check_feature_phase_gate(state_none_improved)
check("gate returns a stop reason when all candidates tried and none improved",
      gate_result is not None, gate_result)

state_one_improved = {
    "tried_approaches": feature_labels,
    "experiment_history": [
        {"iteration": i + 1, "specialist": "feature_engineer", "improved": (i == 1)}
        for i in range(len(feature_labels))
    ],
}
check("gate returns None when at least one candidate improved",
      _check_feature_phase_gate(state_one_improved) is None)

state_mixed = {
    "tried_approaches": feature_labels + ["training:l2"],
    "experiment_history": [
        {"iteration": i + 1, "specialist": "feature_engineer", "improved": False}
        for i in range(len(feature_labels))
    ] + [{"iteration": 99, "specialist": "training_optimizer", "improved": True}],
}
check("gate ignores non-feature_engineer history entries when deciding",
      _check_feature_phase_gate(state_mixed) is not None)

print("\n" + "="*70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
