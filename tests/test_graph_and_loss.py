"""Offline verification -- NO real API calls. loss_function_changer is
checked as static text only (never invoked -- calling it would construct a
real Anthropic client and make a real request).
Run from anywhere: python3 tests/test_graph_and_loss.py
"""
import sys, os, ast

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
os.chdir(os.path.join(_REPO, "agent"))

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("--", detail) if detail and not cond else "")

# ---------------------------------------------------------------------------
# graph.py wiring (import only -- build_graph() itself needs real tools/mcp,
# out of scope for an offline test; importability + dict membership is what
# matters here)
# ---------------------------------------------------------------------------
import graph
from specialists.loss_function_changer import (
    CANDIDATES as LOSS_CANDIDATES, _build_baseline_py)
check("feature_engineer is in graph.SPECIALIST_FNS", "feature_engineer" in graph.SPECIALIST_FNS)
check("feature_engineer is in graph.SPECIALISTS", "feature_engineer" in graph.SPECIALISTS)
check("graph.SPECIALIST_FNS['feature_engineer'] is the real function",
      graph.SPECIALIST_FNS["feature_engineer"].__module__.endswith("feature_engineer"))

out = graph._extract_specialist_output({"_deterministic_edit": {"file": "data.py"}, "junk": "dropped"})
check("_extract_specialist_output keeps _deterministic_edit", "_deterministic_edit" in out)
check("_extract_specialist_output drops unlisted keys", "junk" not in out)

# ---------------------------------------------------------------------------
# loss_function_changer -- STATIC TEXT CHECKS ONLY. Never call the function.
# ---------------------------------------------------------------------------
with open("specialists/loss_function_changer.py") as f:
    src = f.read()

try:
    ast.parse(src)
    check("loss_function_changer.py parses", True)
except SyntaxError as e:
    check("loss_function_changer.py parses", False, str(e))

# Phase 0 restored the shipped POINTWISE LOG-LOSS FM, so the prompt premise
# flipped back: baseline is log-loss again, and BPR is a valid move onto it.
check("premise says the baseline is pointwise log-loss (Phase 0 restored it)",
      "pointwise log-loss" in src.lower())
check("premise no longer claims BPR is already the baseline",
      "already trained with BPR" not in src)
check("within-user invariance insight is injected",
      "WITHIN_USER_INVARIANCE" in src)

# The four checks below replace the previous prompt-guardrail assertions
# ("CRITICAL EXECUTION CONSTRAINT", "INSIDE FM.step()", the LOSS_CHOICE menu).
# Those guarded an LLM free-form path that no longer exists: per-user BPR has
# to change run_fm's loop AND the FM class body at once, which a single
# find-and-replace cannot express, so this specialist became a closed
# deterministic menu shipping whole-file baseline.py replacements. The
# guardrails are not weakened, they are unnecessary -- so assert the stronger
# property directly instead.
check("loss specialist makes no LLM call (deterministic menu)",
      "messages.create" not in src and "Anthropic(" not in src)
check("ships whole-file replacements via the _deterministic_edit fast path",
      "_deterministic_edit" in src and "baseline_official.py" in src)
check("menu is the two within-user candidates",
      [c[0] for c in LOSS_CANDIDATES] == ["loss:bpr_per_user", "loss:drop_user_linear"],
      str([c[0] for c in LOSS_CANDIDATES]))

_official = open(os.path.join(_REPO, "starter-kit", "baseline_official.py")).read()
for _lbl, _d in LOSS_CANDIDATES:
    try:
        _out = _build_baseline_py(_lbl, _official); ast.parse(_out)
        check(f"{_lbl}: builds and parses against the frozen official baseline", True)
    except Exception as _e:
        check(f"{_lbl}: builds and parses against the frozen official baseline", False, str(_e))

_bpr = _build_baseline_py("loss:bpr_per_user", _official)
check("bpr_per_user pairs WITHIN user (precomputed index, not batch-global)",
      "_build_user_pairs" in _bpr and "step_pairs" in _bpr)
check("bpr_per_user passes the train users run_fm already unpacks",
      "Xtr, ytr, utr = enc['train']" in _bpr)
_drop = _build_baseline_py("loss:drop_user_linear", _official)
check("drop_user_linear removes W[user_id] from the linear term",
      "self.W[X[:, 1:]].sum(1)" in _drop)

# ---------------------------------------------------------------------------
# supervisor.py -- static check that the stale hardcoded ban is gone
# ---------------------------------------------------------------------------
with open("specialists/supervisor.py") as f:
    sup_src = f.read()
check("stale 'loss_function_changer has already been tried' ban removed",
      "loss_function_changer has already been tried" not in sup_src)
check("feature_engineer appears in ROUTE_TO enum text",
      "feature_engineer" in sup_src and "ROUTE_TO: [" in sup_src)

print("\n" + "="*70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
