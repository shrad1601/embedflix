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

# Phase 2 rewrote this specialist from an LLM free-form proposer into a closed
# deterministic menu, so it is now safe to IMPORT and CALL offline: it makes no
# Anthropic client and no network request at all.
check("no Anthropic client is constructed at import time",
      "from anthropic import Anthropic" not in src and "Anthropic(" not in src)
check("makes zero LLM calls", "messages.create" not in src)

import ast as _ast
from specialists.loss_function_changer import (
    CANDIDATES as LOSS_CANDIDATES, loss_function_changer, _build_baseline_py)

check("loss menu is the two within-user candidates",
      [c[0] for c in LOSS_CANDIDATES] == ["loss:bpr_per_user", "loss:drop_user_linear"],
      str([c[0] for c in LOSS_CANDIDATES]))

with open("../starter-kit/baseline_official.py") as f:
    official = f.read()
check("frozen official baseline is the pointwise log-loss version",
      "g = ((sigmoid(z) - y) / B)" in official and "BPR" not in official)

for label, _desc in LOSS_CANDIDATES:
    out = _build_baseline_py(label, official)
    try:
        _ast.parse(out); ok = True
    except SyntaxError as e:
        ok = False; print("  syntax error:", e)
    check(f"{label}: generated baseline.py parses", ok)
    check(f"{label}: is a real change, not a no-op", out != official)

bpr = _build_baseline_py("loss:bpr_per_user", official)
check("bpr_per_user pairs WITHIN user (not globally across the batch)",
      "_build_user_pairs" in bpr and "step_pairs" in bpr)
check("bpr_per_user actually binds the train users run_fm unpacked",
      "Xtr, ytr, utr = enc['train']" in bpr and "_build_user_pairs(utr, ytr)" in bpr)
check("bpr_per_user's training loop uses pair batches, not row batches",
      "m.step_pairs(" in bpr and "m.step(Xtr[idx[" not in bpr)

drop = _build_baseline_py("loss:drop_user_linear", official)
check("drop_user_linear excludes field 0 (user_id) from the linear term",
      "self.W[X[:, 1:]].sum(1)" in drop)
check("drop_user_linear leaves the interaction term untouched",
      "inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))" in drop)

# exercise the real function with mocked tools -- no client exists to mock
mock = {"read_file": lambda kw: official}
o = loss_function_changer({"tried_approaches": [], "current_scores": {"primary": 0.6015}}, mock)
check("sets _deterministic_edit targeting baseline.py",
      o["_deterministic_edit"]["file"] == "baseline.py")
check("picks the first untried candidate", o["tried_approaches"] == ["loss:bpr_per_user"])
check("whole-file replacement: old_code is the file as read",
      o["_deterministic_edit"]["old_code"] == official)
o2 = loss_function_changer({"tried_approaches": ["loss:bpr_per_user"],
                            "current_scores": {"primary": 0.6031}}, mock)
check("advances to loss:drop_user_linear", o2["tried_approaches"][-1] == "loss:drop_user_linear")
check("falls back safely when baseline_official.py is unreadable",
      loss_function_changer({"tried_approaches": []},
          {"read_file": lambda kw: "ERROR: nope" if "official" in kw["file_path"] else official}
      )["_deterministic_edit"]["new_code"] != "")

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
