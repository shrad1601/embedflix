"""Offline verification -- NO real Anthropic/Tavily calls anywhere in this file.
feature_engineer is a closed deterministic menu with no Anthropic client at
all, so this is safe by construction (not by mocking discipline).
Run from anywhere: python3 tests/test_feature_gate.py
"""
import sys, os, ast, csv, re

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
sys.path.insert(0, os.path.join(_REPO, "agent", "specialists"))
os.chdir(os.path.join(_REPO, "agent"))  # so `from specialists.X import Y` resolves

DATA_DIR = os.path.join(_REPO, "starter-kit", "KuaiRand-Pure", "data")
STARTER_KIT = os.path.join(_REPO, "starter-kit")

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("--", detail) if detail and not cond else "")

from specialists.feature_engineer import (
    CANDIDATES, feature_engineer, _build_data_py, _resolve_combined, _COMBINABLE)

# ---------------------------------------------------------------------------
# 1. Menu shape + every candidate generates valid, correct source
# ---------------------------------------------------------------------------
check("feature_engineer has 5 candidates", len(CANDIDATES) == 5, str([c[0] for c in CANDIDATES]))
check("features:combined is last (it depends on the others' results)",
      CANDIDATES[-1][0] == "features:combined")

with open(os.path.join(STARTER_KIT, "data.py")) as f:
    real_data_py = f.read()

def fields_of(src):
    return ast.literal_eval(re.search(r"^FIELDS = (\[.*?\])$", src, re.M).group(1))

BASE = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

for label, cfg, _desc in _COMBINABLE:
    src = _build_data_py(**cfg)
    try:
        ast.parse(src); parses = True
    except SyntaxError as e:
        parses = False; print("  syntax error:", e)
    check(f"{label}: generated data.py parses", parses)
    flds = fields_of(src)
    check(f"{label}: keeps the 5 baseline fields first, in order", flds[:5] == BASE, str(flds[:5]))
    expected_extra = ([n + "_bucket" for n, _, _ in cfg["stat_feats"]]
                      + list(cfg["cat_cols"])
                      + (["video_te", "author_te"] if cfg["target_enc"] else []))
    check(f"{label}: FIELDS adds exactly the configured domains",
          flds[5:] == expected_extra, f"{flds[5:]} != {expected_extra}")

# ---------------------------------------------------------------------------
# 2. Referenced columns really exist in the real CSVs (else silent KeyError)
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "video_features_statistic_pure.csv")) as f:
    stat_header = next(csv.reader(f))
with open(os.path.join(DATA_DIR, "video_features_basic_pure.csv")) as f:
    basic_header = next(csv.reader(f))

stat_cols = {c for _l, cfg, _d in _COMBINABLE for (_n, num, den) in cfg["stat_feats"]
             for c in (num, den) if c}
missing = sorted(c for c in stat_cols if c not in stat_header)
check("all stat columns exist in video_features_statistic_pure.csv", not missing, str(missing))

cat_cols = {c for _l, cfg, _d in _COMBINABLE for c in cfg["cat_cols"]}
missing_c = sorted(c for c in cat_cols if c not in basic_header)
check("all categorical columns exist in video_features_basic_pure.csv", not missing_c, str(missing_c))

# ---------------------------------------------------------------------------
# 3. T4 LEAKAGE AUDIT -- the single most important check in this file
# ---------------------------------------------------------------------------
LEAKY = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
         "play_time_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter"]
for label, cfg, _d in _COMBINABLE:
    src = _build_data_py(**cfg)
    hits = [c for c in LEAKY if re.search(r"r\[['\"]%s['\"]\]|['\"]%s['\"]\s*[,\]]" % (c, c), src)]
    check(f"{label}: never reads a per-interaction log column as a feature", not hits, str(hits))
    # long_view may appear ONLY as the LABEL constant, never as an input field
    lv = [ln.strip() for ln in src.splitlines()
          if "long_view" in ln and not ln.strip().startswith("#")]
    check(f"{label}: long_view appears only as LABEL", all("LABEL" in l for l in lv), str(lv))
    # every fit must be train-only
    check(f"{label}: bucket edges fit on the train split only",
          "_bucket_edges([x[5] for x in tr]" in src and "for x in tr]" in src)
    if cfg["target_enc"]:
        check(f"{label}: target-encoding counters accumulate over train rows only",
              re.search(r"for x in tr:\n\s+vpos\[x\[2\]\]", src) is not None)
        # LOO was measured WORSE (0.5846 vs 0.6021): LOO and full statistics
        # bucket the same row differently, so the model trains on one field and
        # is served another. One statistic must be used for all three splits.
        check(f"{label}: no leave-one-out train/serve skew", "te_loo" not in src)
        check(f"{label}: the same train-fitted statistic serves every split",
              src.count("vv = te_full(x[2], vpos, vcnt)") == 1 and "is_train" not in src)

# ---------------------------------------------------------------------------
# 4. The load() tuple contract baseline.py / submit.py / ablation depend on
# ---------------------------------------------------------------------------
for label, cfg, _d in _COMBINABLE:
    src = _build_data_py(**cfg)
    check(f"{label}: load() keeps user_id at x[1] and video_id at x[2]",
          "rows.append((int(r['date']), r['user_id'], vid," in src)
    check(f"{label}: encode() reads the label from x[6] and users from x[1]",
          "y[n] = x[6]" in src and "users.append(x[1])" in src)

# tag is multi-valued (quoted "20,43"); confirm it is split, not used whole
src_cat = _build_data_py(**dict(_COMBINABLE[2][1]))
check("item_cat: multi-valued tag is reduced to its first value",
      "v.split(',')[0].strip()" in src_cat)

# ---------------------------------------------------------------------------
# 5. feature_engineer(state, tools) -- exercise the real function, mock tools
# ---------------------------------------------------------------------------
mock_tools = {"read_file": lambda kwargs: real_data_py}
state = {"tried_approaches": [], "current_scores": {"primary": 0.6015}}
out = feature_engineer(state, mock_tools)
check("sets _deterministic_edit", out.get("_deterministic_edit") is not None)
check("picks the first untried candidate", out["tried_approaches"] == [CANDIDATES[0][0]],
      str(out["tried_approaches"]))
check("_deterministic_edit targets data.py", out["_deterministic_edit"]["file"] == "data.py")
check("old_code matches the real data.py on disk byte for byte",
      out["_deterministic_edit"]["old_code"] == real_data_py)
try:
    ast.parse(out["_deterministic_edit"]["new_code"]); check("new_code parses", True)
except SyntaxError as e:
    check("new_code parses", False, str(e))
check("hypothesis states the within-user invariance rationale",
      "WITHIN each user" in out["hypothesis"])

# advances through the menu in order
for i in range(1, len(CANDIDATES)):
    st = {"tried_approaches": [c[0] for c in CANDIDATES[:i]],
          "current_scores": {"primary": 0.6015},
          "experiment_history": [{"specialist": "feature_engineer", "improved": False}] * i}
    o = feature_engineer(st, mock_tools)
    check(f"advances to {CANDIDATES[i][0]} after {i} tried",
          o["tried_approaches"][-1] == CANDIDATES[i][0], str(o["tried_approaches"][-1]))

# exhausted menu wraps instead of crashing
st = {"tried_approaches": [c[0] for c in CANDIDATES], "current_scores": {"primary": 0.6015},
      "experiment_history": []}
check("does not crash once the menu is exhausted",
      feature_engineer(st, mock_tools)["_deterministic_edit"] is not None)

# ---------------------------------------------------------------------------
# 6. features:combined resolution
# ---------------------------------------------------------------------------
hist = [{"specialist": "feature_engineer", "improved": b} for b in (True, False, True, False)]
cfg, winners = _resolve_combined({"experiment_history": hist})
check("combined unions exactly the candidates that improved",
      winners == [_COMBINABLE[0][0], _COMBINABLE[2][0]], str(winners))
check("combined carries target_enc from the winning candidate", cfg["target_enc"] is True)
check("combined carries the winning categoricals", cfg["cat_cols"] == _COMBINABLE[2][1]["cat_cols"])
ast.parse(_build_data_py(**cfg))
check("combined generates valid source", True)

cfg0, w0 = _resolve_combined({"experiment_history": []})
check("combined with no winners is empty (reproduces baseline)",
      w0 == [] and fields_of(_build_data_py(**cfg0)) == BASE)
out_c = feature_engineer({"tried_approaches": [c[0] for c in _COMBINABLE],
                          "current_scores": {"primary": 0.6015},
                          "experiment_history": []}, mock_tools)
check("combined with no winners says so in the hypothesis",
      "NO_OP" in out_c["hypothesis"], out_c["hypothesis"][-160:])

# combined must ignore other specialists' history entries
mixed = [{"specialist": "training_optimizer", "improved": True},
         {"specialist": "feature_engineer", "improved": True},
         {"specialist": "model_swapper", "improved": True},
         {"specialist": "feature_engineer", "improved": False}]
_c, w = _resolve_combined({"experiment_history": mixed})
check("combined ignores non-feature_engineer history entries", w == [_COMBINABLE[0][0]], str(w))

print("\n" + "="*70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
