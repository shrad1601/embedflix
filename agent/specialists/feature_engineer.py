"""Item-side feature specialist.

THE INSIGHT THIS SPECIALIST IS BUILT ON
---------------------------------------
starter-kit/evaluate.py computes BOTH scored metrics strictly WITHIN each
user's own impression list: it groups rows by user_id, sorts each user's list
by predicted score, then takes a per-user AUC (GAUC) and a per-user nDCG@5.

It follows that **any score component that is constant within a user is
mathematically invisible to both metrics** -- it shifts every row in that
user's list by the same amount and so cannot reorder any of them.

Consequences, all checkable against evaluate.py:
  * the W[user_id] linear term contributes EXACTLY ZERO to GAUC and nDCG@5
  * all 30 columns of user_features_pure.csv, used as linear features,
    contribute EXACTLY ZERO in isolation
  * user features only matter through the FM INTERACTION term V_user . V_item,
    which does vary within the list
  * ITEM-side features are the only direct lever

This explains the organizers' own ablation_features.py result that adding
user-side categoricals makes things WORSE on the real held-out test split
(0.5953 -> 0.5936 / 0.5933): they add parameters and noise while contributing
nothing to within-user ordering. It is also why this specialist's previous
menu (raw play_cnt / like_cnt / show_cnt) was aimed at the wrong quantity --
raw counts encode POPULARITY, which is largely constant within a user's list
and is not the same thing as long-view propensity.

Every candidate below is item-side and varies within a single user's list.

MECHANISM (unchanged, it was already sound)
-------------------------------------------
Each candidate is a FULL replacement of starter-kit/data.py applied through
code_writer's _deterministic_edit fast path -- zero LLM calls, zero
translation risk. old_code is read fresh from disk each call, so candidates
are alternative full-file states rather than stacked patches: every one is
built from the same 5-field baseline and is therefore directly comparable to
it, never to whichever candidate ran before.

FM here is purely categorical-embedding based (every FIELDS entry is a vocab
lookup; there is no continuous-input path), so continuous signals are
quantile-bucketed the way dur_bucket already is. No model surgery.

LEAKAGE DISCIPLINE
------------------
Every bucket edge and every target-encoding statistic is fit on the TRAIN
split only, and the identical statistic is applied to train, valid and test so
there is no train/serve skew. (Leave-one-out on train rows was tried and is
WORSE -- 0.5846 vs 0.6021 valid -- because LOO and full statistics bucket the
same row differently, so the model is served a field it never trained on.)
The log file's per-interaction
columns (is_click, is_like, is_follow, is_comment, is_forward, play_time_ms)
are never candidates: LABEL ('long_view') is derived from the same row's
play_time_ms/duration_ms and the is_* columns are simultaneous outcomes of
the very impression being ranked.
"""

N_BUCKETS = 24
TE_ALPHA = 20.0

# (label, config, description). Config keys feed _build_data_py directly.
#   stat_feats: (field_name, numerator_col, denominator_col|None) from
#               video_features_statistic_pure.csv
#   cat_cols:   item-side categoricals from video_features_basic_pure.csv
#   target_enc: adds video_te + author_te (train-only, leave-one-out)
CANDIDATES = [
    ("features:target_enc",
     dict(stat_feats=[], cat_cols=[], target_enc=True),
     "per-video and per-author historical long-view rate from the train split "
     "only, Bayesian-smoothed (k + a*p_bar)/(n + a) with a=20 -- a direct prior "
     "on the label that varies within every user's list and uses no "
     "forward-looking data. MEASURED NEUTRAL (valid 0.6021 vs 0.6015 baseline): "
     "100% of valid/test rows carry a video_id and author_id already seen in "
     "train (median 44 train impressions each), so the FM already holds a free "
     "embedding per video and per author that can learn this propensity "
     "directly -- a target encoding of the same entity is redundant with it. "
     "Kept as a documented negative result"),

    ("features:longview_prior",
     dict(stat_feats=[("longplay_rate", "long_time_play_cnt", "show_cnt"),
                      ("complete_rate", "complete_play_cnt", "play_cnt"),
                      ("valid_rate", "valid_play_cnt", "show_cnt"),
                      ("play_progress", "play_progress", None)],
          cat_cols=[], target_enc=False),
     "near-direct priors on long_view from video_features_statistic_pure.csv, "
     "as RATIOS not raw counts (raw counts encode popularity, the wrong "
     "quantity): long_time_play_cnt/show_cnt, complete_play_cnt/play_cnt, "
     "valid_play_cnt/show_cnt, play_progress"),

    ("features:item_cat",
     dict(stat_feats=[],
          cat_cols=["music_id", "tag", "video_type", "upload_type"],
          target_enc=False),
     "the item-side categoricals sitting unused in video_features_basic_pure.csv "
     "-- music_id and tag are not in FIELDS at all despite being free item-side "
     "signal (tag is multi-valued, first tag only)"),

    ("features:play_quality",
     dict(stat_feats=[("avg_play_dur", "play_duration", "play_cnt"),
                      ("like_rate", "like_cnt", "show_cnt"),
                      ("follow_rate", "follow_cnt", "show_cnt"),
                      ("share_rate", "share_cnt", "show_cnt")],
          cat_cols=[], target_enc=False),
     "per-impression engagement quality ratios: play_duration/play_cnt, "
     "like_cnt/show_cnt, follow_cnt/show_cnt, share_cnt/show_cnt"),

    ("features:combined",
     None,   # resolved at call time from which earlier candidates improved
     "union of whichever of the preceding candidates actually improved the "
     "score, so the phase ends on the best combination rather than the best "
     "single family"),
]

_COMBINABLE = CANDIDATES[:-1]


def _resolve_combined(state):
    """Union the configs of the earlier candidates that actually improved.

    Routing to this specialist is forced and sequential (see supervisor.py),
    so the i-th feature_engineer entry in experiment_history corresponds to
    _COMBINABLE[i]. score_analyst already computed `improved` for each, against
    a noise-aware threshold, so that flag is the honest signal here -- no
    re-deriving deltas from raw scores.
    """
    hist = [h for h in state.get("experiment_history", [])
            if h.get("specialist") == "feature_engineer"]
    winners = [_COMBINABLE[i] for i, h in enumerate(hist)
               if i < len(_COMBINABLE) and h.get("improved")]
    cfg = dict(stat_feats=[], cat_cols=[], target_enc=False)
    for _lbl, c, _d in winners:
        cfg["stat_feats"] = cfg["stat_feats"] + c["stat_feats"]
        cfg["cat_cols"] = cfg["cat_cols"] + [x for x in c["cat_cols"]
                                             if x not in cfg["cat_cols"]]
        cfg["target_enc"] = cfg["target_enc"] or c["target_enc"]
    return cfg, [w[0] for w in winners]


def feature_engineer(state: dict, tools: dict) -> dict:
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

    untried = [c for c in CANDIDATES if c[0] not in tried]
    label, cfg, desc = (untried or CANDIDATES)[0]

    note = ""
    if label == "features:combined":
        cfg, winners = _resolve_combined(state)
        if winners:
            note = f" Combining the candidates that improved: {', '.join(winners)}."
        else:
            note = (" No single candidate improved the score, so this combination "
                    "is empty and reproduces the baseline exactly -- expect it to "
                    "be flagged NO_OP, after which the feature-phase gate stops "
                    "the run.")

    current_data_py = tools["read_file"]({"file_path": "data.py"})
    new_data_py = _build_data_py(**cfg)

    hypothesis = (
        f"Feature engineer ({label}): {desc}. Current primary {primary:.4f}.{note} "
        "Rationale: evaluate.py ranks strictly WITHIN each user's impression "
        "list, so any component constant within a user cannot reorder that list "
        "and is invisible to GAUC and nDCG@5 -- which is why user-side features "
        "already measured worse (0.5953 -> 0.5936/0.5933) and why every field "
        "added here is item-side. All bucket edges and target-encoding "
        "statistics are fit on the train split only."
    )

    return {
        **state,
        "hypothesis": hypothesis,
        "code_change_instruction": (
            "(deterministic) replace the full contents of data.py with an "
            f"item-side feature variant: stat_feats={cfg['stat_feats']}, "
            f"cat_cols={cfg['cat_cols']}, target_enc={cfg['target_enc']}."
        ),
        "reasoning": hypothesis,
        "tried_approaches": tried + [label],
        "_deterministic_edit": {
            "file": "data.py",
            "old_code": current_data_py,
            "new_code": new_data_py,
        },
    }


# --------------------------------------------------------------------------
# data.py generator. A small interpolated CONFIG header + a large static
# BODY constant -- the body needs no f-string brace-escaping, which is
# where a template this size otherwise goes wrong.
# --------------------------------------------------------------------------
_BODY = '''

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _div(a, b):
    a = _f(a); b = _f(b)
    return a / b if b > 0 else 0.0


def get_feature_info(user_feature_file=USER_FEATURE_FILE, video_feature_file=VIDEO_FEATURE_FILE):
    """Real column names in the untapped user/video feature files. Does NOT
    include log-file columns -- see the leakage note above."""
    info = {}
    for name, path in [('user', user_feature_file), ('video', video_feature_file)]:
        with open(path) as f:
            info[f'{name}_feature_columns'] = next(csv.reader(f))
    return info


def load(data_dir):
    """Log rows + item-side features. Returns splits dict.

    Tuple layout (positions 0-6 are the ORIGINAL contract -- baseline.py,
    submit.py and ablation_features.py all index them positionally):
      0 date, 1 user_id, 2 video_id, 3 author_id, 4 tab, 5 duration_ms,
      6 label, 7 [numeric video-stat features], 8 [item categoricals]
    """
    vid2author = {}
    vid2cat = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
            if ITEM_CAT_COLS:
                vals = []
                for c in ITEM_CAT_COLS:
                    v = (r.get(c) or '').strip()
                    if c == 'tag':
                        # tag is MULTI-VALUED (quoted comma-separated, e.g. "20,43";
                        # 1791 rows). Use the first tag only -- generalizes better
                        # than treating each combination as its own vocab entry.
                        v = v.split(',')[0].strip()
                    vals.append(v if v else 'UNK')
                vid2cat[r['video_id']] = vals
    cat_unk = ['UNK'] * len(ITEM_CAT_COLS)

    vid2stats = {}
    if VIDEO_STAT_FEATS:
        with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as fh:
            for r in csv.DictReader(fh):
                vid2stats[r['video_id']] = [
                    _f(r.get(num)) if den is None else _div(r.get(num), r.get(den))
                    for (_nm, num, den) in VIDEO_STAT_FEATS
                ]
    stat_zeros = [0.0] * len(VIDEO_STAT_FEATS)

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid = r['video_id']
                rows.append((int(r['date']), r['user_id'], vid,
                             vid2author.get(vid, 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             vid2stats.get(vid, stat_zeros),
                             vid2cat.get(vid, cat_unk)))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def _bucket_edges(values, n):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Categorical ids. Unseen values fall into each field's UNK slot.
    Returns (X, y, users) per split and the total field dim.

    LEAKAGE DISCIPLINE: every bucket edge and every target-encoding statistic
    below is fit on the TRAIN split only. Target encodings use leave-one-out
    on train rows so a row can never encode its own label.
    """
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr], 10)
    stat_edges = [_bucket_edges([x[7][i] for x in tr], N_BUCKETS)
                  for i in range(len(VIDEO_STAT_FEATS))]

    if USE_TARGET_ENC:
        vpos = collections.Counter(); vcnt = collections.Counter()
        apos = collections.Counter(); acnt = collections.Counter()
        tot = 0
        for x in tr:
            vpos[x[2]] += x[6]; vcnt[x[2]] += 1
            apos[x[3]] += x[6]; acnt[x[3]] += 1
            tot += x[6]
        pbar = (tot / len(tr)) if tr else 0.0

        def te_full(key, pos, cnt):
            n = cnt[key]
            return (pos[key] + TE_ALPHA * pbar) / (n + TE_ALPHA) if n else pbar

        te_v_edges = _bucket_edges([te_full(x[2], vpos, vcnt) for x in tr], N_BUCKETS)
        te_a_edges = _bucket_edges([te_full(x[3], apos, acnt) for x in tr], N_BUCKETS)

    def raw(x):
        vals = [x[1], x[2], x[3], x[4], str(int(np.searchsorted(dur_edges, x[5])))]
        for i in range(len(VIDEO_STAT_FEATS)):
            vals.append(str(int(np.searchsorted(stat_edges[i], x[7][i]))))
        vals.extend(x[8])
        if USE_TARGET_ENC:
            # The SAME train-fitted statistic is used for train, valid and test.
            # Using leave-one-out on train rows instead was measured at
            # valid 0.5846 vs 0.6021 here -- LOO and full statistics land a row
            # in different quantile buckets, so the field the model trains on is
            # not the field it is served, and that train/serve skew cost 0.0175.
            vv = te_full(x[2], vpos, vcnt)
            av = te_full(x[3], apos, acnt)
            vals.append(str(int(np.searchsorted(te_v_edges, vv))))
            vals.append(str(int(np.searchsorted(te_a_edges, av))))
        return vals

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
'''

def _build_data_py(stat_feats, cat_cols, target_enc, n_buckets=N_BUCKETS, te_alpha=TE_ALPHA):
    """stat_feats: list of (field_name, numerator_col, denominator_col_or_None)
    cat_cols:    list of column names from video_features_basic_pure.csv
    target_enc: adds video_te + author_te (train-only, leave-one-out)"""
    names = [nm for (nm, _n, _d) in stat_feats]
    fields = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
    fields += [n + '_bucket' for n in names]
    fields += list(cat_cols)
    if target_enc:
        fields += ['video_te', 'author_te']

    header = '"""KuaiRand-Pure data loading + official splits + feature encoding.\n\n' \
        'Generated by feature_engineer. Item-side features only.\n\n' \
        'WHY ITEM-SIDE: evaluate.py computes GAUC and nDCG@5 strictly WITHIN each\n' \
        "user's own impression list. Any score component that is constant within a\n" \
        'user therefore cannot reorder that list and is mathematically invisible to\n' \
        'both metrics -- which is why the organizers\' own ablation_features.py found\n' \
        'user-side categorical features make things WORSE (0.5953 -> 0.5936/0.5933).\n' \
        'Every field added here varies within a single user\'s impression list.\n' \
        '"""\n' \
        'import csv, os, collections\n' \
        'import numpy as np\n\n' \
        "LABEL = 'long_view'\n" \
        "SPLITS = {'train': (20220408, 20220421),\n" \
        "          'valid': (20220422, 20220428),\n" \
        "          'test':  (20220429, 20220508)}\n" \
        'FIELDS = ' + repr(fields) + '\n\n' \
        'N_BUCKETS = ' + repr(n_buckets) + '\n' \
        'TE_ALPHA = ' + repr(te_alpha) + '\n' \
        '# (field_name, numerator_col, denominator_col or None) from video_features_statistic_pure.csv\n' \
        'VIDEO_STAT_FEATS = ' + repr(stat_feats) + '\n' \
        '# item-side categoricals from video_features_basic_pure.csv\n' \
        'ITEM_CAT_COLS = ' + repr(list(cat_cols)) + '\n' \
        'USE_TARGET_ENC = ' + repr(bool(target_enc)) + '\n\n' \
        "USER_FEATURE_FILE = os.path.join(os.path.dirname(__file__), 'KuaiRand-Pure/data/user_features_pure.csv')\n" \
        "VIDEO_FEATURE_FILE = os.path.join(os.path.dirname(__file__), 'KuaiRand-Pure/data/video_features_statistic_pure.csv')\n\n" \
        '# NOTE: the log file\'s per-interaction columns (is_click, is_like, is_follow,\n' \
        '# is_comment, is_forward, play_time_ms, ...) are deliberately NOT exposed as\n' \
        "# candidate input features. LABEL ('long_view') is derived from the same row's\n" \
        '# play_time_ms/duration_ms, and the other is_* columns are simultaneous\n' \
        '# outcomes of the impression being predicted -- using any of them as an FM\n' \
        '# input field would leak the label.\n'

    return header + _BODY
