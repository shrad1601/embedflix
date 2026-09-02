"""Loss-alignment specialist.

WHY THIS SPECIALIST EXISTS IN THIS FORM
---------------------------------------
The scored metric is WITHIN-USER ranking: evaluate.py groups rows by user_id,
sorts each user's own impression list by predicted score, and takes a per-user
AUC (GAUC) and a per-user nDCG@5. The official baseline trains with POINTWISE
log-loss, which optimizes calibrated probability, not ordering. Aligning the
training objective with the metric is therefore a first-class lever.

This file used to be an LLM free-form proposal node. It is now a closed
deterministic menu, for two reasons:

  1. Dead-code risk. code_writer's LLM path applies exactly ONE find-and-replace
     at ONE location. Per-user BPR has to change TWO sites at once (run_fm's
     training loop and the FM class body). A single-hunk edit physically cannot
     express that, so the LLM path would land unreachable code while the real
     loss kept running unchanged and silently -- producing a normal-looking
     score that reflected nothing about the proposed loss. Whole-file
     replacement via _deterministic_edit sidesteps the limitation entirely.
  2. Zero LLM calls, zero translation risk, and no import-time Anthropic client
     (so this module is importable in offline tests).

A NOTE ON THE PREVIOUS "BPR" IN THIS REPO
-----------------------------------------
A prior agent iteration wrote a BPR into baseline.py and it was committed as
the baseline. It was broken on its own terms: it formed pairs GLOBALLY across
the batch rather than within-user, paired positionally (pos_idxs[:n] against
neg_idxs[:n]), gave zero gradient to every row past n_pairs, accepted a `users`
argument it never used, and updated the bias with a sum that is identically
zero. Phase 0 restored the official log-loss baseline; candidate 1 below is
the correct within-user formulation.

WHY PAIRS CANNOT BE FORMED INSIDE A BATCH
-----------------------------------------
KuaiRand-Pure train holds ~1.4M rows over ~27k users (~50 rows/user), so a
random 8192-row batch contains almost no user holding BOTH a positive and a
negative. Pairing within a batch yields almost no usable pairs -- which is
exactly why the earlier attempt paired across users and so never optimized the
ordering the metric scores. The fix is to precompute a within-user
(positive -> candidate negatives) index ONCE over the training rows and then
batch over PAIRS. See _build_user_pairs in the generated file.
"""

import os
import re

# (label, description). Each is a whole-file replacement of baseline.py built
# from the frozen official copy, so candidates are independent alternatives
# rather than patches stacked on each other.
CANDIDATES = [
    ("loss:bpr_per_user",
     "sample each positive against a negative FROM THE SAME USER and train "
     "pairwise BPR on those pairs, so every gradient step pushes on an "
     "ordering GAUC/nDCG@5 can actually see; the pair index is precomputed "
     "once over the train rows and sampled vectorized per epoch"),

    ("loss:drop_user_linear",
     "drop W[user_id] from the FM linear term. It is a per-user CONSTANT "
     "offset, so it cannot reorder that user's list and its contribution to "
     "both metrics is exactly zero -- removing it frees capacity and deletes "
     "a parameter that can only overfit"),
]


def loss_function_changer(state: dict, tools: dict) -> dict:
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

    untried = [c for c in CANDIDATES if c[0] not in tried]
    label, desc = (untried or CANDIDATES)[0]

    current = tools["read_file"]({"file_path": "baseline.py"})
    official = tools["read_file"]({"file_path": "baseline_official.py"})
    if not official or official.startswith("ERROR"):
        # baseline_official.py is the frozen reference written in Phase 0. If it
        # is somehow absent, fall back to what is on disk rather than failing.
        official = current

    hypothesis = (
        f"Loss alignment ({label}): {desc}. Current primary {primary:.4f}. "
        "The metric ranks strictly WITHIN each user's impression list, so the "
        "training objective should be a within-user ordering objective; the "
        "official baseline's pointwise log-loss optimizes calibration instead."
    )

    return {
        **state,
        "hypothesis": hypothesis,
        "code_change_instruction": (
            f"(deterministic) replace the full contents of baseline.py with the "
            f"{label} variant, built from the frozen official baseline."
        ),
        "reasoning": hypothesis,
        "tried_approaches": tried + [label],
        "_deterministic_edit": {
            "file": "baseline.py",
            "old_code": current,
            "new_code": _build_baseline_py(label, official),
        },
    }


# --------------------------------------------------------------------------
# baseline.py variant templates. Each anchor below is asserted to match
# EXACTLY ONCE before substitution, so a drifted baseline fails loudly
# here instead of silently producing a half-applied file.
# --------------------------------------------------------------------------
_PAIR_HELPERS = '''

def _build_user_pairs(users, y):
    """Precompute, ONCE, a within-user (positive -> candidate negatives) index
    over the training rows.

    Why this cannot be done inside a batch: KuaiRand-Pure train has ~1.4M rows
    over ~27k users (~50 rows/user), so a random 8192-row batch contains almost
    no user holding BOTH a positive and a negative. Pairing inside the batch
    therefore yields almost no usable pairs -- which is exactly why the previous
    BPR attempt paired GLOBALLY across the batch (mixing users) and so never
    optimized the within-user ordering the metric actually scores.

    Returns per-POSITIVE arrays, so sampling later is fully vectorized:
      pos_rows  (P,)  train row index of each positive
      neg_flat  (M,)  concatenated negative row indices, grouped by user
      neg_start (P,)  where this positive's user's negatives begin in neg_flat
      neg_len   (P,)  how many negatives that user has
    """
    byu = {}
    for i, u in enumerate(users):
        s = byu.get(u)
        if s is None:
            s = byu[u] = ([], [])
        s[int(y[i])].append(i)

    pos_rows, neg_flat, starts, lens = [], [], [], []
    for _u, (neg, pos) in byu.items():
        if not neg or not pos:
            continue                      # user gives no within-user signal
        st = len(neg_flat)
        neg_flat.extend(neg)
        n = len(neg)
        for p in pos:
            pos_rows.append(p); starts.append(st); lens.append(n)
    return (np.asarray(pos_rows, dtype=np.int64), np.asarray(neg_flat, dtype=np.int64),
            np.asarray(starts, dtype=np.int64), np.asarray(lens, dtype=np.int64))
'''

_STEP_PAIRS = '''
    def step_pairs(self, Xp, Xn):
        """BPR on aligned (positive, negative) rows drawn from the SAME user.
        Xp[i] and Xn[i] belong to one user, so the gradient only ever pushes
        on orderings the metric can actually see."""
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        sg = sigmoid(zp - zn)
        loss = float(-np.mean(np.log(sg + 1e-9)))
        g = ((sg - 1.0) / B).astype(np.float32)      # dL/dzp ; dL/dzn = -g
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, g[:, None])
        np.add.at(gW, Xn, (-g)[:, None])
        np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, (-g)[:, None, None] * (Sn[:, None, :] - En))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        # self.b is deliberately NOT updated: it cancels exactly in (zp - zn).
        return loss
'''

_PAIR_LOOP = '''        idx = rng.permutation(len(pos_rows)); t0 = time.time()
        r = rng.random(len(pos_rows))
        neg_rows = neg_flat[neg_start + (r * neg_len).astype(np.int64)]
        losses = [m.step_pairs(Xtr[pos_rows[idx[i:i + bs]]], Xtr[neg_rows[idx[i:i + bs]]])
                  for i in range(0, len(idx), bs)]'''

_ORIG_LOOP = '''        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]'''

def _build_baseline_py(variant, official_src):
    src = official_src

    if variant == "loss:bpr_per_user":
        # 1) add the pair-index helper just above the FM class
        marker = "# ---------------- Factorization Machine ----------------"
        assert src.count(marker) == 1
        src = src.replace(marker, _PAIR_HELPERS.strip("\n") + "\n\n" + marker, 1)
        # 2) add step_pairs to FM, right before predict()
        pm = "    def predict(self, X, bs=200_000):"
        assert src.count(pm) == 1
        src = src.replace(pm, _STEP_PAIRS.strip("\n") + "\n\n" + pm, 1)
        # 3) build the pair index in run_fm
        anchor = "    best, best_state, bad = -1, None, 0"
        assert src.count(anchor) == 1
        src = src.replace(anchor,
            "    pos_rows, neg_flat, neg_start, neg_len = _build_user_pairs(utr, ytr)\n" + anchor, 1)
        # 4) swap the training loop to pair batches
        assert src.count(_ORIG_LOOP) == 1
        src = src.replace(_ORIG_LOOP, _PAIR_LOOP, 1)
        # run_fm must now actually bind the train users it already unpacks
        src = src.replace("Xtr, ytr, _ = enc['train']", "Xtr, ytr, utr = enc['train']", 1)

    elif variant == "loss:drop_user_linear":
        # W[user_id] is a per-user CONSTANT offset. Both metrics rank strictly
        # within one user's impression list, so it cannot reorder anything --
        # its contribution to GAUC and nDCG@5 is exactly zero. Dropping it frees
        # capacity and removes a parameter that can only overfit.
        old = "        return self.b + self.W[X].sum(1) + inter, E, S"
        new = ("        # FIELDS[0] is user_id: its linear weight is constant within a user\n"
               "        # and therefore invisible to GAUC/nDCG@5. Excluded on purpose.\n"
               "        return self.b + self.W[X[:, 1:]].sum(1) + inter, E, S")
        assert src.count(old) == 1
        src = src.replace(old, new, 1)
    else:
        raise ValueError(variant)
    return src
