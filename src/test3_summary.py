"""Test 3, phase A summary: every number of results/test3/REPORT_A.md comes from here.

    python -m src.test3_summary --out results/test3/summary_A.json
Inputs: tc.json (first run, gen-1 only), gen3.json (event simulation, final), count3.json, gridcheck*.json,
validate.json.
"""
import argparse
import json

import numpy as np

R = "results/test3/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old, new = json.load(open(R + "tc.json")), json.load(open(R + "gen3.json"))
    cnt, val = json.load(open(R + "count3.json")), json.load(open(R + "validate.json"))
    grid = json.load(open(R + "gridcheck.json")) + json.load(open(R + "gridcheck_42_77.json"))
    n = new["n"]
    recs = new["gaps"] + new["spawned"]
    ev = new["events"]
    out = dict(n=n, n_pairs=len(recs), n_gen1_pairs=len(new["gaps"]), n_spawned=len(new["spawned"]),
               status=new["pair_status"], events_per_generation=new["events_per_generation"],
               events_total=len(ev), events_both_zeros_le_100=sum(1 for e in ev if e["zj"] <= 100),
               survivors=new["survivors"], missing_survivor_pairs=new["missing_survivor_pairs"])
    col = [r for r in new["gaps"] if r["status"] == "collided" and r["i"] + 1 < n]
    ra = np.array([r["ratio"] for r in col])
    out["gen1_in_range"] = dict(n=len(col), ratio_min=ra.min(), q10=np.quantile(ra, 0.1), median=np.median(ra),
                                q90=np.quantile(ra, 0.9), ratio_max=ra.max(), violations=new["gen1_ratio_violations"],
                                tc_min=min(r["tc"] for r in col), tc_max=max(r["tc"] for r in col),
                                max_stab_dt=max(r["stab_dt"] for r in col), all_sign_flip=all(r["sign_flip"] for r in col))
    out["censored"] = sorted([dict(pair=[r["i"] + 1, r["j"] + 1], gen=r["gen"], tc0=r["tc0"], t_start=r["t_start"])
                              for r in recs if r["status"] == "censored"], key=lambda d: d["tc0"])
    out["lost"] = [dict(pair=[r["i"] + 1, r["j"] + 1], gen=r["gen"], tc0=r["tc0"]) for r in recs if r["status"] == "lost"]
    oldg = {r["gap"]: r for r in old["gaps"]}
    oc = {g for g, r in oldg.items() if r["status"] == "collided" and not r["edge"]}
    nc = {r["gap"] for r in col}
    out["old_run"] = dict(collided=len(oc), absorbed=sum(1 for r in oldg.values() if r["status"] == "absorbed"))
    out["common_collisions_max_abs_dtc"] = max(abs(oldg[g]["tc"] - r["tc"]) for r in col for g in [r["gap"]] if g in oc)
    out["old_collided_subset_of_new"] = oc <= nc
    chg = []
    for r in new["gaps"]:
        o = oldg.get(r["gap"])
        if o and o["status"] == "absorbed" and r["status"] in ("collided", "preempted"):
            tu = o["t_absorbed_upto"]
            chg.append(dict(gap=r["gap"], edge=bool(r["edge"]), new_status=r["status"], tc=r["tc"], tc0=r["tc0"], old_t_last_good=tu,
                            old_ratio_at_loss=tu / r["tc0"], old_passed_new_tc=bool(tu < r["tc"])))
    inr = [c for c in chg if not c["edge"]]
    out["status_changes"] = dict(n_total=len(chg), n_edge=len(chg) - len(inr), n=len(inr),
                                 n_collided=sum(1 for c in inr if c["new_status"] == "collided"),
                                 n_preempted=sum(1 for c in inr if c["new_status"] == "preempted"),
                                 old_ratio_at_loss_values=sorted(set(round(c["old_ratio_at_loss"], 3) for c in chg)),
                                 n_old_passed_new_tc=sum(c["old_passed_new_tc"] for c in inr), rows=chg)
    match = []
    for r in recs:
        if r["status"] == "preempted":
            m = [e for e in ev if e["gen"] > 1 and abs(e["tc"] - r["tc"]) < 1e-6]
            match.append(dict(pair=[r["i"] + 1, r["j"] + 1], gen=r["gen"], tc=r["tc"],
                              gen2_event=[[e["zi"], e["zj"]] for e in m],
                              abs_dtc=[abs(e["tc"] - r["tc"]) for e in m]))
    out["preempted"] = dict(n=len(match), all_have_gen2_partner=all(m["gen2_event"] for m in match),
                            max_abs_dtc=max((max(m["abs_dtc"]) for m in match if m["abs_dtc"]), default=None), rows=match)
    ch = [r for r in cnt["rows"] if "diff" in r]
    out["count_dividers"] = dict(n_points=cnt["n_points"], all_match=cnt["all_match"], t_min=cnt["t_min_checked"],
                                 p_used=sorted(set(r["p"] for r in ch)), max_unverified=max(r["unverified_events"] for r in ch),
                                 points_below_m76_6=cnt["points_below_m76_6_p_ge_57"], match_below_m76_6=cnt.get("match_below_76_6"))
    out["gridcheck"] = dict(pairs=[g["pair"] for g in grid], fallback_used=[g["pair"] for g in grid if g["tracked"]],
                            max_dx0=max(x["dx0"] for g in grid for x in g["runs"]),
                            max_dout=max(x["dout"] for g in grid for x in g["runs"]))
    out["H_t0_positive"] = old["sym_gap_H_t0"]
    out["validate"] = {k: val[k] for k in ("ok", "max_rel_err_H0", "max_zero_dev", "n_sign_changes")}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    s = out
    print(f"pairs {s['n_pairs']} (gen1 {s['n_gen1_pairs']}, spawned {s['n_spawned']}); status {s['status']}")
    print(f"events {s['events_total']} {s['events_per_generation']}, both zeros <=100: {s['events_both_zeros_le_100']}")
    g = s["gen1_in_range"]
    print(f"gen1 in range {g['n']}: ratio min {g['ratio_min']:.3f} q10 {g['q10']:.3f} med {g['median']:.3f} q90 {g['q90']:.3f} max {g['ratio_max']:.3f}; viol {g['violations']}")
    print("censored:", [(c['pair'], c['gen'], round(c['tc0'], 1)) for c in s["censored"]], "lost:", s["lost"])
    print(f"old run collided {s['old_run']['collided']}; common max |dtc| {s['common_collisions_max_abs_dtc']:.2e}; old subset {s['old_collided_subset_of_new']}")
    c = s["status_changes"]
    print(f"status changes in range {c['n']} (+{c['n_edge']} edge; collided {c['n_collided']}, preempted {c['n_preempted']}); old loss ratios {c['old_ratio_at_loss_values']}; old passed new tc: {c['n_old_passed_new_tc']}")
    p = s["preempted"]
    print(f"preempted {p['n']}: all have gen>=2 partner {p['all_have_gen2_partner']}, max |dtc| {p['max_abs_dtc']}")
    print("count:", s["count_dividers"])
    print("gridcheck:", s["gridcheck"])


if __name__ == "__main__":
    main()
