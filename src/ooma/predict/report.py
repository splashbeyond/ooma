from __future__ import annotations

from datetime import datetime, timezone

from .loop import RecursiveLoop


def digest(loop: RecursiveLoop) -> str:
    c = loop.cfg
    out = [f"# ooma prediction digest, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
           "Question each hour: will the close H hours from now be above the close now? Scored vs the base-rate baseline, ",
           "out-of-sample only. skill > 0 means better than the baseline. n_eff = n / H (overlapping labels). "
           "edge = mean return of confident calls after costs.", ""]
    for H in loop.H:
        P = loop.preds[H]
        ens, base = P[f"ensemble@{H}"], P[f"baseline@{H}"]
        alive = [p for p in P.values() if p.kind in ("stump", "logit") and p.status != "retired"]
        trusted = [p for p in alive if p.status == "trusted"]
        e = ens.stats
        edge = e.edge()
        out += [f"## {H}h horizon",
                f"- population: {len(alive)} alive, {len(trusted)} trusted, "
                f"{sum(p.status == 'retired' for p in P.values())} retired",
                f"- ensemble: n={e.n} n_eff={e.n_eff(H):.0f} skill={e.mean_d():+.5f} z={e.z(H):+.2f} "
                f"hit={(e.hit_rate() or 0):.1%} edge={'n/a' if edge is None else f'{edge:+.3%}'} (n_edge={e.n_edge})",
                f"- baseline up-rate: {base.predict({}):.3f}"]
        top = sorted((p for p in alive if p.stats.n_eff(H) >= 10), key=lambda p: p.stats.z(H), reverse=True)[:5]
        if top:
            out.append("- top predictors by z:")
            for p in top:
                out.append(f"  - {p.id}  z={p.stats.z(H):+.2f} n_eff={p.stats.n_eff(H):.0f} status={p.status}")
        if trusted:
            out.append("- TRUSTED: " + ", ".join(p.id for p in trusted))
        out.append("")
    if loop.history:
        h = loop.history[-1]
        out.append(f"Last maintenance: trusted {len(h.trusted)}, demoted {len(h.demoted)}, "
                   f"retired {h.retired}, spawned {h.spawned}.")
    return "\n".join(out)


def state_summary(loop: RecursiveLoop) -> dict:
    """Compact machine-readable state for the director (a separate process, so it cannot open the loop's database)."""
    out = {"at": datetime.now(timezone.utc).isoformat(), "bars": loop.bars, "horizons": {}}
    for H in loop.H:
        P = loop.preds[H]
        e = P[f"ensemble@{H}"].stats
        alive = [p for p in P.values() if p.kind in ("stump", "logit") and p.status != "retired"]
        top = sorted((p for p in alive if p.stats.n_eff(H) >= 10), key=lambda p: p.stats.z(H), reverse=True)[:5]
        out["horizons"][str(H)] = {
            "ensemble": {"n_eff": round(e.n_eff(H)), "skill_nats": round(e.mean_d(), 5), "z": round(e.z(H), 2),
                         "hit": round(e.hit_rate() or 0, 4), "net_edge_per_confident_call": e.edge()},
            "trusted": [p.id for p in alive if p.status == "trusted"][:10],
            "top_by_z": [{"id": p.id, "z": round(p.stats.z(H), 2), "n_eff": round(p.stats.n_eff(H))} for p in top]}
    return out
