"""End-to-end pipeline: raw data -> indexes -> models -> evaluation -> report.

    python -m finagent_pulse.pipeline              # full run
    python -m finagent_pulse.pipeline --from rag   # resume from a stage
    python -m finagent_pulse.pipeline --force      # ignore all caches

Stages are individually cached, so a rerun only redoes what is missing.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import time

from finagent_pulse import config

log = logging.getLogger("pipeline")

STAGES = ["ingest", "sentiment", "features", "rag", "forecast", "evaluate", "report"]


def stage_ingest(force: bool) -> None:
    from finagent_pulse.data.ingest import ingest_all
    news, market = ingest_all(force=force)
    log.info("headlines=%d  market_sessions=%d (%s)",
             len(news), len(market), market["source"].iloc[0])


def stage_sentiment(force: bool) -> None:
    from finagent_pulse.data.preprocess import clean_headlines
    from finagent_pulse.models.sentiment import score_corpus
    clean_headlines(force=force)
    scored = score_corpus(force=force)
    log.info("scored %d headlines; mean sentiment %.4f",
             len(scored), scored["sentiment"].mean())


def stage_features(force: bool) -> None:
    from finagent_pulse.data.preprocess import merge_features
    df = merge_features(force=force)
    log.info("feature table %d sessions %s -> %s",
             len(df), df["date"].min().date(), df["date"].max().date())


def stage_rag(force: bool) -> None:
    from finagent_pulse.rag.index import build_all
    art = build_all(force=force)
    g = art["kg"]
    log.info("indexes ready: %d docs, KG %d nodes / %d edges",
             len(art["documents"]), g.number_of_nodes(), g.number_of_edges())


def stage_forecast(force: bool) -> None:
    from finagent_pulse.models import forecaster
    if forecaster.CKPT.exists() and not force:
        log.info("forecaster checkpoint present; skipping (use --force to retrain)")
        return
    m = forecaster.train()
    t = m["test"]
    log.info("test RMSE=%.5f  R2(return)=%+.4f  direction@h7=%.3f",
             t["rmse_return_overall"], t["r2_return_overall"],
             t["directional_accuracy_h7"])


def stage_evaluate(force: bool) -> None:
    from finagent_pulse.evaluation import run_all
    from finagent_pulse.rag.evaluate import run_ablation
    if config.REPORTS.joinpath("rag_ablation.csv").exists() and not force:
        log.info("RAG ablation present; skipping")
    else:
        run_ablation()
    run_all()


def stage_report(force: bool, brief_dir: str | None = None,
                 narration_dir: str | None = None) -> None:
    from finagent_pulse.agents.committee import executive_report, run_committee
    from finagent_pulse.data.preprocess import merge_features

    df = merge_features()
    as_of = df["date"].iloc[-1]
    state = run_committee(df, as_of, brief_dir=brief_dir,
                          narration_dir=narration_dir)
    if brief_dir:
        log.info("wrote narration briefs to %s", brief_dir)
    md = executive_report(state)
    out = config.REPORTS / f"executive_report_{as_of.date()}.md"
    out.write_text(md)
    log.info("wrote %s (directive: %s)", out.name, state["risk"]["directive"])


RUNNERS = {
    "ingest": stage_ingest, "sentiment": stage_sentiment, "features": stage_features,
    "rag": stage_rag, "forecast": stage_forecast, "evaluate": stage_evaluate,
    "report": stage_report,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="FinAgent-Pulse end-to-end pipeline")
    ap.add_argument("--from", dest="start", choices=STAGES, default="ingest",
                    help="resume from this stage")
    ap.add_argument("--only", choices=STAGES, help="run a single stage")
    ap.add_argument("--force", action="store_true", help="ignore caches")
    # Narration without an API key: export the briefs, write the prose against
    # them elsewhere, then render the report from what you wrote.
    ap.add_argument("--export-briefs", metavar="DIR",
                    help="write each agent's narration brief to DIR")
    ap.add_argument("--narration-dir", metavar="DIR",
                    help="render the report from prose files in DIR "
                         "(<agent>.md); missing files fall back to templates")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    runners = dict(RUNNERS)
    runners["report"] = functools.partial(
        stage_report, brief_dir=args.export_briefs,
        narration_dir=args.narration_dir)

    stages = [args.only] if args.only else STAGES[STAGES.index(args.start):]
    timings: dict[str, float] = {}

    for name in stages:
        log.info("=" * 62)
        log.info("STAGE: %s", name)
        log.info("=" * 62)
        t0 = time.time()
        runners[name](args.force)
        timings[name] = round(time.time() - t0, 1)
        log.info("stage '%s' completed in %.1fs", name, timings[name])

    (config.REPORTS / "pipeline_timings.json").write_text(json.dumps(timings, indent=2))
    log.info("Pipeline complete: %s", timings)


if __name__ == "__main__":
    main()
