
"""Offline entry point for the data, equity, bonds, allocation, and report stages."""
import argparse
import logging
import os
from pathlib import Path
from time import perf_counter

# Avoid machine-dependent oversubscription during small repeated fits.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
STAGES = ("data", "equity", "bond", "allocation", "report")
LOG = logging.getLogger("pipeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "strategy_config.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from src.utils.config import read_config

    raw = args.raw_dir.resolve()
    config = read_config(args.config)
    processed = ROOT / "data" / "processed"
    outputs = ROOT / "outputs"
    for path in (processed, outputs):
        if path.resolve().is_relative_to(raw):
            raise ValueError(f"Generated output would overlap read-only raw data: {path}")
        path.mkdir(parents=True, exist_ok=True)

    stages = STAGES if args.stage == "all" else (args.stage,)
    for step in stages:
        LOG.info("Running stage: %s", step)
        started = perf_counter()
        if step == "data":
            import json
            from scripts.audit_raw import inspect
            from src.data.pipeline import prepare_data

            audit = inspect(raw)
            audit["industry_model_provenance"] = {
                "static_proxy_enabled": config["equity"].get("use_static_industry_proxy", False),
                "static_proxy_table": "stock_industry.parquet",
                "source_date_field": "updateDate",
                "point_in_time_join": "latest available updateDate not later than decision_date",
                "static_proxy_join": "latest snapshot by code, including labels dated after decisions",
                "static_proxy_is_point_in_time": False,
            }
            (outputs / "raw_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            prepare_data(raw, processed, config)
        elif step == "equity":
            from src.equity.pipeline import equity_stage

            equity_stage(raw, processed, outputs, config)
        elif step == "bond":
            from src.fixed_income.pipeline import bond_stage

            bond_stage(processed, outputs, config)
        elif step == "allocation":
            from src.allocation.pipeline import allocation_stage

            allocation_stage(processed, outputs, config)
        else:
            from src.analytics.pipeline import analytics_stage
            from src.analytics.reporting import build_reports
            from src.analytics.visualization import generate_figures

            analytics_stage(processed, outputs, config, args.config)
            generate_figures(outputs, ROOT / "reports" / "figures")
            build_reports(ROOT, outputs)
        LOG.info("Completed %s in %.1f seconds", step, perf_counter() - started)


if __name__ == "__main__":
    main()
