#!/usr/bin/env python3
"""
BioML Autopilot — Entry point for training and autonomous research.

Usage:
    # Single experiment with default config
    python train.py

    # Single experiment with overrides
    python train.py --arch transformer --lr 0.0005 --epochs 150

    # Autonomous LLM-in-the-loop research loop
    python train.py --auto

    # Autonomous with budget
    python train.py --auto --max-experiments 10

    # View leaderboard
    python train.py --leaderboard
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(
        description="BioML Autopilot — Autonomous Bio-ML Research Platform"
    )
    parser.add_argument("--auto", action="store_true", help="Run autonomous LLM-in-the-loop research loop")
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML file")
    parser.add_argument("--arch", "-a", default=None, help="Model architecture (mlp/cnn1d/transformer/ensemble)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-experiments", "-n", type=int, default=None, help="Max experiments (auto mode)")
    parser.add_argument("--time-budget", "-t", type=int, default=None, help="Time budget per experiment (seconds)")
    parser.add_argument("--leaderboard", action="store_true", help="Show experiment leaderboard")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    if args.leaderboard:
        from src.experiment.tracker import ExperimentTracker
        from rich.console import Console
        from rich.table import Table

        console = Console()
        tracker = ExperimentTracker()
        results = tracker.get_leaderboard(top_n=20)

        if not results:
            console.print("[yellow]No experiments found.[/yellow]")
            return

        table = Table(title="Experiment Leaderboard")
        table.add_column("Rank", style="bold")
        table.add_column("ID")
        table.add_column("Architecture")
        table.add_column("val_auroc", style="green")
        table.add_column("val_f1")
        table.add_column("val_accuracy")
        table.add_column("Duration")

        for rank, exp in enumerate(results, 1):
            m = exp.get("metrics", {})
            arch = exp.get("config", {}).get("model", {}).get("architecture", "?")
            table.add_row(
                str(rank), exp["id"], arch,
                f"{m.get('val_auroc', 0):.5f}",
                f"{m.get('val_f1', 0):.5f}",
                f"{m.get('val_accuracy', 0):.5f}",
                f"{exp.get('duration', 0):.1f}s",
            )
        console.print(table)
        return

    # Build overrides
    overrides = {}
    if args.arch:
        overrides["model.architecture"] = args.arch
    if args.lr:
        overrides["training.lr"] = args.lr
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    if args.max_experiments:
        overrides["autonomous.max_experiments"] = args.max_experiments
    if args.time_budget:
        overrides["experiment.time_budget"] = args.time_budget

    from src.orchestrator import AutonomousOrchestrator
    from rich.console import Console

    console = Console()

    orch = AutonomousOrchestrator(
        config_path=args.config,
        overrides=overrides if overrides else None,
    )

    if args.auto:
        result = orch.run()
        console.print(f"\n[bold green]Autonomous run complete![/bold green]")
        console.print(f"  Experiments run: {result['num_experiments']}")
        console.print(f"  Best val_auroc: {result['best_metrics'].get('val_auroc', 'N/A')}")
        console.print(f"  Report: {result['report_path']}")
    else:
        config_overrides = {}
        if args.arch:
            config_overrides.setdefault("model", {})["architecture"] = args.arch
        metrics = orch.run_single(config_overrides if config_overrides else None)
        if metrics:
            console.print(f"\n[bold green]Training complete![/bold green]")
            for k, v in metrics.items():
                console.print(f"  {k}: {v}")
        else:
            console.print("[bold red]Training failed.[/bold red]")
            sys.exit(1)


if __name__ == "__main__":
    main()
