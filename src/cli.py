"""CLI entry points for BioML Autopilot."""

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

# Load .env file so HF_TOKEN and other env vars are available automatically
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose):
    """BioML Autopilot — Autonomous Bio-ML Research Platform"""
    setup_logging(verbose)


@main.command()
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--arch", "-a", default=None, help="Model architecture override")
@click.option("--lr", type=float, default=None, help="Learning rate override")
@click.option("--epochs", type=int, default=None, help="Max epochs override")
@click.option("--batch-size", type=int, default=None, help="Batch size override")
def train(config, arch, lr, epochs, batch_size):
    """Run a single training experiment."""
    from src.orchestrator import AutonomousOrchestrator

    overrides = {}
    if arch:
        overrides["model.architecture"] = arch
    if lr:
        overrides["training.lr"] = lr
    if epochs:
        overrides["training.epochs"] = epochs
    if batch_size:
        overrides["training.batch_size"] = batch_size

    orch = AutonomousOrchestrator(config_path=config, overrides=overrides if overrides else None)
    metrics = orch.run_single()

    if metrics:
        console.print(f"\n[bold green]Training complete![/bold green]")
        for k, v in metrics.items():
            console.print(f"  {k}: {v}")
    else:
        console.print("[bold red]Training failed.[/bold red]")
        sys.exit(1)


@main.command()
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--max-experiments", "-n", type=int, default=None, help="Max experiments to run")
@click.option("--time-budget", "-t", type=int, default=None, help="Time budget per experiment (seconds)")
def auto(config, max_experiments, time_budget):
    """Run the autonomous research loop with LLM-in-the-loop."""
    from src.orchestrator import AutonomousOrchestrator

    overrides = {}
    if max_experiments:
        overrides["autonomous.max_experiments"] = max_experiments
    if time_budget:
        overrides["experiment.time_budget"] = time_budget

    orch = AutonomousOrchestrator(config_path=config, overrides=overrides if overrides else None)
    result = orch.run()

    console.print(f"\n[bold green]Autonomous run complete![/bold green]")
    console.print(f"  Experiments run: {result['num_experiments']}")
    console.print(f"  Best val_auroc: {result['best_metrics'].get('val_auroc', 'N/A')}")
    console.print(f"  Report: {result['report_path']}")


@main.command()
@click.option("--top", "-n", type=int, default=10, help="Number of results to show")
def leaderboard(top):
    """Show the experiment leaderboard."""
    from src.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    results = tracker.get_leaderboard(top_n=top)

    if not results:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"Experiment Leaderboard (Top {top})")
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
            str(rank),
            exp["id"],
            arch,
            f"{m.get('val_auroc', 0):.5f}",
            f"{m.get('val_f1', 0):.5f}",
            f"{m.get('val_accuracy', 0):.5f}",
            f"{exp.get('duration', 0):.1f}s",
        )

    console.print(table)


if __name__ == "__main__":
    main()
