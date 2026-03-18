"""
Autonomous orchestration loop — the core of BioML Autopilot.

Coordinates the LLM agent, experiment tracker, data loading, model creation,
and training into a fully autonomous research loop.
"""

import copy
import json
import logging
import time
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_config, load_search_space, _deep_merge
from src.data.loader import DataManager
from src.experiment.tracker import ExperimentTracker
from src.experiment.registry import ExperimentRegistry
from src.llm.agent import LLMAgent
from src.models.factory import ModelFactory
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)
console = Console()


class AutonomousOrchestrator:
    """
    Runs a fully autonomous ML research loop:
    1. LLM proposes experiment config
    2. System trains the model
    3. Results are logged
    4. LLM analyzes and decides next step
    5. Repeat until budget exhausted or LLM stops
    """

    def __init__(self, config_path: str | None = None, overrides: dict | None = None):
        self.config = load_config(config_path, overrides)
        self.search_space = load_search_space()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.data_manager = DataManager(self.config)
        self.tracker = ExperimentTracker(self.config.get("experiment", {}).get("base_dir", "experiments"))
        self.registry = ExperimentRegistry()
        self.llm_agent = LLMAgent(self.config)

        # Load data once
        self.data = self.data_manager.prepare()

        # Dataset info for LLM
        self.dataset_info = {
            "num_features": self.data["num_features"],
            "num_classes": self.data["num_classes"],
            "num_train": len(self.data["y_train"]),
            "num_val": len(self.data["y_val"]),
            "class_distribution": self._format_class_distribution(),
        }

        # Autonomous settings
        auto_cfg = self.config.get("autonomous", {})
        self.max_experiments = auto_cfg.get("max_experiments", 20)
        self.stop_on_plateau = auto_cfg.get("stop_on_plateau", True)
        self.plateau_patience = auto_cfg.get("plateau_patience", 5)
        self.plateau_threshold = auto_cfg.get("plateau_threshold", 0.001)

    def _format_class_distribution(self) -> str:
        """Format class distribution for LLM consumption."""
        import numpy as np

        counts = np.bincount(self.data["y_train"], minlength=self.data["num_classes"])
        parts = []
        for name, count in zip(self.data["class_names"], counts):
            parts.append(f"{name}: {count}")
        return ", ".join(parts)

    def _format_search_space(self) -> str:
        """Format search space for LLM consumption."""
        return yaml.dump(self.search_space, default_flow_style=False)

    def _merge_experiment_config(self, llm_config: dict) -> dict:
        """Merge LLM-proposed config overrides into the base config."""
        return _deep_merge(self.config, llm_config)

    def _run_single_experiment(self, experiment_config: dict, experiment_id: str) -> dict | None:
        """
        Run a single training experiment with the given config.

        Returns metrics dict or None if the experiment failed.
        """
        try:
            # Re-prepare data with the experiment's config (may differ in scaler/oversampling)
            exp_data_manager = DataManager(experiment_config)
            data = exp_data_manager.prepare()

            # Create model
            model = ModelFactory.create(
                experiment_config,
                num_features=data["num_features"],
                num_classes=data["num_classes"],
            )
            model = model.to(self.device)

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            console.print(f"  Model parameters: {param_count:,}")

            # Train
            trainer = Trainer(model, experiment_config, self.device)
            metrics = trainer.train(
                train_loader=data["train_loader"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                num_classes=data["num_classes"],
                class_weights=data.get("class_weights"),
            )

            # Save best model
            if trainer.best_model_state is not None:
                self.tracker.log_model(experiment_id, trainer.best_model_state)

            return metrics

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
            console.print(f"  [red]Experiment failed: {e}[/red]")
            return None

    def run(self) -> dict:
        """
        Run the full autonomous research loop.

        Returns:
            Dict with best_metrics, best_config, num_experiments, final_report
        """
        console.print(Panel(
            "[bold green]BioML Autopilot — Autonomous Research Loop[/bold green]\n"
            f"Device: {self.device} | Max experiments: {self.max_experiments}\n"
            f"Dataset: {self.dataset_info['num_train']} train, "
            f"{self.dataset_info['num_val']} val, "
            f"{self.dataset_info['num_features']} features, "
            f"{self.dataset_info['num_classes']} classes",
            title="Starting",
        ))

        best_auroc = 0.0
        no_improvement_count = 0
        search_space_str = self._format_search_space()

        for i in range(1, self.max_experiments + 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold cyan]Experiment {i}/{self.max_experiments}[/bold cyan]")
            console.print(f"{'='*60}")

            # --- Step 1: LLM proposes experiment ---
            history_str = self.tracker.format_history_for_llm()
            best_str = self._format_best()

            console.print("[yellow]Querying LLM for next experiment proposal...[/yellow]")
            proposal = self.llm_agent.propose_experiment(
                experiment_history=history_str,
                current_best=best_str,
                search_space=search_space_str,
                dataset_info=self.dataset_info,
            )

            reasoning = proposal.get("reasoning", "N/A")
            console.print(f"  LLM reasoning: [italic]{reasoning}[/italic]")

            # Check for duplicates (merge first so fingerprint sees full nested config)
            candidate_config = self._merge_experiment_config(proposal["config"])
            if self.registry.has_been_tried(candidate_config):
                console.print("  [yellow]Duplicate config detected, requesting new proposal...[/yellow]")
                # Retry once with a note about duplication
                proposal = self.llm_agent.propose_experiment(
                    experiment_history=history_str + "\n\nNOTE: Your last proposal was a duplicate. Try something different.",
                    current_best=best_str,
                    search_space=search_space_str,
                    dataset_info=self.dataset_info,
                )

            # Merge with base config
            experiment_config = self._merge_experiment_config(proposal["config"])

            # --- Step 2: Start experiment ---
            experiment_id = self.tracker.start_experiment(experiment_config)
            console.print(f"  Experiment ID: [bold]{experiment_id}[/bold]")

            arch = experiment_config.get("model", {}).get("architecture", "mlp")
            lr = experiment_config.get("training", {}).get("lr", "?")
            bs = experiment_config.get("training", {}).get("batch_size", "?")
            console.print(f"  Config: arch={arch}, lr={lr}, batch_size={bs}")

            # --- Step 3: Train ---
            start_time = time.time()
            metrics = self._run_single_experiment(experiment_config, experiment_id)
            duration = time.time() - start_time

            if metrics is None:
                self.tracker.finish_experiment(experiment_id, {"val_auroc": 0.0, "error": True}, duration)
                self.registry.register(experiment_id, experiment_config)
                console.print(f"  [red]Experiment failed after {duration:.1f}s[/red]")
                continue

            # --- Step 4: Log results ---
            self.tracker.log_metrics(experiment_id, metrics)
            self.tracker.finish_experiment(experiment_id, metrics, duration)
            self.registry.register(experiment_id, experiment_config)

            val_auroc = metrics.get("val_auroc", 0.0)
            console.print(f"  [green]val_auroc: {val_auroc:.5f}[/green] (duration: {duration:.1f}s)")

            # Track improvement
            if val_auroc > best_auroc + self.plateau_threshold:
                best_auroc = val_auroc
                no_improvement_count = 0
                console.print(f"  [bold green]New best! val_auroc = {best_auroc:.5f}[/bold green]")
            else:
                no_improvement_count += 1

            # --- Step 5: LLM analyzes (every 3 experiments or at the end) ---
            if i % 3 == 0 or i == self.max_experiments:
                console.print("[yellow]LLM analyzing results...[/yellow]")
                analysis = self.llm_agent.analyze_results(
                    experiment_history=self.tracker.format_history_for_llm(),
                    current_best=self._format_best(),
                )
                console.print(f"  Analysis: [italic]{analysis['analysis'][:200]}[/italic]")
                console.print(f"  Should continue: {analysis['should_continue']} | Confidence: {analysis['confidence']}")

                if not analysis["should_continue"]:
                    console.print("[bold yellow]LLM recommends stopping — diminishing returns.[/bold yellow]")
                    break

            # Plateau check
            if self.stop_on_plateau and no_improvement_count >= self.plateau_patience:
                console.print(
                    f"[bold yellow]Stopping: no improvement for {self.plateau_patience} experiments[/bold yellow]"
                )
                break

        # --- Final report ---
        console.print(f"\n{'='*60}")
        console.print("[bold green]Autonomous run complete. Generating final report...[/bold green]")

        self._print_leaderboard()

        report = self.llm_agent.generate_report(
            experiment_history=self.tracker.format_history_for_llm(),
            best_result=self._format_best(),
        )
        self.tracker.save_report(report)
        console.print(Panel(report[:2000], title="Final Report"))

        best_exp = self.tracker.get_best_experiment()
        return {
            "best_metrics": best_exp.get("metrics", {}),
            "best_config": best_exp.get("config", {}),
            "num_experiments": len(self.tracker.get_history()),
            "report_path": str(Path(self.tracker.base_dir) / "final_report.md"),
        }

    def run_single(self, config_overrides: dict | None = None) -> dict:
        """Run a single experiment (non-autonomous mode)."""
        config = self.config
        if config_overrides:
            config = _deep_merge(config, config_overrides)

        experiment_id = self.tracker.start_experiment(config)
        start_time = time.time()
        metrics = self._run_single_experiment(config, experiment_id)
        duration = time.time() - start_time

        if metrics:
            self.tracker.log_metrics(experiment_id, metrics)
            self.tracker.finish_experiment(experiment_id, metrics, duration)
        else:
            self.tracker.finish_experiment(experiment_id, {"error": True}, duration)

        return metrics or {}

    def _format_best(self) -> str:
        """Format the current best experiment for LLM consumption."""
        best = self.tracker.get_best_experiment()
        if not best:
            return "No experiments completed yet."
        return (
            f"Experiment {best['id']}: val_auroc={best['metrics'].get('val_auroc', 'N/A')}, "
            f"val_f1={best['metrics'].get('val_f1', 'N/A')}, "
            f"val_accuracy={best['metrics'].get('val_accuracy', 'N/A')} | "
            f"Architecture: {best['config'].get('model', {}).get('architecture', 'N/A')}"
        )

    def _print_leaderboard(self):
        """Print a formatted leaderboard table."""
        leaderboard = self.tracker.get_leaderboard(top_n=10)
        if not leaderboard:
            return

        table = Table(title="Experiment Leaderboard (Top 10)")
        table.add_column("Rank", style="bold")
        table.add_column("ID")
        table.add_column("Architecture")
        table.add_column("val_auroc", style="green")
        table.add_column("val_f1")
        table.add_column("val_accuracy")
        table.add_column("Duration")

        for rank, exp in enumerate(leaderboard, 1):
            metrics = exp.get("metrics", {})
            arch = exp.get("config", {}).get("model", {}).get("architecture", "?")
            table.add_row(
                str(rank),
                exp["id"],
                arch,
                f"{metrics.get('val_auroc', 0):.5f}",
                f"{metrics.get('val_f1', 0):.5f}",
                f"{metrics.get('val_accuracy', 0):.5f}",
                f"{exp.get('duration', 0):.1f}s",
            )

        console.print(table)
