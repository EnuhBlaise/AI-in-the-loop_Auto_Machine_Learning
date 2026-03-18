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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.manifold import TSNE

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

    def _run_single_experiment(self, experiment_config: dict, experiment_id: str) -> tuple:
        """
        Run a single training experiment with the given config.

        Returns (metrics, history, model, data) or (None, None, None, None) on failure.
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

            return metrics, trainer.history, model, data

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
            console.print(f"  [red]Experiment failed: {e}[/red]")
            return None, None, None, None

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
        best_model = None
        best_data = None
        best_config = None
        best_experiment_id = None
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

            # --- Verbose config printout ---
            arch = experiment_config.get("model", {}).get("architecture", "mlp")
            train_cfg = experiment_config.get("training", {})
            data_cfg = experiment_config.get("data", {})
            model_cfg = experiment_config.get("model", {})
            arch_params = model_cfg.get(arch, {})

            config_lines = [
                f"[bold]Experiment ID:[/bold] {experiment_id}",
                f"[bold]Architecture:[/bold] {arch}",
            ]
            # Architecture-specific params
            for k, v in arch_params.items():
                config_lines.append(f"  {k}: {v}")
            # Training params
            config_lines.append(f"[bold]Optimizer:[/bold] {train_cfg.get('optimizer', 'adamw')}")
            config_lines.append(f"[bold]Learning rate:[/bold] {train_cfg.get('lr', 0.001)}")
            config_lines.append(f"[bold]Batch size:[/bold] {train_cfg.get('batch_size', 64)}")
            config_lines.append(f"[bold]Epochs:[/bold] {train_cfg.get('epochs', 200)}")
            config_lines.append(f"[bold]Scheduler:[/bold] {train_cfg.get('scheduler', 'cosine')}")
            config_lines.append(f"[bold]Weight decay:[/bold] {train_cfg.get('weight_decay', 0.0001)}")
            config_lines.append(f"[bold]Label smoothing:[/bold] {train_cfg.get('label_smoothing', 0.0)}")
            config_lines.append(f"[bold]Gradient clip:[/bold] {train_cfg.get('gradient_clip', 0.0)}")
            config_lines.append(f"[bold]Class weighting:[/bold] {train_cfg.get('class_weighting', False)}")
            config_lines.append(f"[bold]Oversampling:[/bold] {train_cfg.get('oversampling', False)}")
            config_lines.append(f"[bold]Scaler:[/bold] {data_cfg.get('scaler', 'standard')}")
            aug = data_cfg.get("augmentation", {})
            config_lines.append(f"[bold]Mixup:[/bold] {aug.get('mixup', False)} (alpha={aug.get('mixup_alpha', 0.2)})")

            console.print(Panel(
                "\n".join(config_lines),
                title=f"Experiment {i} Configuration",
                border_style="blue",
            ))

            # --- Step 3: Train ---
            start_time = time.time()
            metrics, history, model, exp_data = self._run_single_experiment(experiment_config, experiment_id)
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
            console.print(Panel(
                f"[bold]val_auroc:[/bold] {val_auroc:.5f}\n"
                f"[bold]val_f1:[/bold] {metrics.get('val_f1', 0.0):.5f}\n"
                f"[bold]val_accuracy:[/bold] {metrics.get('val_accuracy', 0.0):.5f}\n"
                f"[bold]Duration:[/bold] {duration:.1f}s",
                title=f"Experiment {experiment_id} Results",
                border_style="green",
            ))

            # --- Generate training curves ---
            if history and history.get("epoch"):
                exp_dir = Path(self.tracker.base_dir) / experiment_id
                self._plot_training_curves(history, exp_dir, experiment_id, arch)
                console.print(f"  [cyan]Training curves saved to {exp_dir}/training_curves.png[/cyan]")

            # Track improvement
            if val_auroc > best_auroc + self.plateau_threshold:
                best_auroc = val_auroc
                best_model = model
                best_data = exp_data
                best_config = experiment_config
                best_experiment_id = experiment_id
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

        # --- Winning model visualizations ---
        if best_model is not None and best_data is not None:
            winner_dir = Path(self.tracker.base_dir) / best_experiment_id
            console.print(f"\n[bold cyan]Generating winning model visualizations ({best_experiment_id})...[/bold cyan]")
            self._plot_winner_analysis(best_model, best_data, best_config, winner_dir)
            console.print(f"  [cyan]Winner analysis saved to {winner_dir}/[/cyan]")

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
        metrics, history, model, exp_data = self._run_single_experiment(config, experiment_id)
        duration = time.time() - start_time

        if metrics:
            self.tracker.log_metrics(experiment_id, metrics)
            self.tracker.finish_experiment(experiment_id, metrics, duration)
            # Generate training curves for single experiments too
            if history and history.get("epoch"):
                exp_dir = Path(self.tracker.base_dir) / experiment_id
                arch = config.get("model", {}).get("architecture", "mlp")
                self._plot_training_curves(history, exp_dir, experiment_id, arch)
                console.print(f"  [cyan]Training curves saved to {exp_dir}/training_curves.png[/cyan]")
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

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _plot_training_curves(self, history: dict, exp_dir: Path, exp_id: str, arch: str):
        """Generate training loss and validation metric curves for an experiment."""
        exp_dir.mkdir(parents=True, exist_ok=True)
        epochs = history["epoch"]
        if not epochs:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Curves — {exp_id} ({arch})", fontsize=14, fontweight="bold")

        # 1. Training loss
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], color="#E53935", linewidth=2, label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Validation AUC-ROC
        ax = axes[0, 1]
        ax.plot(epochs, history["val_auroc"], color="#1E88E5", linewidth=2, label="Val AUC-ROC")
        best_idx = int(np.argmax(history["val_auroc"]))
        ax.scatter([epochs[best_idx]], [history["val_auroc"][best_idx]], color="gold",
                   s=100, zorder=5, edgecolors="black", label=f"Best: {history['val_auroc'][best_idx]:.4f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Validation AUC-ROC")
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Validation Accuracy & F1
        ax = axes[1, 0]
        ax.plot(epochs, history["val_accuracy"], color="#43A047", linewidth=2, label="Val Accuracy")
        ax.plot(epochs, history["val_f1"], color="#FB8C00", linewidth=2, label="Val F1 (macro)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Validation Accuracy & F1")
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Learning rate schedule
        ax = axes[1, 1]
        ax.plot(epochs, history["lr"], color="#8E24AA", linewidth=2, label="Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(exp_dir / "training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def _plot_winner_analysis(self, model, data: dict, config: dict, output_dir: Path):
        """Generate comprehensive visualizations of what the winning model learned."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        model = model.to(self.device)

        X_val = data["X_val"]
        y_val = data["y_val"]
        class_names = data["class_names"]
        num_classes = data["num_classes"]

        X_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        logits = model(X_tensor)
        proba = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(proba, axis=1)

        # ---- Figure 1: Confusion Matrix ----
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_val, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
        ax.set_title("Winning Model — Confusion Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(output_dir / "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 2: Per-class ROC curves ----
        fig, ax = plt.subplots(figsize=(10, 8))
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_val, classes=list(range(num_classes)))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        for cls_idx in range(num_classes):
            if y_bin[:, cls_idx].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], proba[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[cls_idx], linewidth=1.5,
                    label=f"{class_names[cls_idx]} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Winning Model — Per-Class ROC Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / "roc_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 3: Feature importance via gradient-based saliency ----
        fig, ax = plt.subplots(figsize=(10, 6))
        X_grad = torch.tensor(X_val, dtype=torch.float32, device=self.device, requires_grad=True)
        model.zero_grad()
        out = model(X_grad)
        # Sum of all class logits w.r.t. input — average absolute gradient
        out.sum().backward()
        importance = X_grad.grad.abs().mean(dim=0).cpu().numpy()

        # Try to get feature names from the dataframe
        try:
            import pandas as pd
            data_cfg = config.get("data", {})
            df = pd.read_csv(data_cfg["path"], nrows=1)
            feature_names = [c for c in df.columns if c != data_cfg.get("target_column", "name")]
        except Exception:
            feature_names = [f"Feature {j}" for j in range(len(importance))]

        sorted_idx = np.argsort(importance)[::-1]
        ax.barh(range(len(importance)), importance[sorted_idx], color="#1976D2", edgecolor="white")
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels([feature_names[j] for j in sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Mean |Gradient|")
        ax.set_title("Winning Model — Feature Importance (Gradient Saliency)", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 4: t-SNE of learned representations ----
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: t-SNE of raw input features
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_val) - 1))
        X_2d_raw = tsne.fit_transform(X_val)
        ax = axes[0]
        scatter = ax.scatter(X_2d_raw[:, 0], X_2d_raw[:, 1], c=y_val, cmap="tab10",
                            s=15, alpha=0.7, edgecolors="none")
        ax.set_title("t-SNE of Raw Input Features", fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        # Right: t-SNE of model's penultimate layer representations
        # Extract representations by hooking the layer before the classification head
        representations = []
        def _hook(module, inp, out):
            representations.append(inp[0].detach().cpu())

        # Find the classification head (last linear layer named 'head')
        head = None
        for name, module in model.named_modules():
            if name == "head":
                head = module
                break
        if head is None:
            # Fallback: use last Linear layer
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Linear):
                    head = module
                    break

        if head is not None:
            handle = head.register_forward_hook(_hook)
            with torch.no_grad():
                model(X_tensor)
            handle.remove()

            if representations:
                reps = representations[0].numpy()
                tsne2 = TSNE(n_components=2, random_state=42, perplexity=min(30, len(reps) - 1))
                X_2d_learned = tsne2.fit_transform(reps)
                ax = axes[1]
                ax.scatter(X_2d_learned[:, 0], X_2d_learned[:, 1], c=y_val, cmap="tab10",
                          s=15, alpha=0.7, edgecolors="none")
                ax.set_title("t-SNE of Learned Representations", fontsize=12, fontweight="bold")
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")

        # Shared colorbar
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label("Class")
        if len(class_names) <= 15:
            cbar.set_ticks(range(len(class_names)))
            cbar.set_ticklabels(class_names)

        fig.suptitle("Winning Model — What It Learned (Representation Space)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(output_dir / "learned_representations.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 5: Prediction confidence distribution ----
        fig, ax = plt.subplots(figsize=(10, 6))
        max_proba = np.max(proba, axis=1)
        correct = preds == y_val
        ax.hist(max_proba[correct], bins=30, alpha=0.7, color="#43A047", label="Correct", density=True)
        ax.hist(max_proba[~correct], bins=30, alpha=0.7, color="#E53935", label="Incorrect", density=True)
        ax.set_xlabel("Prediction Confidence (max probability)")
        ax.set_ylabel("Density")
        ax.set_title("Winning Model — Prediction Confidence Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / "confidence_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        console.print(f"  Saved: confusion_matrix.png, roc_curves.png, feature_importance.png, "
                      f"learned_representations.png, confidence_distribution.png")

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
