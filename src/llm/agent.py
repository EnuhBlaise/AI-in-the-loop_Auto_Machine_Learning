"""LLM agent for autonomous experiment design using HuggingFace Inference API."""

import json
import os
import logging

from huggingface_hub import InferenceClient

from src.llm.prompts import (
    SYSTEM_PROMPT,
    PROPOSE_EXPERIMENT_PROMPT,
    ANALYZE_RESULTS_PROMPT,
    FINAL_REPORT_PROMPT,
)
from src.llm.parser import extract_json, validate_experiment_config, validate_analysis

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM-powered scientific advisor that proposes experiments,
    analyzes results, and generates reports.

    Uses HuggingFace Inference API for model access.
    """

    def __init__(self, config: dict):
        self.config = config.get("llm", {})
        self.model_id = self.config.get("model", "mistralai/Mistral-7B-Instruct-v0.3")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 4096)

        token = self.config.get("hf_token") or os.environ.get("HF_TOKEN")
        if not token:
            logger.warning(
                "No HF_TOKEN found. Set HF_TOKEN env var or hf_token in config. "
                "Falling back to fallback mode (random proposals)."
            )

        self.client = InferenceClient(token=token) if token else None
        self._fallback = self.client is None

    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the HuggingFace LLM and return the response text."""
        if self._fallback:
            raise RuntimeError("No HF_TOKEN — cannot call LLM")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat_completion(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    def propose_experiment(
        self,
        experiment_history: str,
        current_best: str,
        search_space: str,
        dataset_info: dict,
    ) -> dict:
        """
        Ask the LLM to propose the next experiment configuration.

        Args:
            experiment_history: Formatted string of past experiments
            current_best: String describing current best result
            search_space: Formatted search space definition
            dataset_info: Dict with num_features, num_classes, num_train, num_val, class_distribution

        Returns:
            Dict with 'reasoning' and 'config' keys
        """
        if self._fallback:
            return self._fallback_proposal(dataset_info)

        prompt = PROPOSE_EXPERIMENT_PROMPT.format(
            num_features=dataset_info["num_features"],
            num_classes=dataset_info["num_classes"],
            num_train=dataset_info["num_train"],
            num_val=dataset_info["num_val"],
            class_distribution=dataset_info["class_distribution"],
            search_space=search_space,
            experiment_history=experiment_history if experiment_history else "No experiments run yet.",
            current_best=current_best if current_best else "No results yet.",
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_text = self._call_llm(prompt)
                logger.info(f"LLM response (attempt {attempt + 1}):\n{response_text[:500]}")
                parsed = extract_json(response_text)
                validated = validate_experiment_config(parsed)
                return validated
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                logger.warning(f"LLM response parse failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error("All LLM retries failed, using fallback proposal")
                    return self._fallback_proposal(dataset_info)

    def analyze_results(self, experiment_history: str, current_best: str) -> dict:
        """
        Ask the LLM to analyze experiment results and decide whether to continue.

        Returns:
            Dict with 'analysis', 'should_continue', 'confidence', 'key_findings'
        """
        if self._fallback:
            return {
                "analysis": "Fallback mode — no LLM analysis available.",
                "should_continue": True,
                "confidence": "low",
                "key_findings": [],
            }

        prompt = ANALYZE_RESULTS_PROMPT.format(
            experiment_history=experiment_history,
            current_best=current_best,
        )

        try:
            response_text = self._call_llm(prompt)
            parsed = extract_json(response_text)
            return validate_analysis(parsed)
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return {
                "analysis": f"Analysis failed: {e}",
                "should_continue": True,
                "confidence": "low",
                "key_findings": [],
            }

    def generate_report(self, experiment_history: str, best_result: str) -> str:
        """Generate a final markdown report summarizing all experiments."""
        if self._fallback:
            return self._fallback_report(experiment_history, best_result)

        prompt = FINAL_REPORT_PROMPT.format(
            experiment_history=experiment_history,
            best_result=best_result,
        )

        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"LLM report generation failed: {e}")
            return self._fallback_report(experiment_history, best_result)

    # ------------------------------------------------------------------
    # Fallback methods (when no HF_TOKEN is available)
    # ------------------------------------------------------------------

    def _fallback_proposal(self, dataset_info: dict) -> dict:
        """Generate a reasonable experiment proposal without LLM."""
        import random

        architectures = ["mlp", "cnn1d", "transformer"]
        arch = random.choice(architectures)

        configs = {
            "mlp": {
                "model": {
                    "architecture": "mlp",
                    "mlp": {
                        "hidden_dims": random.choice([
                            [64, 32],
                            [128, 64, 32],
                            [256, 128, 64],
                            [128, 128, 64, 32],
                            [64, 64],
                        ]),
                        "dropout": round(random.uniform(0.1, 0.5), 2),
                        "activation": random.choice(["gelu", "relu", "silu"]),
                        "use_residual": random.choice([True, False]),
                        "use_batchnorm": True,
                    },
                },
            },
            "cnn1d": {
                "model": {
                    "architecture": "cnn1d",
                    "cnn1d": {
                        "channels": random.choice([
                            [32, 64],
                            [64, 128, 64],
                            [32, 64, 128],
                        ]),
                        "kernel_size": random.choice([1, 3]),
                        "dropout": round(random.uniform(0.1, 0.4), 2),
                    },
                },
            },
            "transformer": {
                "model": {
                    "architecture": "transformer",
                    "transformer": {
                        "d_model": random.choice([32, 64]),
                        "n_heads": random.choice([2, 4]),
                        "n_layers": random.choice([1, 2, 3]),
                        "dropout": round(random.uniform(0.1, 0.3), 2),
                        "dim_feedforward": random.choice([64, 128]),
                    },
                },
            },
        }

        config = configs[arch]
        config["training"] = {
            "batch_size": random.choice([32, 64, 128]),
            "epochs": random.choice([100, 150, 200]),
            "optimizer": random.choice(["adam", "adamw"]),
            "lr": round(10 ** random.uniform(-4, -2), 6),
            "weight_decay": round(10 ** random.uniform(-5, -3), 6),
            "scheduler": random.choice(["cosine", "onecycle"]),
            "label_smoothing": round(random.uniform(0.0, 0.2), 2),
            "gradient_clip": 1.0,
            "class_weighting": True,
            "oversampling": random.choice([True, False]),
        }
        config["data"] = {
            "scaler": random.choice(["standard", "robust"]),
            "augmentation": {
                "mixup": random.choice([True, False]),
                "mixup_alpha": round(random.uniform(0.1, 0.4), 2),
            },
        }

        return {
            "reasoning": f"Fallback mode: randomly exploring {arch} architecture with varied hyperparameters.",
            "config": config,
        }

    def _fallback_report(self, experiment_history: str, best_result: str) -> str:
        """Generate a basic report without LLM."""
        return f"""# Autonomous Bio-ML Research — Final Report

## Summary
This report was generated in fallback mode (no LLM available).
The autonomous system explored multiple configurations using random search.

## Best Result
{best_result}

## Experiment History
{experiment_history}

## Recommendation
To get LLM-powered scientific analysis, set the HF_TOKEN environment variable.
"""
