"""Parse and validate LLM responses into structured experiment configs."""

import json
import re


def extract_json(text: str) -> dict:
    """Extract the first valid JSON object from LLM output text."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON block in markdown code fences
    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
        r"\{.*\}",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            candidate = match.strip()
            if not candidate.startswith("{"):
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


def validate_experiment_config(parsed: dict) -> dict:
    """
    Validate and normalize an experiment config from LLM output.

    Returns a clean config dict that can be merged with the base config.
    Raises ValueError if the config is invalid.
    """
    if "config" not in parsed:
        raise ValueError("LLM response missing 'config' key")

    config = parsed["config"]
    reasoning = parsed.get("reasoning", "No reasoning provided")

    # Validate architecture
    arch = config.get("model", {}).get("architecture", "mlp")
    valid_archs = {"mlp", "cnn1d", "transformer", "ensemble"}
    if arch not in valid_archs:
        raise ValueError(f"Invalid architecture '{arch}'. Must be one of {valid_archs}")

    # Validate model-specific config exists
    if arch != "ensemble" and arch not in config.get("model", {}):
        # Use defaults — not an error, just means LLM didn't specify
        pass

    # Validate training params
    training = config.get("training", {})

    if "lr" in training:
        lr = float(training["lr"])
        if not (1e-6 <= lr <= 1.0):
            training["lr"] = max(1e-6, min(lr, 1.0))

    if "batch_size" in training:
        bs = int(training["batch_size"])
        if bs < 8:
            training["batch_size"] = 8
        elif bs > 512:
            training["batch_size"] = 512

    if "epochs" in training:
        epochs = int(training["epochs"])
        training["epochs"] = max(10, min(epochs, 1000))

    if "dropout" in config.get("model", {}).get(arch, {}):
        d = float(config["model"][arch]["dropout"])
        config["model"][arch]["dropout"] = max(0.0, min(d, 0.8))

    if "optimizer" in training:
        valid_opts = {"adam", "adamw", "sgd"}
        if training["optimizer"] not in valid_opts:
            training["optimizer"] = "adamw"

    if "scheduler" in training:
        valid_scheds = {"cosine", "step", "plateau", "onecycle"}
        if training["scheduler"] not in valid_scheds:
            training["scheduler"] = "cosine"

    if "scaler" in config.get("data", {}):
        valid_scalers = {"standard", "robust", "minmax", "none"}
        if config["data"]["scaler"] not in valid_scalers:
            config["data"]["scaler"] = "standard"

    return {
        "reasoning": reasoning,
        "config": config,
    }


def validate_analysis(parsed: dict) -> dict:
    """Validate an analysis response from the LLM."""
    return {
        "analysis": parsed.get("analysis", "No analysis provided"),
        "should_continue": parsed.get("should_continue", True),
        "confidence": parsed.get("confidence", "low"),
        "key_findings": parsed.get("key_findings", []),
    }
