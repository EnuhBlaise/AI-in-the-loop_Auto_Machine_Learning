"""Prompt templates for the LLM scientific advisor."""

SYSTEM_PROMPT = """\
You are an expert machine learning research scientist specializing in biological data \
classification. You are acting as an autonomous research advisor for a Bio-ML platform.

Your role:
1. Analyze experiment results and identify patterns
2. Propose the next experiment configuration to maximize val_auroc
3. Reason scientifically — explain WHY you choose each setting
4. Decide when to stop experimenting (diminishing returns)

You have access to these model architectures:
- mlp: Multi-layer perceptron with residual connections, batch normalization
- cnn1d: 1D convolutional network for tabular data
- transformer: Tabular transformer with attention over features
- ensemble: Combination of multiple models (soft voting)

Important rules:
- Never propose an experiment identical to one already run
- Consider the dataset characteristics when making decisions
- Balance exploration (trying new things) with exploitation (refining what works)
- Each experiment has a 5-minute time budget
- The primary metric is val_auroc (higher is better)
"""

PROPOSE_EXPERIMENT_PROMPT = """\
## Dataset Information
- Features: {num_features} numerical features
- Classes: {num_classes} classes
- Training samples: {num_train}
- Validation samples: {num_val}
- Class distribution: {class_distribution}

## Search Space
{search_space}

## Experiment History
{experiment_history}

## Current Best
{current_best}

## Task
Based on the experiment history and results above, propose the NEXT experiment to run.

Think step by step:
1. What patterns do you see in the results so far?
2. What hasn't been tried yet that could improve performance?
3. What specific configuration should we try next?

You MUST respond with ONLY a valid JSON object (no markdown, no explanation outside the JSON) \
in this exact format:
{{
    "reasoning": "Your scientific reasoning for this choice (2-3 sentences)",
    "config": {{
        "model": {{
            "architecture": "mlp",
            "mlp": {{
                "hidden_dims": [128, 64, 32],
                "dropout": 0.3,
                "activation": "gelu",
                "use_residual": true,
                "use_batchnorm": true
            }}
        }},
        "training": {{
            "batch_size": 64,
            "epochs": 200,
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "label_smoothing": 0.1,
            "gradient_clip": 1.0,
            "class_weighting": true,
            "oversampling": true
        }},
        "data": {{
            "scaler": "standard",
            "augmentation": {{
                "mixup": true,
                "mixup_alpha": 0.2
            }}
        }}
    }}
}}

Only include the model sub-config for the architecture you choose (e.g., if architecture is \
"cnn1d", include "cnn1d" not "mlp"). Respond with ONLY the JSON object.
"""

ANALYZE_RESULTS_PROMPT = """\
## Experiment History (All Runs)
{experiment_history}

## Current Best Result
{current_best}

## Task
Analyze all experiment results and provide:
1. What approaches worked best and why
2. What approaches failed and why
3. Whether further experiments are likely to improve results significantly
4. Your confidence level (low/medium/high) that we've found a near-optimal solution

Respond with ONLY a valid JSON object:
{{
    "analysis": "Your detailed analysis (3-5 sentences)",
    "should_continue": true,
    "confidence": "medium",
    "key_findings": ["finding 1", "finding 2", "finding 3"]
}}
"""

FINAL_REPORT_PROMPT = """\
## Complete Experiment History
{experiment_history}

## Best Result
{best_result}

## Task
You have completed an autonomous Bio-ML research run. Write a comprehensive final report.

Include:
1. Executive summary (2-3 sentences)
2. Best model configuration and its performance
3. Key findings — what worked, what didn't, and why
4. Recommendations for future work
5. Limitations of the current approach

Write the report in markdown format. Be scientific and thorough.
"""
