# Autonomous Bio-ML Research Program

## Overview
A fully autonomous machine learning research platform that uses an LLM (via HuggingFace)
as a scientific advisor to iteratively design, run, and evaluate experiments on biological
classification data — without human intervention.

## Task
Classification of biological samples (e.g., protein localization sites, tumor types,
drug responses) from tabular feature data (gene expression, protein properties, genomic
variants, etc.).

## Dataset
- Training: 60% of samples — used for model training
- Validation: 20% of samples — used for experiment comparison and LLM decision-making
- Test: 20% of samples — NEVER touched until final evaluation

## Metric
- Primary: **val_auroc** (validation AUC-ROC, macro one-vs-rest) — higher is better
- Secondary: val_f1 (macro), val_accuracy — logged for LLM analysis

## Autonomous Loop
The system operates as a closed-loop autonomous research agent:

1. **Initialize** — Load dataset, search space, and base configuration
2. **LLM Proposes** — HuggingFace LLM receives experiment history + search space,
   reasons about what to try next, and outputs a valid experiment config
3. **System Trains** — The proposed config is validated, the model is trained, and
   metrics are recorded
4. **LLM Analyzes** — The LLM reviews all experiment results, identifies patterns
   (what works, what doesn't), and decides whether to continue or stop
5. **Repeat or Conclude** — If the LLM determines further improvement is unlikely
   (or budget is exhausted), it generates a final report with conclusions about the
   best model, key findings, and recommendations

## LLM Integration
- **Provider**: HuggingFace Inference API (serverless or dedicated endpoint)
- **Role**: Scientific advisor — proposes experiments, analyzes results, draws conclusions
- **Autonomy**: The LLM makes ALL decisions — architecture, hyperparameters, strategies,
  and when to stop. No human intervention required during a run.
- **Output**: Structured JSON configs for experiment proposals + natural language analysis

## Model Architectures to Explore
- MLP with residual connections and batch normalization
- 1D CNN for tabular data
- Tabular Transformer (attention over features)
- Ensemble (soft voting / stacking)

## What the LLM Can Explore
- Model architectures (MLP, CNN, Transformer, ensemble)
- Learning rate schedules (cosine, step, plateau, one-cycle)
- Regularization (dropout, weight decay, mixup, label smoothing)
- Feature preprocessing strategies (standard, robust, minmax scaling)
- Class imbalance handling (weighted loss, oversampling)
- Batch size optimization
- Training duration and early stopping configuration

## Constraints
- Single GPU, 5-minute training budget per experiment
- Do NOT modify prepare.py
- Must call `evaluate()` from prepare.py at the end of each experiment
- Keep training under the time budget

## What NOT to Do
- Don't overfit to validation set by peeking at test data
- Don't use external pretrained models unless explicitly allowed
- Keep each experiment within the time budget
- Don't repeat identical experiments

## Deliverables
After the autonomous run completes, the system produces:
- A leaderboard of all experiments sorted by val_auroc
- The best model checkpoint
- A final LLM-generated analysis report with conclusions
- All experiment configs and metrics in structured JSON
