"""
Custom HuggingFace Trainer callbacks for experiment tracking.

Provides an MLflow callback that logs:
- Training loss at each logging step
- Validation loss at each eval step  
- GPU memory usage
- Learning rate schedule
"""
from __future__ import annotations

from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MLflowDetailedCallback(TrainerCallback):
    """
    Enhanced MLflow callback for detailed experiment tracking.

    Extends HuggingFace's built-in MLflow integration with:
    - GPU memory tracking per step
    - Learning rate logging
    - Per-epoch validation loss
    - Training speed (samples/sec)
    """

    def __init__(self, log_gpu_every_n_steps: int = 50) -> None:
        """
        Args:
            log_gpu_every_n_steps: How often to log GPU memory usage.
        """
        self.log_gpu_every_n_steps = log_gpu_every_n_steps
        self._step_count = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> TrainerControl:
        """
        Called on each logging event. Logs metrics to MLflow.
        """
        if logs is None:
            return control

        try:
            import mlflow
            step = state.global_step

            # Log all available metrics
            numeric_logs = {
                k: v for k, v in logs.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric_logs:
                mlflow.log_metrics(numeric_logs, step=step)

            # Log GPU memory periodically
            self._step_count += 1
            if self._step_count % self.log_gpu_every_n_steps == 0:
                self._log_gpu_memory(step)

        except Exception as e:
            logger.warning(f"MLflow logging failed at step {state.global_step}: {e}")

        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Called at the end of each epoch."""
        try:
            import mlflow
            mlflow.log_metric("epoch_completed", state.epoch, step=state.global_step)
            logger.info(
                f"Epoch {state.epoch:.0f} complete | "
                f"step={state.global_step} | "
                f"best_metric={state.best_metric}"
            )
        except Exception as e:
            logger.warning(f"MLflow epoch logging failed: {e}")
        return control

    def _log_gpu_memory(self, step: int) -> None:
        """Log current GPU memory usage to MLflow."""
        try:
            import mlflow
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                mlflow.log_metrics(
                    {"gpu_allocated_gb": allocated, "gpu_reserved_gb": reserved},
                    step=step,
                )
        except Exception:
            pass  # Non-critical
