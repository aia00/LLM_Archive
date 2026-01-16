"""
Logging and monitoring utilities for DAPO training.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import wandb


class TrainingLogger:
    """
    Logger for DAPO training that supports multiple backends.
    """

    def __init__(
        self,
        output_dir: str,
        project_name: str = "dapo-training",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = False,
    ):
        """
        Initialize the training logger.

        Args:
            output_dir: Directory to save logs
            project_name: Name of the project
            experiment_name: Name of the experiment (auto-generated if None)
            config: Training configuration dictionary
            use_wandb: Whether to use Weights & Biases
        """
        self.output_dir = output_dir
        self.project_name = project_name

        if experiment_name is None:
            experiment_name = f"dapo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name

        # Create log directory
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize log file
        self.log_file = os.path.join(self.log_dir, f"{experiment_name}.jsonl")

        # Save configuration
        if config is not None:
            config_file = os.path.join(self.log_dir, f"{experiment_name}_config.json")
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

        # Initialize wandb if requested
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
            )

    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics at a given step.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step
        """
        # Add timestamp and step
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_text(self, text: str, title: Optional[str] = None):
        """
        Log text content.

        Args:
            text: Text to log
            title: Optional title
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "text",
            "title": title,
            "content": text,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if self.use_wandb and title:
            wandb.log({title: wandb.Html(f"<pre>{text}</pre>")})

    def finish(self):
        """Finish logging and cleanup."""
        if self.use_wandb:
            wandb.finish()


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary for display.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6f}")
        else:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def compute_metrics_summary(
    rewards: list,
    advantages: list,
    clip_ratios: list,
) -> Dict[str, float]:
    """
    Compute summary statistics for training metrics.

    Args:
        rewards: List of rewards
        advantages: List of advantages
        clip_ratios: List of clip ratios

    Returns:
        Dictionary of summary metrics
    """
    import numpy as np

    summary = {}

    if rewards:
        summary["reward/mean"] = float(np.mean(rewards))
        summary["reward/std"] = float(np.std(rewards))
        summary["reward/min"] = float(np.min(rewards))
        summary["reward/max"] = float(np.max(rewards))

    if advantages:
        summary["advantage/mean"] = float(np.mean(advantages))
        summary["advantage/std"] = float(np.std(advantages))

    if clip_ratios:
        summary["clip_ratio/mean"] = float(np.mean(clip_ratios))
        summary["clip_ratio/fraction_clipped"] = float(
            np.mean([r > 1.0 or r < -1.0 for r in clip_ratios])
        )

    return summary


class MetricsTracker:
    """
    Track and compute running statistics for metrics.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics = {}
        self.counts = {}

    def add(self, name: str, value: float):
        """
        Add a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(value)

    def add_batch(self, name: str, values: list):
        """
        Add a batch of metric values.

        Args:
            name: Metric name
            values: List of metric values
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].extend(values)

    def get_mean(self, name: str) -> Optional[float]:
        """
        Get the mean of a metric.

        Args:
            name: Metric name

        Returns:
            Mean value or None if metric not found
        """
        if name not in self.metrics or not self.metrics[name]:
            return None

        import numpy as np
        return float(np.mean(self.metrics[name]))

    def get_summary(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary of summary statistics or None if metric not found
        """
        if name not in self.metrics or not self.metrics[name]:
            return None

        import numpy as np

        values = self.metrics[name]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

    def reset(self, name: Optional[str] = None):
        """
        Reset metrics.

        Args:
            name: Metric name to reset. If None, reset all metrics.
        """
        if name is None:
            self.metrics = {}
            self.counts = {}
        elif name in self.metrics:
            del self.metrics[name]
            if name in self.counts:
                del self.counts[name]

    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """
        Get summaries for all tracked metrics.

        Returns:
            Dictionary mapping metric names to their summaries
        """
        summaries = {}
        for name in self.metrics:
            summary = self.get_summary(name)
            if summary:
                summaries[name] = summary

        return summaries


if __name__ == "__main__":
    # Test the logger
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(
            output_dir=tmpdir,
            experiment_name="test_exp",
            config={"lr": 1e-6, "batch_size": 32},
        )

        # Log some metrics
        logger.log({"loss": 0.5, "reward": 0.8}, step=1)
        logger.log({"loss": 0.4, "reward": 0.85}, step=2)

        logger.log_text("This is a test log", title="Test")

        logger.finish()

        print(f"Logs saved to: {logger.log_file}")

    # Test the metrics tracker
    tracker = MetricsTracker()
    tracker.add_batch("reward", [0.5, 0.6, 0.7, 0.8])
    tracker.add("loss", 0.3)

    print("\nMetrics summary:")
    print(tracker.get_all_summaries())
