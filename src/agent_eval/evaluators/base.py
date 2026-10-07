"""Abstract base class for all evaluation metric processors."""

from abc import ABC, abstractmethod
from agent_eval.models import AgentTrajectory, BenchmarkCase, MetricScore


class BaseEvaluator(ABC):
    """Interface for evaluating an agent trajectory against benchmark criteria."""

    def __init__(self, name: str, threshold: float = 0.8, weight: float = 1.0):
        self.name = name
        self.threshold = threshold
        self.weight = weight

    @abstractmethod
    def evaluate(self, case: BenchmarkCase, trajectory: AgentTrajectory) -> MetricScore:
        """Compute metric score for a given test case and trajectory.

        Args:
            case: Benchmark case specification with ground truth/expectations.
            trajectory: Recorded agent execution trace.

        Returns:
            MetricScore object with score (0.0 to 1.0), pass/fail status, and diagnostic details.
        """
        pass
