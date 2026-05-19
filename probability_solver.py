"""Mjete te riperdorshme per Probabilitetin Total dhe Formulen e Bayes-it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ScenarioItem:
    """Nje burim/hipoteze ne nje skenar probabiliteti."""

    name: str
    prior: float
    likelihood: float

    @property
    def contribution(self) -> float:
        """Kthen P(Bi) * P(E|Bi)."""
        return self.prior * self.likelihood


class ProbabilitySolver:
    """Zgjidh skenare me Probabilitet Total dhe Bayes.

    Per cdo element pritet:
    - prior: P(Bi), probabiliteti fillestar i hipotezes/burimit
    - likelihood: P(E|Bi), probabiliteti i ngjarjes kur hipoteza eshte e vertete
    """

    def __init__(self, items: Iterable[ScenarioItem]):
        self.items = list(items)
        if not self.items:
            raise ValueError("Duhet te jepet te pakten nje element.")

        for item in self.items:
            self._validate_probability(item.prior, f"P({item.name})")
            self._validate_probability(item.likelihood, f"P(E|{item.name})")

    @staticmethod
    def _validate_probability(value: float, label: str) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"{label} duhet te jete midis 0 dhe 1.")

    @staticmethod
    def percent_to_probability(value: float) -> float:
        """Konverton nje vlere ne perqindje, p.sh. 4, ne probabilitet 0.04."""
        return value / 100

    @staticmethod
    def format_percent(value: float) -> str:
        """Formaton nje probabilitet si perqindje me dy shifra pas presjes."""
        return f"{value * 100:.2f}%"

    def total_probability(self) -> float:
        """Llogarit P(E) = sum(P(Bi) * P(E|Bi))."""
        return sum(item.contribution for item in self.items)

    def bayes_probability(self, target_name: str) -> float:
        """Llogarit P(Bk|E) per elementin me emrin target_name."""
        total = self.total_probability()
        if total == 0:
            raise ZeroDivisionError("P(E) eshte 0, prandaj formula e Bayes-it nuk mund te aplikohet.")

        target = self.get_item(target_name)
        return target.contribution / total

    def get_item(self, target_name: str) -> ScenarioItem:
        """Gjen nje element sipas emrit."""
        for item in self.items:
            if item.name == target_name:
                return item
        raise KeyError(f"Nuk u gjet elementi: {target_name}")

    def contributions(self) -> list[dict[str, float | str]]:
        """Kthen kontributet individuale per tabela dhe grafike."""
        return [
            {
                "Emri": item.name,
                "P(Bi)": item.prior,
                "P(E|Bi)": item.likelihood,
                "Kontributi": item.contribution,
            }
            for item in self.items
        ]

    def total_formula_steps(self, event_label: str = "E") -> str:
        """Krijon tekstin e hapave per Probabilitetin Total."""
        terms = [
            f"{item.prior:.4f} × {item.likelihood:.4f}"
            for item in self.items
        ]
        return f"P({event_label}) = " + " + ".join(terms)

    def bayes_formula_steps(self, target_name: str, event_label: str = "E") -> str:
        """Krijon tekstin e hapave per Formulen e Bayes-it."""
        target = self.get_item(target_name)
        total = self.total_probability()
        return (
            f"P({target_name}|{event_label}) = "
            f"[P({target_name}) × P({event_label}|{target_name})] / P({event_label}) "
            f"= ({target.prior:.4f} × {target.likelihood:.4f}) / {total:.4f}"
        )

    def prior_sum(self) -> float:
        """Kthen shumen e probabiliteteve fillestare."""
        return sum(item.prior for item in self.items)
