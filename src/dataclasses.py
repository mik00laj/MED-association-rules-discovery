
from dataclasses import dataclass
from functools import total_ordering


@total_ordering
@dataclass
class Item:
    item: frozenset
    support: float

    def __str__(self):
        return f"Item: {str(tuple(self.item)):<40} | Supp: {self.support:.3f}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: "Item") -> bool:
        return self.support == other.support

    def __lt__(self, other: "Item") -> bool:
        return self.support < other.support


@total_ordering
@dataclass
class Rule:
    pre: frozenset
    post: frozenset
    confidence: float

    def raw_str(self):
        return f"{str(tuple(self.pre))} ==> {str(tuple(self.post))}"

    def __str__(self):
        rule = f"{str(tuple(self.pre))} ==> {str(tuple(self.post))}"
        return f"{rule:<40} | Confidence: {self.confidence:.3f}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: "Rule") -> bool:
        return self.confidence == other.confidence

    def __lt__(self, other: "Rule") -> bool:
        return self.confidence < other.confidence


@dataclass
class RuleMetrics:
    rule: Rule
    relative_support: float
    lift_factor: float
    certainty_factor: float
    jaaccard: float
    odds_ratio: float

    def __str__(self):
        return f"{str(self.rule)} " + \
            f"| rSup: {self.relative_support:.2f} " + \
            f"| Lift: {self.lift_factor:.2f} " + \
            f"| Certainty: {self.certainty_factor:.2f} " + \
            f"| Jaccard: {self.jaaccard:.2f} " + \
            f"| Odds Ratio: {self.odds_ratio:.2f}"
