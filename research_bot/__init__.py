"""research_bot — a small OpenAlex crawler that grows EMERALD-AI's literature brain.

Scope is deliberately tiny: it is a *lit-review aid*, not a dissertation contribution. It finds
papers about METHODS for extreme-imbalance credit scoring (cost-sensitive learning, focal loss,
SMOTE variants, calibration, conformal, anomaly detection, reject inference, green-finance ML)
and files them so the review chapter is thorough.

Curated, hand-vetted papers live in ``literature/index.yaml``; auto-discovered ones are kept
separate in ``literature/auto_index.yaml`` so the human-vetted set is never polluted.
"""

__version__ = "0.1.0"
