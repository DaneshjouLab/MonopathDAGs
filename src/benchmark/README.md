## Modular Benchmarking: Design Rules and Guidelines

To support benchmarking across any task, data function, or model type, a modular benchmark must be built from interchangeable, well-scoped components. The following rules define how to structure such a benchmark to support flexibility, extensibility, and insight.

---

### Rule 1: Decompose into independent axes
Separate all components of the system under test into **three disjoint axes**:

- **Data function**: A parameterized function or generator that defines input-output pairs under specific sampling, augmentation, or preprocessing schemes.
- **Model interface**: A callable or wrapped object that takes input and returns predictions, with a clear API across model types.
- **Evaluation kernel**: A set of metrics or transformation functions that take predictions and targets and return scores, possibly per slice.

Each axis must be independently swappable without requiring changes to the others.

---

### Rule 2: Treat data as a first-class function
Data should not be loaded as static artifacts but built as **declarative recipes** with clearly scoped variability:

- Input: Sampling strategy, transformation pipeline, split logic, perturbation level
- Output: Deterministic or stochastic dataset objects conforming to a unified format
- Example: `data_fn(task='translation', domain='medical', noise=0.1) -> Dataset`

This ensures all experiments can vary data meaningfully and reproducibly.

---

### Rule 3: Enforce interface constraints
Every component must conform to a simple, well-documented API:

- **Model**: `predict(batch) -> outputs`
- **Evaluator**: `evaluate(preds, targets, **kwargs) -> score_dict`
- **Data function**: `data_config -> dataset object or iterator`

Adapters should be used to wrap legacy models or datasets into these interfaces.

---

### Rule 4: Capture all variation as configuration
All variable aspects (e.g., seed, data slice, model variant, metric set) must be represented as **configurable parameters**, not code changes. This allows:

- Automatic sweep generation
- Logging and reproducibility
- Interpolation across experimental conditions

Prefer structured configs (e.g., YAML, Pydantic) over hardcoded logic.

---

### Rule 5: Benchmark = Cartesian product of controlled variations
A benchmark run is defined as the evaluation over a **Cartesian product** of selected variations from each axis:

- `(model_i, data_j, eval_k) → score_ijk`

This allows benchmarking to expose not just best performers but meaningful **interactions** between choices.

---

### Rule 6: Support hierarchical slicing and stratification
To assess robustness and generalization, benchmarks must allow evaluation over:

- Subpopulations (e.g., by label, length, domain, metadata)
- Perturbations (e.g., noisy, adversarial, low-resource)
- Shifts (e.g., domain, temporal, source-target pairs)

Evaluators must optionally support slice-aware evaluation: `evaluate(preds, targets, slices=...)`.

---

### Rule 7: Log provenance at every level
Every benchmark result must be traceable back to:

- The exact model config
- The data function and its parameters
- The evaluation strategy and metric set
- The random seed and runtime environment

All runs should emit structured metadata with full parameter context (e.g., JSON, database, or hashable IDs).

---

### Rule 8: No single score without decomposition
Avoid single-number metrics without exposing score components:

- Report per-slice, per-class, or per-perturbation breakdowns
- Visualize variance, not just central tendencies
- Make performance *explainable*, not just measurable

Benchmarks should yield diagnostic insight, not just rankings.

---

These rules make benchmarking **composable**, **transparent**, and **task-agnostic**, enabling research to scale across problem types while remaining grounded in meaningful comparisons.
