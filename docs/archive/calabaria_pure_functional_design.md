
# Calabaria — Pure Functional Core, Usable Surface  
**Design & Engineering Plan (Draft)**

> _tl;dr_: We adopt a **pure functional core + pragmatic object façade** to guarantee
reproducibility and provenance, enable content‑addressed caching, and keep the user
experience (decorators, scenarios, extractors) delightful. This document motivates the
approach, details the API, and lays out an implementation plan with code sketches.

---

## 0) Why start by “being dumb” about engineering choices

Before choosing purity, imagine we took the seemingly easy path:

- **Mutable models**: a `Model` object with internal state that is mutated by scenarios,
  seeds, and runs.  
- **Hidden globals**: RNGs, env variables, temp dirs, and implicit defaults.  
- **Ad‑hoc provenance**: human-written run notes, filenames that embed params, and
  “trust me” comments.

This looks fast, but it breaks in all the ways that matter:

- **Non-reproducibility**: tiny hidden state (e.g., an RNG advanced elsewhere) yields
  different outputs for the same inputs.
- **Uncacheable**: if outputs depend on outside state, you cannot safely re-use past
  results; you burn CPU on redundant simulation.
- **Un-deployable**: distributed evaluators cannot reason about idempotency;
  retries double-count; partial failures corrupt runs.
- **Un-auditable**: provenance becomes narrative instead of machine-verifiable.

We can’t fix those after the fact; they stem from **effects leaking into the core**.

---

## 1) The Pure Functional Approach (PFA)

### 1.1 Principles

- **Functional core, OO façade**: All computation paths important for correctness are
  **pure** (output depends only on explicit inputs). The façade provides decorators,
  ergonomics, and registries—without introducing hidden state.
- **Immutability by default**: Use frozen dataclasses or tuples for specs; explicit new
  objects for “edits” instead of mutation.
- **Explicit effects at the boundary**: File I/O, subprocess calls, and RNG use are
  surfaced as arguments and **hashed** into provenance inputs. Effects become data.
- **Content‑addressed provenance**: Inputs are organized into a **hash tree**; the root
  hash identifies the run. Artifacts are stored under a sharded directory scheme.

### 1.2 Why PFA is essential for provenance

- **Deterministic mapping**: `f(inputs) → outputs` lets us store and retrieve purely by
  input hash (no flaky cache invalidation rules).
- **Auditable**: Root hash is recomputable from leaves (code, config, params, scenario
  patches, transforms, seeds, container digest, etc.).
- **Distributed-safe**: Identical tasks deduplicate across workers/services; retries are
  idempotent; partial results reconcile safely.

### 1.3 Python mechanisms we rely on

- `@dataclass(frozen=True)` for immutable metadata objects.
- `typing.Protocol` for pluggable boundaries (simulation services, transforms).
- `typing.Literal` + `Union` for safe, concise configuration.
- `hashlib` for leaf and root hashes (BLAKE2 or SHA‑256).
- `pathlib.Path` and **content-addressed** storage trees for artifacts.
- **No** runtime globals inside core paths; effects are explicit inputs.

---

## 2) Types & Parameters (Minimal, Future‑Proof)

We introduce a single scalar frontier for values we optimize or serialize:

```python
from typing import Union, Mapping, Protocol, Optional, Dict, Iterator
from dataclasses import dataclass, field

Scalar = Union[float, int, bool, str]  # boundary scalar
```

### 2.1 ParameterSpec & ParameterSpace

- Specs are immutable; they carry the **shape** of the space—bounds, kind, doc.
- Values are not embedded in specs; they come via `ParameterSet`.

```python
from typing import Literal, Tuple

@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    kind: Literal["real", "int", "cat"] = "real"
    doc: str = ""  # optional help text

@dataclass(frozen=True)
class ParameterSpace:
    specs: Tuple[ParameterSpec, ...]  # immutable ordered collection

    def by_name(self) -> Dict[str, ParameterSpec]:
        return {s.name: s for s in self.specs}

    def free_names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.specs)
```

### 2.2 ParameterSet (values) & ParameterView (fix/free)

- `ParameterSet` holds **values** only.  
- `ParameterView` decides which names are free vs fixed. It’s an immutable lens over a
  space+values.

```python
@dataclass(frozen=True)
class ParameterSet:
    values: Dict[str, Scalar]  # typically floats/ints for real/int kinds

@dataclass(frozen=True)
class ParameterView:
    space: ParameterSpace
    fixed: Dict[str, Scalar] = field(default_factory=dict)
    free: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_space(cls, space: ParameterSpace) -> "ParameterView":
        return cls(space=space, fixed={}, free=tuple(space.free_names()))

    def fix(self, **kv: Scalar) -> "ParameterView":
        # Create a new view with certain parameters fixed.
        names = set(self.space.by_name().keys())
        for k in kv:
            if k not in names:
                raise KeyError(f"Unknown parameter: {k}")
        new_fixed = {**self.fixed, **kv}
        new_free = tuple(n for n in self.free if n not in kv)
        return ParameterView(space=self.space, fixed=new_fixed, free=new_free)

    def bind(self, **free_values: Scalar) -> ParameterSet:
        # Total parameter assignment = fixed ∪ provided free values
        names = set(self.space.by_name().keys())
        missing = [n for n in self.free if n not in free_values]
        if missing:
            raise ValueError(f"Missing free values for: {missing}")
        total = {**self.fixed, **free_values}
        unknown = [k for k in total if k not in names]
        if unknown:
            raise KeyError(f"Unknown parameter(s): {unknown}")
        return ParameterSet(values=total)
```

> **Fix vs Bind**: _Fix_ chooses which parameters are constant for an analysis
(view-level decision). _Bind_ supplies concrete values to produce a complete set.

### 2.3 TransformedView (optimizer-friendly coordinates)

- For optimization, we map natural params → transformed coordinates (e.g., log, logit).
- This is a **view** over a `ParameterView`; all transforms are separable (per-dimension).

```python
from typing import Callable

class Transform(Protocol):
    def forward(self, x: float) -> float: ...
    def backward(self, y: float) -> float: ...
    # Optional: bounds mapping helpers if needed (not strictly required)

@dataclass(frozen=True)
class Identity:
    def forward(self, x: float) -> float: return x
    def backward(self, y: float) -> float: return y

@dataclass(frozen=True)
class Log10:
    eps: float = 1e-12
    def forward(self, x: float) -> float:
        if x <= 0: raise ValueError("log10 requires x>0")
        import math; return math.log10(x)
    def backward(self, y: float) -> float:
        import math; return 10.0 ** y

@dataclass(frozen=True)
class Logit:
    eps: float = 1e-12
    def forward(self, x: float) -> float:
        import math
        if not (0.0 < x < 1.0): raise ValueError("logit requires 0<x<1")
        x = min(max(x, self.eps), 1.0 - self.eps)
        return math.log(x/(1.0-x))
    def backward(self, y: float) -> float:
        import math
        p = 1.0/(1.0+math.exp(-y))
        # clamp away from 0/1 to avoid downstream issues
        eps = self.eps
        return min(max(p, eps), 1.0 - eps)

@dataclass(frozen=True)
class TransformedView:
    view: ParameterView
    transforms: Dict[str, Transform]

    def to_transformed(self, pset: ParameterSet) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, val in pset.values.items():
            t = self.transforms.get(name, Identity())
            out[name] = t.forward(float(val)) if isinstance(val, (int, float)) else float(val)
        return out

    def from_transformed(self, coords: Mapping[str, float]) -> ParameterSet:
        nat: Dict[str, Scalar] = dict(self.view.fixed)
        for name in self.view.free:
            y = coords[name]
            t = self.transforms.get(name, Identity())
            nat[name] = t.backward(float(y))
        return ParameterSet(values=nat)

    def transformed_bounds(self) -> Dict[str, tuple[float, float]]:
        b: Dict[str, tuple[float, float]] = {}
        specs = self.view.space.by_name()
        for name in self.view.free:
            spec = specs[name]
            t = self.transforms.get(name, Identity())
            # Conservative: map endpoints; assume monotone transforms
            lo, hi = float(spec.lower), float(spec.upper)
            b[name] = (t.forward(lo), t.forward(hi))
        return b
```

Notes:
- We keep transforms **optional and per-parameter** (separable). Future coupled
  transforms can be introduced via a higher-level “coordinate system” without breaking this.
- We’ve chosen to **skip Jacobians now**; they can be added later with an optional
  interface on `Transform` without API churn.

---

## 3) Model Interface & Scenarios (Pure & Usable)

We retain your great usability surface (decorators) while keeping purity. The **core rule**:
`simulate(...)` must be pure in the sense that its **contracted inputs** fully determine outputs.

### 3.1 Decorators

```python
from typing import Callable, Any, Optional, Dict
import polars as pl

def model_output(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Marks a method that converts raw sim output → pl.DataFrame (or handle)."""
    def deco(fn: Callable[..., pl.DataFrame]):
        setattr(fn, "_is_model_output", True)
        setattr(fn, "_output_name", name)
        setattr(fn, "_output_metadata", metadata or {})
        return fn
    return deco

def model_scenario(name: str):
    """Declares a pure scenario patch that returns (config_patch, param_patch)."""
    def deco(fn: Callable[..., "ScenarioSpec"]):
        setattr(fn, "_is_model_scenario", True)
        setattr(fn, "_scenario_name", name)
        return fn
    return deco
```

### 3.2 ScenarioSpec (pure patches)

```python
from dataclasses import dataclass
from typing import Optional, Callable, Mapping, Any

@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    config_patch: Optional[Mapping[str, Any]] = None      # JSON-serializable knobs
    param_patch: Optional[Mapping[str, Scalar]] = None     # natural param edits
    doc: str = ""
```

### 3.3 BaseModel (OO façade, pure core)

```python
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Mapping
import polars as pl

OutputTable = pl.DataFrame  # reserve alias for later extension to handles

class BaseModel(ABC):
    """OO façade; core computation remains pure and provenance-driven."""

    # --- Required pure inputs ---
    space: ParameterSpace   # immutable by design
    base_config: Mapping[str, Any]  # JSON-serializable; immutable (treat as frozen)

    def __init__(self, space: ParameterSpace, base_config: Mapping[str, Any]):
        self.space = space
        self.base_config = dict(base_config)

    # --- Subclass responsibilities ---
    @abstractmethod
    def build_sim(self, params: ParameterSet, seed: int, config: Mapping[str, Any]) -> Any: ...
    @abstractmethod
    def run_sim(self, sim: Any, seed: int) -> Any: ...  # returns raw backend object

    # --- Extractor registry (auto-discovered by decorator) ---
    def get_extractors(self) -> Dict[str, Callable[[Any, int], OutputTable]]:
        out: Dict[str, Callable] = {}
        for attr in dir(self):
            fn = getattr(self, attr)
            if callable(fn) and getattr(fn, "_is_model_output", False):
                out[getattr(fn, "_output_name")] = fn
        return out

    # --- Scenario registry (decorators + programmatic) ---
    _scenarios: Dict[str, ScenarioSpec] = {}

    def register_scenario(self, spec: ScenarioSpec) -> "BaseModel":
        # Safe copy; registry is per-instance
        self._scenarios = {**self._scenarios, spec.name: spec}
        return self

    def scenarios(self) -> Sequence[str]:
        return sorted(self._scenarios.keys())

    # --- Pure scenario application ---
    def _apply_scenario(self, name: Optional[str], pset: ParameterSet, config: Mapping[str, Any]):
        if not name:
            return pset, config, None
        spec = self._scenarios[name]
        # apply param & config patches
        new_vals = dict(pset.values)
        if spec.param_patch: new_vals.update(spec.param_patch)
        new_pset = ParameterSet(values=new_vals)
        new_cfg = dict(config)
        if spec.config_patch: new_cfg.update(spec.config_patch)
        return new_pset, new_cfg

    # --- Pure simulate: compose build/run and extractors ---
    def simulate(self, pset: ParameterSet, seed: int, scenario: Optional[str] = None) -> Dict[str, OutputTable]:
        pset2, cfg2 = self._apply_scenario(scenario, pset, self.base_config)
        sim = self.build_sim(pset2, seed, cfg2)
        raw = self.run_sim(sim, seed)
        outputs: Dict[str, OutputTable] = {}
        for name, fn in self.get_extractors().items():
            df = fn(raw, seed)  # DataFrame or future handle
            outputs[name] = df
        return outputs
```

This BaseModel preserves your decorator UX, allows programmatic scenario registration,
and keeps the simulation path free of hidden mutation. Scenarios are **patches** that
modify the _inputs_ to `build_sim` and optionally apply a pure modifier to the fresh sim.

---

## 4) Provenance & Caching (Hash Tree + Sharded Storage)

### 4.1 What we hash (leaves)

- `ParameterSpace` (names, kinds, bounds, docs)  
- `ParameterSet` (values) **or** the `ParameterView` + free vector in transformed coords
- `BaseModel` code identity (module path + source hash of `build_sim`, `run_sim`, extractors)
- ScenarioSpec (name + patches)
- `base_config` (JSON canonicalized)
- Seed (base + replicate derivations policy string)
- Environment capsule (container image digest / Python pkg lockfile tl;dr hash)

### 4.2 Root hash

- Concatenate leaf hashes as a canonical JSON, BLAKE2b/SHA‑256 → **root hash**.
- Everything downstream (artifacts, diagnostics, Parquet tables) is stored under
  `root/` with **directory sharding** to avoid hot-dir issues:
  `root_prefix/root_mid/root_full/…` (e.g., `ab/cd/abcdef.../`).

**Why sharding?** Many filesystems (and object stores’ directory emulations)
degrade with huge flat directories. Sharding evenly distributes entries so directory
operations stay O(1) and caches (client and server) stay warm across smaller listings.

### 4.3 Artifacts layout (example)

```
ab/cd/abcdef.../
  run_report.json
  meta.json
  outputs/
    prevalence.parquet
    timeseries.parquet
  trials/
    trials.parquet
  logs/
    stdout.txt
    stderr.txt
```

---

## 5) Optimization Results (separate, immutable)

```python
from dataclasses import dataclass
from typing import Mapping, Optional, Literal, Sequence

@dataclass(frozen=True)
class TrialRecord:
    param_id: str
    params: Mapping[str, Scalar]
    loss: Optional[float]
    status: Literal["completed", "failed", "timeout", "pruned"]
    seed: int
    scenario: str
    per_target: Mapping[str, Optional[float]] = None
    diagnostics: Mapping[str, object] = None  # small, JSON-serializable

@dataclass(frozen=True)
class StudyResult:
    study_id: str
    trials: Sequence[TrialRecord]
    meta: Mapping[str, object]

    def best(self) -> TrialRecord:
        return min((t for t in self.trials if t.loss is not None),
                   key=lambda t: t.loss)
```

- Persist **trials** as a Parquet table (denormalized columns: params, loss, status…).
- Persist **meta** as JSON (optimizer kind, bounds, transforms, root hash, versions).
- This pairs naturally with ModelOps `UniqueParameterSet` at the service boundary.

---

## 6) Simulation Service Boundary (Contracts‑friendly)

Provide small adapters so evaluators can run locally or on a cluster:

- **Inbound**: `ParameterSet` ↔️ `UniqueParameterSet` (stable `param_id` via canonical
  JSON).  
- **Submission**: `fn_ref` is `"package.module:function"` pointing to a thin wrapper that
  reconstructs model, pset, seed, scenario, then calls `simulate`.
- **Gather/Aggregate**: accept either a reducer reference (`"package:function"`) or a
  callable; the reducer operates on small tabular `SimReturn`s only.

This boundary remains pure: it is **data in, data out**—no smeared implicit context.

---

## 7) Implementation Plan (Phased)

### Phase 1 — Core types & purity skeleton (1–2 weeks)

- [ ] Implement `Scalar`, `ParameterSpec`, `ParameterSpace`, `ParameterSet`, `ParameterView`.
- [ ] Implement `Transform` + `Identity`, `Log10`, `Logit`, and `TransformedView`.
- [ ] Implement `BaseModel` skeleton with decorator and programmatic registries.
- [ ] Implement `ScenarioSpec` and pure scenario application.
- [ ] Write **unit tests** for: view.fix/bind; transforms (round‑trip), bounds; decorator discovery.

### Phase 2 — Provenance & storage (1–2 weeks)

- [ ] Implement leaf hashers for specs, params, config, code identity, scenarios, seeds, env.
- [ ] Implement root hash computation.  
- [ ] Implement sharded artifact store layout helpers.
- [ ] Add `RunReport` (JSON) with seeds, hashes, env capsule, timings.
- [ ] Tests: hash stability, sharding correctness, idempotent simulate() across processes.

### Phase 3 — Usability & extractors (1–2 weeks)

- [ ] Add `OutputTable` alias (pl.DataFrame for now), leave room for handles.
- [ ] Add `model.describe()`, `view.describe()`, and `preview_output()` helpers.
- [ ] Asof join/aligners remain as-is (or thin wrappers) to keep scope tight.
- [ ] Tests: extractor discovery, schema sanity checks (optional), residual diagnostics.

### Phase 4 — Optimization results & adapters (1 week)

- [ ] Implement `TrialRecord`, `StudyResult`, Parquet/JSON persistence.
- [ ] Implement adapters to/from `UniqueParameterSet`.
- [ ] Add simple random/LHS/Sobol samplers on `ParameterView` (optional MVP).
- [ ] Tests: best-trial extraction, round-trip persistence.

### Phase 5 — Integration & examples (ongoing)

- [ ] Port one toy SIR model using decorators + programmatic scenarios.
- [ ] Example notebook: baseline vs intervention scenario; transformed optimization bounds.
- [ ] CLI sketch (optional): `calabaria simulate --scenario ... --seed ...`

---

## 8) Risks & Mitigations

- **Large outputs**: defer to future `TableHandle` in `OutputTable` to support Arrow/Parquet
  scanning; keep current interface simple.
- **Transform edge-cases**: enforce epsilons and strict bound checks; round‑trip tests.
  enforce via pre-flight lint and provenance hash of patch code object.
- **Hash fragility**: keep hashing components small and canonical (JSON with sorted keys);
  include versions of hashers in `RunReport` to spot changes.

---

## 9) Mini End-to-End Example

```python
# 1) Define parameter space
space = ParameterSpace(specs=(
    ParameterSpec("beta", lower=1e-5, upper=1.0, doc="transmission rate"),
    ParameterSpec("gamma", lower=1e-5, upper=1.0, doc="recovery rate"),
))

# 2) Create a view and fix gamma for a sweep
view = ParameterView.from_space(space).fix(gamma=0.1)

# 3) Optimizer-friendly transform (log10 on positives)
tview = TransformedView(view, transforms={"beta": Log10(), "gamma": Log10()})

# 4) Bind a candidate
pset = tview.from_transformed({"beta": -1.0, "gamma": -1.0})  # 10^-1

# 5) Build a model (subclass implements build_sim/run_sim/extractors)
model = MySIRModel(space=space, base_config={"duration_days": 100})

# 6) Register a scenario programmatically
model.register_scenario(ScenarioSpec(
    name="double_contacts",
    param_patch={"beta": 2.0},  # simplistic illustration
    doc="Double effective contact rate"
))

# 7) Simulate with scenario and seed
outputs = model.simulate(pset, seed=42, scenario="double_contacts")
# outputs: {"timeseries": pl.DataFrame(...), "prevalence": pl.DataFrame(...)}
```

---

## 10) Conclusion

A **pure functional core with a friendly OO façade** buys us reproducibility, clean
provenance, robust caching, and distributed safety—without giving up the usability
scientists expect (decorators, scenarios, extractor discovery). The design here is
minimal but extensible; layers like constraints, units, priors, and streaming outputs
can be added without refactoring the core. The result is a foundation that is both
**scientifically trustworthy** and **engineerable at scale**.
