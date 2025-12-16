"""Complete workflow demonstration of the fluent builder API.

This example showcases the Grammar of Model Parameters implementation,
demonstrating how to:
- Build simulators with the fluent API
- Use different coordinate systems (transforms)
- Work with multiple scenarios
- Integrate with calibration

The fluent API provides an expressive, immutable builder pattern for
creating ModelSimulator instances.
"""

import numpy as np
import polars as pl

from modelops_calabaria import (
    BaseModel,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ConfigSpec,
    ConfigurationSpace,
    ConfigurationSet,
    model_output,
    model_scenario,
    ScenarioSpec,
    LogTransform,
    AffineSqueezedLogit,
)


# ==============================================================================
# Example Model: Simple SIR Epidemic Model
# ==============================================================================

class SimpleSIR(BaseModel):
    """Simple SIR epidemiological model for demonstration.

    Parameters:
    - beta: transmission rate (contact rate × transmission probability)
    - gamma: recovery rate (1/infectious period)
    - population: total population size
    - initial_infected: number initially infected

    Configuration:
    - dt: time step size
    - duration: simulation duration in days
    """

    @classmethod
    def default_space(cls):
        """Define parameter space."""
        return ParameterSpace([
            ParameterSpec("beta", 0.1, 3.0, "float", doc="Transmission rate"),
            ParameterSpec("gamma", 0.05, 1.0, "float", doc="Recovery rate"),
            ParameterSpec("population", 1000, 1000000, "int", doc="Population size"),
            ParameterSpec("initial_infected", 1, 100, "int", doc="Initial infections"),
        ])

    @classmethod
    def default_config_space(cls):
        """Define configuration space."""
        return ConfigurationSpace([
            ConfigSpec("dt", default=0.1, doc="Time step"),
            ConfigSpec("duration", default=100.0, doc="Simulation days"),
        ])

    def __init__(self):
        """Initialize with default spaces."""
        space = self.default_space()
        config_space = self.default_config_space()
        base_config = ConfigurationSet(config_space, {
            "dt": 0.1,
            "duration": 100.0,
        })
        super().__init__(space, config_space, base_config)

    def build_sim(self, params: ParameterSet, config: ConfigurationSet) -> dict:
        """Build simulation state."""
        return {
            "beta": params["beta"],
            "gamma": params["gamma"],
            "N": params["population"],
            "I0": params["initial_infected"],
            "dt": config["dt"],
            "duration": config["duration"],
        }

    def run_sim(self, state: dict, seed: int) -> dict:
        """Run SIR simulation."""
        rng = np.random.RandomState(seed)

        # Initial conditions
        N = state["N"]
        I = state["I0"]
        S = N - I
        R = 0

        # Time parameters
        dt = state["dt"]
        duration = state["duration"]
        n_steps = int(duration / dt)

        # Store trajectory
        times = []
        S_vals = []
        I_vals = []
        R_vals = []

        for step in range(n_steps):
            t = step * dt

            # Record state
            times.append(t)
            S_vals.append(S)
            I_vals.append(I)
            R_vals.append(R)

            # Compute rates
            infection_rate = state["beta"] * S * I / N
            recovery_rate = state["gamma"] * I

            # Stochastic transitions
            new_infections = rng.poisson(infection_rate * dt)
            new_recoveries = rng.poisson(recovery_rate * dt)

            # Update compartments
            S = max(0, S - new_infections)
            I = max(0, I + new_infections - new_recoveries)
            R = min(N, R + new_recoveries)

        return {
            "times": np.array(times),
            "S": np.array(S_vals),
            "I": np.array(I_vals),
            "R": np.array(R_vals),
        }

    @model_output("timeseries")
    def extract_timeseries(self, raw: dict, seed: int) -> pl.DataFrame:
        """Extract time series output."""
        return pl.DataFrame({
            "time": raw["times"],
            "S": raw["S"],
            "I": raw["I"],
            "R": raw["R"],
        })

    @model_output("peak_infected")
    def extract_peak(self, raw: dict, seed: int) -> pl.DataFrame:
        """Extract peak infection statistics."""
        peak_idx = np.argmax(raw["I"])
        return pl.DataFrame({
            "peak_time": [raw["times"][peak_idx]],
            "peak_value": [raw["I"][peak_idx]],
        })

    @model_scenario("high_transmission")
    def high_transmission_scenario(self) -> ScenarioSpec:
        """Scenario with increased transmission."""
        return ScenarioSpec(
            name="high_transmission",
            param_patches={"beta": 2.5},
            doc="High transmission rate scenario"
        )

    @model_scenario("long_duration")
    def long_duration_scenario(self) -> ScenarioSpec:
        """Scenario with extended simulation time."""
        return ScenarioSpec(
            name="long_duration",
            config_patches={"duration": 200.0},
            doc="Extended simulation duration"
        )


# ==============================================================================
# Example 1: Basic Fluent API Usage
# ==============================================================================

def example_1_basic_usage():
    """Demonstrate basic fluent API usage."""
    print("="*70)
    print("Example 1: Basic Fluent API Usage")
    print("="*70)

    model = SimpleSIR()

    # Build simulator with fluent API
    sim = (model
           .as_sim("baseline")
           .fix(population=10000, initial_infected=10)
           .build())

    print(f"Simulator dimension: {sim.dim}")
    print(f"Free parameters: {sim.free_param_names}")
    print(f"Bounds:\n{sim.bounds()}")

    # Execute simulation
    z = np.array([0.5, 0.2])  # beta=0.5, gamma=0.2
    outputs = sim(z, seed=42)

    print(f"\nOutput keys: {list(outputs.keys())}")
    print(f"Timeseries shape: {outputs['timeseries'].shape}")
    print(f"Peak infected: {outputs['peak_infected']}")
    print()


# ==============================================================================
# Example 2: Using Transforms
# ==============================================================================

def example_2_transforms():
    """Demonstrate coordinate transforms."""
    print("="*70)
    print("Example 2: Coordinate Transforms")
    print("="*70)

    model = SimpleSIR()

    # Build simulator with log transforms
    sim = (model
           .as_sim("baseline")
           .fix(population=10000, initial_infected=10)
           .with_transforms(beta="log", gamma="log")
           .build())

    print(f"Free parameters: {sim.free_param_names}")
    print(f"Bounds in transformed space:\n{sim.bounds()}")

    # z is now in log space
    z = np.array([np.log(0.5), np.log(0.2)])  # log(beta), log(gamma)
    outputs = sim(z, seed=42)

    print(f"\nPeak infected with beta=0.5, gamma=0.2:")
    print(outputs['peak_infected'])
    print()


# ==============================================================================
# Example 3: Multiple Scenarios
# ==============================================================================

def example_3_scenarios():
    """Demonstrate multiple scenarios."""
    print("="*70)
    print("Example 3: Multiple Scenarios")
    print("="*70)

    model = SimpleSIR()

    # Create simulators for different scenarios
    sim_baseline = (model
                    .as_sim("baseline")
                    .fix(population=10000, initial_infected=10)
                    .build())

    sim_high_trans = (model
                      .as_sim("high_transmission")
                      .fix(population=10000, initial_infected=10)
                      .build())

    z = np.array([0.5, 0.2])  # Same z for both

    outputs_baseline = sim_baseline(z, seed=42)
    outputs_high = sim_high_trans(z, seed=42)

    print("Baseline scenario:")
    print(f"  Peak: {outputs_baseline['peak_infected']}")

    print("\nHigh transmission scenario:")
    print(f"  Peak: {outputs_high['peak_infected']}")
    print()


# ==============================================================================
# Example 4: Reusable Builders
# ==============================================================================

def example_4_reusable_builders():
    """Demonstrate reusable builder pattern."""
    print("="*70)
    print("Example 4: Reusable Builders")
    print("="*70)

    model = SimpleSIR()

    # Create base builder
    base = (model
            .as_sim("baseline")
            .fix(population=10000, initial_infected=10))

    # Create different simulators from same base
    sim1 = base.fix(gamma=0.2).build()  # Fix gamma, free beta
    sim2 = base.fix(beta=0.8).build()   # Fix beta, free gamma

    print("Simulator 1 (gamma fixed):")
    print(f"  Free params: {sim1.free_param_names}")
    print(f"  Dimension: {sim1.dim}")

    print("\nSimulator 2 (beta fixed):")
    print(f"  Free params: {sim2.free_param_names}")
    print(f"  Dimension: {sim2.dim}")
    print()


# ==============================================================================
# Example 5: Transform Instances
# ==============================================================================

def example_5_custom_transforms():
    """Demonstrate custom transform instances."""
    print("="*70)
    print("Example 5: Custom Transform Instances")
    print("="*70)

    model = SimpleSIR()

    # Use custom transform instances for fine control
    sim = (model
           .as_sim("baseline")
           .fix(population=10000, initial_infected=10)
           .with_transforms(
               beta=LogTransform(),
               gamma=AffineSqueezedLogit(eps=1e-5)  # Custom epsilon
           )
           .build())

    print(f"Free parameters: {sim.free_param_names}")
    print(f"Custom transforms applied")
    print(f"Bounds:\n{sim.bounds()}")
    print()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    example_1_basic_usage()
    example_2_transforms()
    example_3_scenarios()
    example_4_reusable_builders()
    example_5_custom_transforms()

    print("="*70)
    print("All examples completed successfully!")
    print("="*70)
