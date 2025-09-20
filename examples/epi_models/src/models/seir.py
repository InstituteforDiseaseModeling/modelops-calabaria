"""Stochastic SEIR epidemiological model.

This is a canonical implementation of a Susceptible-Exposed-Infected-Recovered
compartmental model with stochastic transitions between states.

The model simulates disease spread through a population using differential
equations with Poisson-distributed transitions for stochasticity.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Mapping

from modelops_calabaria import (
    BaseModel, ParameterSpace, ParameterSpec, ParameterSet,
    model_output, model_scenario, ScenarioSpec
)


class StochasticSEIR(BaseModel):
    """Stochastic SEIR compartmental epidemiological model.

    This model simulates the spread of an infectious disease through a population
    divided into four compartments:
    - S: Susceptible individuals
    - E: Exposed individuals (infected but not yet infectious)
    - I: Infected individuals (infectious)
    - R: Recovered individuals (immune)

    The model uses stochastic transitions between compartments based on
    Poisson processes for realistic variability.
    """

    @classmethod
    def parameter_space(cls):
        """Get the parameter space for this model."""
        return ParameterSpace([
            ParameterSpec(
                "beta", 0.1, 2.0, "float",
                doc="Transmission rate (contacts per day × probability of transmission)"
            ),
            ParameterSpec(
                "sigma", 0.05, 0.5, "float",
                doc="Incubation rate (1/latent period in days)"
            ),
            ParameterSpec(
                "gamma", 0.05, 0.5, "float",
                doc="Recovery rate (1/infectious period in days)"
            ),
            ParameterSpec(
                "population", 1000, 1000000, "int",
                doc="Total population size"
            ),
            ParameterSpec(
                "initial_infected", 1, 100, "int",
                doc="Initial number of infected individuals"
            ),
            ParameterSpec(
                "initial_exposed", 0, 100, "int",
                doc="Initial number of exposed individuals"
            ),
            ParameterSpec(
                "simulation_days", 100, 365, "int",
                doc="Number of days to simulate"
            ),
        ])

    def __init__(self, space=None):
        """Initialize the SEIR model with parameter space."""
        if space is None:
            space = self.parameter_space()
        super().__init__(space, base_config={
            "dt": 0.1,  # Time step for simulation
            "output_frequency": 1.0,  # Days between output points
        })

    def build_sim(self, params: ParameterSet, config: Mapping[str, Any]) -> Dict:
        """Build the simulation state from parameters and configuration.

        Args:
            params: Parameter values for this simulation
            config: Configuration dictionary (scenarios can modify this)

        Returns:
            Dictionary containing simulation state
        """
        # Extract parameters
        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        N = int(params["population"])
        I0 = int(params["initial_infected"])
        E0 = int(params["initial_exposed"])
        days = int(params["simulation_days"])

        # Initial conditions
        S0 = N - E0 - I0
        R0 = 0

        # Time parameters
        dt = float(config.get("dt", 0.1))
        output_freq = float(config.get("output_frequency", 1.0))

        # Simulation parameters that might be modified by scenarios
        beta_eff = beta * config.get("transmission_multiplier", 1.0)

        return {
            "initial_state": {"S": S0, "E": E0, "I": I0, "R": R0},
            "params": {
                "beta": beta_eff,
                "sigma": sigma,
                "gamma": gamma,
                "N": N
            },
            "time_params": {
                "dt": dt,
                "total_days": days,
                "output_frequency": output_freq,
            }
        }

    def run_sim(self, state: Dict, seed: int) -> Dict:
        """Run the stochastic SEIR simulation.

        Args:
            state: Simulation state from build_sim
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing simulation results
        """
        rng = np.random.RandomState(seed)

        # Extract state
        initial = state["initial_state"]
        params = state["params"]
        time_params = state["time_params"]

        # Current state
        S, E, I, R = initial["S"], initial["E"], initial["I"], initial["R"]
        N = params["N"]

        # Parameters
        beta, sigma, gamma = params["beta"], params["sigma"], params["gamma"]
        dt = time_params["dt"]
        total_days = time_params["total_days"]
        output_freq = time_params["output_frequency"]

        # Time points
        times = np.arange(0, total_days + dt, dt)
        output_times = np.arange(0, total_days + output_freq, output_freq)

        # Storage for results
        results = {
            "times": [],
            "S": [], "E": [], "I": [], "R": [],
            "new_infections": [], "prevalence": []
        }

        # Initial output
        if 0 in output_times:
            results["times"].append(0.0)
            results["S"].append(S)
            results["E"].append(E)
            results["I"].append(I)
            results["R"].append(R)
            results["new_infections"].append(0.0)
            results["prevalence"].append(I)

        next_output = output_freq
        cumulative_new_infections = 0.0

        # Run simulation
        for t in times[1:]:
            # Calculate rates
            infection_rate = beta * S * I / N
            progression_rate = sigma * E
            recovery_rate = gamma * I

            # Stochastic transitions (Poisson processes)
            new_infections = rng.poisson(infection_rate * dt)
            new_progressions = rng.poisson(progression_rate * dt)
            new_recoveries = rng.poisson(recovery_rate * dt)

            # Ensure we don't exceed compartment sizes
            new_infections = min(new_infections, S)
            new_progressions = min(new_progressions, E)
            new_recoveries = min(new_recoveries, I)

            # Update compartments
            S -= new_infections
            E += new_infections - new_progressions
            I += new_progressions - new_recoveries
            R += new_recoveries

            cumulative_new_infections += new_infections

            # Store output if at output time
            if t >= next_output:
                results["times"].append(float(t))
                results["S"].append(S)
                results["E"].append(E)
                results["I"].append(I)
                results["R"].append(R)
                results["new_infections"].append(cumulative_new_infections / output_freq)
                results["prevalence"].append(I)

                next_output += output_freq
                cumulative_new_infections = 0.0

        # Calculate some derived statistics
        peak_infections = max(results["I"])
        total_attack_rate = (N - results["S"][-1]) / N

        results["summary"] = {
            "peak_infections": peak_infections,
            "total_attack_rate": total_attack_rate,
            "final_size": results["R"][-1]
        }

        return results

    @model_output("prevalence")
    def extract_prevalence(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract daily prevalence (number of infected individuals).

        This is the primary output showing the epidemic curve.
        """
        return pl.DataFrame({
            "day": raw["times"],
            "infections": raw["prevalence"]
        })

    @model_output("incidence")
    def extract_incidence(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract daily incidence (new infections per day)."""
        return pl.DataFrame({
            "day": raw["times"],
            "new_infections": raw["new_infections"]
        })

    @model_output("compartments")
    def extract_compartments(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract full SEIR compartment time series."""
        return pl.DataFrame({
            "day": raw["times"],
            "susceptible": raw["S"],
            "exposed": raw["E"],
            "infected": raw["I"],
            "recovered": raw["R"]
        })

    @model_output("summary")
    def extract_summary(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract epidemic summary statistics."""
        summary = raw["summary"]
        return pl.DataFrame({
            "metric": ["peak_infections", "attack_rate", "final_size"],
            "value": [
                float(summary["peak_infections"]),
                float(summary["total_attack_rate"]),
                float(summary["final_size"])
            ]
        })

    @model_scenario("baseline")
    def baseline_scenario(self) -> ScenarioSpec:
        """Baseline scenario with no interventions."""
        return ScenarioSpec(
            "baseline",
            doc="Baseline scenario with natural disease transmission"
        )

    @model_scenario("lockdown")
    def lockdown_scenario(self) -> ScenarioSpec:
        """Lockdown scenario with reduced transmission."""
        return ScenarioSpec(
            "lockdown",
            doc="Lockdown scenario reducing transmission by 50%",
            config_patches={"transmission_multiplier": 0.5}
        )

    @model_scenario("high_transmission")
    def high_transmission_scenario(self) -> ScenarioSpec:
        """High transmission scenario."""
        return ScenarioSpec(
            "high_transmission",
            doc="High transmission scenario increasing transmission by 50%",
            config_patches={"transmission_multiplier": 1.5}
        )