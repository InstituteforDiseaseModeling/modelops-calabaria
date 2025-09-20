"""Age-stratified SEIR epidemiological model.

This extends the basic SEIR model to include age structure, allowing for
different transmission rates and disease outcomes by age group.

The model includes age-specific contact patterns and can simulate
age-targeted interventions like school closures or protecting elderly.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Mapping, List

from modelops_calabaria import (
    BaseModel, ParameterSpace, ParameterSpec, ParameterSet,
    model_output, model_scenario, ScenarioSpec
)


class AgeStratifiedSEIR(BaseModel):
    """Age-stratified SEIR compartmental epidemiological model.

    This model extends the basic SEIR framework to include multiple age groups
    with age-specific contact patterns and susceptibilities.

    Age groups:
    0: Children (0-17)
    1: Adults (18-64)
    2: Elderly (65+)
    """

    @classmethod
    def parameter_space(cls):
        """Get the parameter space for this model."""
        return ParameterSpace([
            ParameterSpec(
                "beta", 0.1, 2.0, "float",
                doc="Base transmission rate"
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
                "population", 10000, 1000000, "int",
                doc="Total population size"
            ),
            ParameterSpec(
                "contact_matrix_scale", 0.1, 2.0, "float",
                doc="Scale factor for contact matrix (controls mixing between age groups)"
            ),
            ParameterSpec(
                "child_susceptibility", 0.3, 1.0, "float",
                doc="Relative susceptibility of children compared to adults"
            ),
            ParameterSpec(
                "elderly_susceptibility", 0.8, 1.5, "float",
                doc="Relative susceptibility of elderly compared to adults"
            ),
            ParameterSpec(
                "initial_infected_children", 0, 50, "int",
                doc="Initial infections in children"
            ),
            ParameterSpec(
                "initial_infected_adults", 1, 100, "int",
                doc="Initial infections in adults"
            ),
            ParameterSpec(
                "initial_infected_elderly", 0, 20, "int",
                doc="Initial infections in elderly"
            ),
            ParameterSpec(
                "simulation_days", 100, 365, "int",
                doc="Number of days to simulate"
            ),
        ])

    def __init__(self, space=None):
        """Initialize the age-stratified SEIR model with parameter space."""
        if space is None:
            space = self.parameter_space()
        super().__init__(space, base_config={
            "dt": 0.1,
            "output_frequency": 1.0,
            # Age structure (children, adults, elderly)
            "age_distribution": [0.25, 0.60, 0.15],
            # Contact matrix (who contacts whom)
            # Rows = infector age group, columns = infectee age group
            "base_contact_matrix": [
                [8.0, 2.0, 0.5],  # Children contact: children, adults, elderly
                [2.0, 6.0, 1.0],  # Adults contact: children, adults, elderly
                [0.5, 1.0, 3.0],  # Elderly contact: children, adults, elderly
            ],
            # Intervention modifiers
            "contact_modifiers": {
                "child_child": 1.0,
                "child_adult": 1.0,
                "child_elderly": 1.0,
                "adult_adult": 1.0,
                "adult_elderly": 1.0,
                "elderly_elderly": 1.0,
            }
        })

    def build_sim(self, params: ParameterSet, config: Mapping[str, Any]) -> Dict:
        """Build the simulation state from parameters and configuration."""
        # Extract parameters
        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        N_total = int(params["population"])
        contact_scale = float(params["contact_matrix_scale"])

        # Age-specific susceptibilities
        susceptibilities = [
            float(params["child_susceptibility"]),
            1.0,  # Adults (reference)
            float(params["elderly_susceptibility"])
        ]

        # Initial infections by age
        I0 = [
            int(params["initial_infected_children"]),
            int(params["initial_infected_adults"]),
            int(params["initial_infected_elderly"])
        ]

        days = int(params["simulation_days"])

        # Age structure
        age_dist = config["age_distribution"]
        age_populations = [int(N_total * prop) for prop in age_dist]

        # Adjust for rounding
        age_populations[-1] += N_total - sum(age_populations)

        # Initial conditions by age group
        S0 = [N_age - I_age for N_age, I_age in zip(age_populations, I0)]
        E0 = [0, 0, 0]  # No initial exposed
        R0 = [0, 0, 0]  # No initial recovered

        # Contact matrix with scaling and interventions
        base_matrix = np.array(config["base_contact_matrix"])
        modifiers = config["contact_modifiers"]

        # Apply contact modifiers
        contact_matrix = base_matrix.copy()
        contact_matrix[0, 0] *= modifiers["child_child"]      # Child-child
        contact_matrix[0, 1] *= modifiers["child_adult"]     # Child-adult
        contact_matrix[1, 0] *= modifiers["child_adult"]     # Adult-child (symmetric)
        contact_matrix[0, 2] *= modifiers["child_elderly"]   # Child-elderly
        contact_matrix[2, 0] *= modifiers["child_elderly"]   # Elderly-child (symmetric)
        contact_matrix[1, 1] *= modifiers["adult_adult"]     # Adult-adult
        contact_matrix[1, 2] *= modifiers["adult_elderly"]   # Adult-elderly
        contact_matrix[2, 1] *= modifiers["adult_elderly"]   # Elderly-adult (symmetric)
        contact_matrix[2, 2] *= modifiers["elderly_elderly"] # Elderly-elderly

        # Scale the entire matrix
        contact_matrix *= contact_scale

        # Time parameters
        dt = float(config.get("dt", 0.1))
        output_freq = float(config.get("output_frequency", 1.0))

        return {
            "initial_state": {
                "S": S0, "E": E0, "I": I0, "R": R0,
                "N_by_age": age_populations
            },
            "params": {
                "beta": beta,
                "sigma": sigma,
                "gamma": gamma,
                "susceptibilities": susceptibilities,
                "contact_matrix": contact_matrix.tolist()
            },
            "time_params": {
                "dt": dt,
                "total_days": days,
                "output_frequency": output_freq,
            }
        }

    def run_sim(self, state: Dict, seed: int) -> Dict:
        """Run the age-stratified stochastic SEIR simulation."""
        rng = np.random.RandomState(seed)

        # Extract state
        initial = state["initial_state"]
        params = state["params"]
        time_params = state["time_params"]

        # Current state by age group
        S = np.array(initial["S"], dtype=float)
        E = np.array(initial["E"], dtype=float)
        I = np.array(initial["I"], dtype=float)
        R = np.array(initial["R"], dtype=float)
        N_by_age = np.array(initial["N_by_age"])

        # Parameters
        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        susceptibilities = np.array(params["susceptibilities"])
        contact_matrix = np.array(params["contact_matrix"])

        dt = time_params["dt"]
        total_days = time_params["total_days"]
        output_freq = time_params["output_frequency"]

        # Time arrays
        times = np.arange(0, total_days + dt, dt)
        output_times = np.arange(0, total_days + output_freq, output_freq)

        # Storage
        n_age_groups = len(S)
        results = {
            "times": [],
            "S_by_age": [[] for _ in range(n_age_groups)],
            "E_by_age": [[] for _ in range(n_age_groups)],
            "I_by_age": [[] for _ in range(n_age_groups)],
            "R_by_age": [[] for _ in range(n_age_groups)],
            "new_infections_by_age": [[] for _ in range(n_age_groups)],
            "total_infected": [],
            "total_new_infections": []
        }

        # Initial output
        if 0 in output_times:
            results["times"].append(0.0)
            for a in range(n_age_groups):
                results["S_by_age"][a].append(S[a])
                results["E_by_age"][a].append(E[a])
                results["I_by_age"][a].append(I[a])
                results["R_by_age"][a].append(R[a])
                results["new_infections_by_age"][a].append(0.0)
            results["total_infected"].append(I.sum())
            results["total_new_infections"].append(0.0)

        next_output = output_freq
        cumulative_new_infections = np.zeros(n_age_groups)

        # Run simulation
        for t in times[1:]:
            # Calculate force of infection for each age group
            force_of_infection = np.zeros(n_age_groups)
            for a in range(n_age_groups):
                for b in range(n_age_groups):
                    force_of_infection[a] += (
                        beta * susceptibilities[a] * contact_matrix[b, a] * I[b] / N_by_age[b]
                    )

            # Calculate transition rates
            infection_rates = force_of_infection * S
            progression_rates = sigma * E
            recovery_rates = gamma * I

            # Stochastic transitions
            new_infections = np.zeros(n_age_groups)
            new_progressions = np.zeros(n_age_groups)
            new_recoveries = np.zeros(n_age_groups)

            for a in range(n_age_groups):
                new_infections[a] = rng.poisson(max(0, infection_rates[a] * dt))
                new_progressions[a] = rng.poisson(max(0, progression_rates[a] * dt))
                new_recoveries[a] = rng.poisson(max(0, recovery_rates[a] * dt))

                # Bound transitions by compartment sizes
                new_infections[a] = min(new_infections[a], S[a])
                new_progressions[a] = min(new_progressions[a], E[a])
                new_recoveries[a] = min(new_recoveries[a], I[a])

            # Update compartments
            S -= new_infections
            E += new_infections - new_progressions
            I += new_progressions - new_recoveries
            R += new_recoveries

            cumulative_new_infections += new_infections

            # Store output if at output time
            if t >= next_output:
                results["times"].append(float(t))
                for a in range(n_age_groups):
                    results["S_by_age"][a].append(S[a])
                    results["E_by_age"][a].append(E[a])
                    results["I_by_age"][a].append(I[a])
                    results["R_by_age"][a].append(R[a])
                    results["new_infections_by_age"][a].append(
                        cumulative_new_infections[a] / output_freq
                    )

                results["total_infected"].append(I.sum())
                results["total_new_infections"].append(cumulative_new_infections.sum() / output_freq)

                next_output += output_freq
                cumulative_new_infections.fill(0.0)

        # Summary statistics
        results["summary"] = {
            "peak_infections_by_age": [max(results["I_by_age"][a]) for a in range(n_age_groups)],
            "attack_rate_by_age": [
                (N_by_age[a] - results["S_by_age"][a][-1]) / N_by_age[a]
                for a in range(n_age_groups)
            ],
            "final_size_by_age": [results["R_by_age"][a][-1] for a in range(n_age_groups)]
        }

        return results

    @model_output("prevalence_by_age")
    def extract_prevalence_by_age(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract prevalence (infections) by age group."""
        data = []
        age_groups = ["children", "adults", "elderly"]

        for i, age_group in enumerate(age_groups):
            for day, infections in zip(raw["times"], raw["I_by_age"][i]):
                data.append({
                    "day": day,
                    "age_group": age_group,
                    "infections": infections
                })

        return pl.DataFrame(data)

    @model_output("total_infections")
    def extract_total_infections(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract total infections across all age groups."""
        return pl.DataFrame({
            "day": raw["times"],
            "total_infections": raw["total_infected"]
        })

    @model_output("incidence_by_age")
    def extract_incidence_by_age(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract new infections by age group."""
        data = []
        age_groups = ["children", "adults", "elderly"]

        for i, age_group in enumerate(age_groups):
            for day, new_inf in zip(raw["times"], raw["new_infections_by_age"][i]):
                data.append({
                    "day": day,
                    "age_group": age_group,
                    "new_infections": new_inf
                })

        return pl.DataFrame(data)

    @model_output("summary_by_age")
    def extract_summary_by_age(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract epidemic summary statistics by age group."""
        summary = raw["summary"]
        age_groups = ["children", "adults", "elderly"]

        data = []
        for i, age_group in enumerate(age_groups):
            data.extend([
                {
                    "age_group": age_group,
                    "metric": "peak_infections",
                    "value": float(summary["peak_infections_by_age"][i])
                },
                {
                    "age_group": age_group,
                    "metric": "attack_rate",
                    "value": float(summary["attack_rate_by_age"][i])
                },
                {
                    "age_group": age_group,
                    "metric": "final_size",
                    "value": float(summary["final_size_by_age"][i])
                }
            ])

        return pl.DataFrame(data)

    @model_scenario("baseline")
    def baseline_scenario(self) -> ScenarioSpec:
        """Baseline scenario with normal contact patterns."""
        return ScenarioSpec(
            "baseline",
            doc="Baseline scenario with normal age-structured contact patterns"
        )

    @model_scenario("school_closure")
    def school_closure_scenario(self) -> ScenarioSpec:
        """School closure scenario reducing child-child contacts."""
        return ScenarioSpec(
            "school_closure",
            doc="School closure scenario reducing child-child contacts by 80%",
            config_patches={
                "contact_modifiers": {
                    "child_child": 0.2,      # Greatly reduced child-child contact
                    "child_adult": 0.8,      # Slightly reduced child-adult contact
                    "child_elderly": 1.0,
                    "adult_adult": 1.0,
                    "adult_elderly": 1.0,
                    "elderly_elderly": 1.0,
                }
            }
        )

    @model_scenario("protect_elderly")
    def protect_elderly_scenario(self) -> ScenarioSpec:
        """Protect elderly scenario reducing their contacts."""
        return ScenarioSpec(
            "protect_elderly",
            doc="Protect elderly scenario reducing contacts with elderly by 70%",
            config_patches={
                "contact_modifiers": {
                    "child_child": 1.0,
                    "child_adult": 1.0,
                    "child_elderly": 0.3,    # Greatly reduced child-elderly contact
                    "adult_adult": 1.0,
                    "adult_elderly": 0.3,    # Greatly reduced adult-elderly contact
                    "elderly_elderly": 0.3,  # Reduced elderly-elderly contact
                }
            }
        )