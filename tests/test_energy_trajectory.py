"""
Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import numpy as np
import pysa.simulation as simulation
import pysa.ising as ising


def test_record_trajectory_parallel():
    couplings = np.array([[0, 1], [1, 0]], dtype="float")
    local_fields = np.zeros(2, dtype="float")
    states = 2 * np.random.randint(2, size=(1, 2)).astype("float") - 1
    energies = np.array([
        ising.get_energy(couplings, local_fields, state) for state in states
    ])
    betas = np.array([1.0], dtype="float")
    beta_idx = np.arange(1)

    (out_states, out_energies, _, _, traj), _ = simulation.simulation_parallel(
        ising.update_spin,
        simulation.random_sweep,
        couplings,
        local_fields,
        states,
        energies,
        beta_idx,
        betas,
        5,
        record_trajectory=True,
    )

    assert traj.shape == (5, 1)
    assert np.isclose(traj[-1, 0], out_energies[0])


def test_record_trajectory_sequential():
    couplings = np.array([[0, 1], [1, 0]], dtype="float")
    local_fields = np.zeros(2, dtype="float")
    states = 2 * np.random.randint(2, size=(1, 2)).astype("float") - 1
    energies = np.array([
        ising.get_energy(couplings, local_fields, state) for state in states
    ])
    betas = np.array([1.0], dtype="float")
    beta_idx = np.arange(1)

    (out_states, out_energies, _, _, traj), _ = simulation.simulation_sequential(
        ising.update_spin,
        simulation.random_sweep,
        couplings,
        local_fields,
        states,
        energies,
        beta_idx,
        betas,
        5,
        record_trajectory=True,
    )

    assert traj.shape == (5, 1)
    assert np.isclose(traj[-1, 0], out_energies[0])
