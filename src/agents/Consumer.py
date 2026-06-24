from .Agent import Agent
import numpy as np

class Consumer(Agent):
    """
    Energy Consumer

    Attributes
    ----------
        agent_id : int
            Agent's unique identifier
        bus_id : int
            The index of the bus in the grid to which the agent is connected
        min_power_consumption: float
            Minimum power which the agent can consume
        max_power_consumption: float
            Maximum power which the agent can consume
        utility_a: float
            Coefficient of quadratic term in consumer utility function
        utility_b: float
            Coefficient of linear term in consumer utility function
    """

    def __init__(self, agent_id: int, bus_id: int, min_power_consumption: float, max_power_consumption: float, utility_a: float, utility_b: float):
        super().__init__(agent_id, bus_id, -max_power_consumption, -min_power_consumption)

        self.a = utility_a
        self.b = utility_b

    def solve_local(self, P_i: np.ndarray, pi_row: np.ndarray, lambda_row: np.ndarray, rho: float, tau: float, phi: float):
        """
        Solve the local optimization problem for a consumer agent in an ADMM-based energy trading setup.

        Parameters
        ----------
        P_i : ndarray
            Current power vector for agent i (including self entry).
        pi_row : ndarray
            Consensus variables π_i, pi_row[j] = π_{i, j}
        lambda_row : ndarray
            Dual variables λ_i for ADMM coupling terms. lambda_row[j] = λ_{i,j}
        rho : float
            ADMM penalty parameter.
        tau : float
            Dual variable for upper power constraint
        phi : float
            Dual variable for lower power constraint

        Returns
        -------
        P_opt : ndarray
            Optimized power vector for agent i with self-trade enforced to zero.
        """

        i = self.agent_id
        P_i_next = np.zeros(len(P_i))

        # Closed form solution
        for j in range(len(P_i)):
            if j == i:
                continue

            P_i_next[j] = min(0, (-self.b + lambda_row[j] + rho * pi_row[j] - tau + phi) / (2*self.a + rho))

        return P_i_next
