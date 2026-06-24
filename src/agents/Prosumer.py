from .Agent import Agent
from .Producer import Producer
from .Consumer import Consumer
import numpy as np

class Prosumer(Agent):
    """
    Energy Prosumer

    Attributes
    ----------
        agent_id : int
            Agent's unique identifier
        bus : int
            The index of the bus in the grid to which the agent is connected
        max_power_consumption: float
            Maximum power which the agent can consume
        max_power_production: float
            Maximum power which the agent can produce
        a: float
            Coefficient of quadratic term in prosumer cost/utility function
        b: float
            Coefficient of linear term in prosumer cost/utility function
    """
    
    def __init__(self, agent_id: int, bus_id: int, max_power_consumption: float, max_power_production: float, a: float, b: float):
        super().__init__(agent_id, bus_id, -max_power_consumption, max_power_production)
        self.a = a
        self.b = b

    def solve_local(self, P_i: np.ndarray, pi_row: np.ndarray, lambda_row: np.ndarray, rho: float, tau: float, phi: float):
        """
        Solve the local optimization problem for a prosumer agent in an ADMM-based energy trading setup.

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
        P_opt : numpy.ndarray
            Optimized power vector for agent i with self-trade enforced to zero.
        """

        i = self.agent_id
        P_i_next = np.zeros(len(P_i))

        # Closed form solution
        for j in range(len(P_i)):
            if j == i:
                continue

            P_i_next[j] = (lambda_row[j] - self.b - tau + phi + rho * pi_row[j]) / (2*self.a + rho)

        return P_i_next
