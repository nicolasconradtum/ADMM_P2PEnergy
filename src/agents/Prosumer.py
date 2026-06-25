from .Agent import Agent
from .Producer import Producer
from .Consumer import Consumer
import numpy as np

class Prosumer(Agent):
    """
    Consumer class representing a market consumer in the
    peer-to-peer energy trading framework.

    Attributes
    ----------
    agent_id : int
        Unique identifier of the agent.

    p_min : float
        Minimum active power limit in kW (must be ≤ 0)

    p_max : float
        Maximum active power limit in kW (must be ≤ 0)

    dual_step_size : float
        Step size used for updating agent's local dual variables during optimization.

    power_factor : float
        Agent power factor used to compute reactive power injections or
        withdrawals from active power values.

    a : float
        Coefficient of quadratic term in prosumer objective function
        
    b : float
        Coefficient of linear term in prosumer objective function
    """
    def __init__(self, a: float, b: float, **kwargs):
        p_min = kwargs.get("p_min")
        p_max = kwargs.get("p_max")

        # ensure p_max ≥ p_min
        if p_max < p_min:
            raise ValueError(
                f"Invalid power limits: p_min ({p_min}) must be ≤ p_max ({p_max})"
            )
        
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def solve_local(self, P_i: np.ndarray, pi_row: np.ndarray, lambda_row: np.ndarray, rho: float, G_i: float):
        i = self.agent_id
        P_i_next = np.zeros(len(P_i))

        # Closed form solution
        for j in range(len(P_i)):
            if j == i:
                continue

            P_i_next[j] = (lambda_row[j] - self.b - self.tau_p + self.phi_p + rho * pi_row[j] - G_i) / (2*self.a + rho)

        return P_i_next