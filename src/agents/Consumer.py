from .Agent import Agent
import numpy as np

class Consumer(Agent):
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

    utility_a : float
        Coefficient of quadratic term in consumer utility function
        
    utility_b : float
        Coefficient of linear term in consumer utility function
    """
    def __init__(self, utility_a: float, utility_b: float, **kwargs):
        p_min = kwargs.get("p_min")
        p_max = kwargs.get("p_max")

        # ensure p bounds ≤ 0
        if p_min > 0 or p_max > 0:
            raise ValueError(
                f"Consumer power limits must be ≤ 0. Got p_min={p_min}, p_max={p_max}"
            )
        
        super().__init__(**kwargs)
        self.a = utility_a
        self.b = utility_b

    def solve_local(self, P_i: np.ndarray, pi_row: np.ndarray, lambda_row: np.ndarray, rho: float, G_i: float):
        i = self.agent_id
        P_i_next = np.zeros(len(P_i))

        # Closed form solution
        for j in range(len(P_i)):
            if j == i:
                continue

            P_i_next[j] = min(0, (lambda_row[j] - self.b - self.tau_p + self.phi_p + rho * pi_row[j] - G_i) / (2*self.a + rho))

        return P_i_next