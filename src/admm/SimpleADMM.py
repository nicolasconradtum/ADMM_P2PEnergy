import numpy as np
from .ADMM import ADMM

class SimpleADMM(ADMM):
    # local primal update
    def update_primal(self):
        for agent in self.agents:
            i = agent.agent_id
            self.P_mat[i, :] = agent.solve_local(
                P_i=self.P_mat[i, :],
                pi_row=self.pi_mat[i, :],
                lambda_row=self.lambda_mat[i, :],
                rho=self.rho,
                tau=self.tau[i],
                phi=self.phi[i]
            )

    # consensus update
    def update_consensus(self):
        pi_new = np.zeros_like(self.P_mat)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                pi_ij = (self.rho * (self.P_mat[i, j] - self.P_mat[j, i]) - (self.lambda_mat[i, j] - self.lambda_mat[j, i])) / (2 * self.rho)

                pi_new[i, j] = pi_ij
                pi_new[j, i] = -pi_ij

        self.pi_mat = pi_new

    # admm dual update
    def update_dual_lambda(self):
        self.lambda_mat += self.rho * (self.pi_mat - self.P_mat)

    # local agent dual update - tau
    def update_dual_tau(self):
        step = 0.1
        for i, agent in enumerate(self.agents):
            p_i = np.sum(self.P_mat[i, :])
            self.tau[i] = max(0.0, self.tau[i] + step * (p_i - agent.p_max))

    # local agent dual update - phi
    def update_dual_phi(self):
        step = 0.1

        for i, agent in enumerate(self.agents):
            p_i = np.sum(self.P_mat[i, :])
            self.phi[i] = max(0.0, self.phi[i] + step * (agent.p_min - p_i))