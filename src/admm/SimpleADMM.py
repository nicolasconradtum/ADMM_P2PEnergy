import numpy as np
from .ADMM import ADMM

class SimpleADMM(ADMM):
    # local primal update
    def update_primal(self):
        for i, agent in enumerate(self.agents):
            # grid constraints not enforced for grid agent, purely nodal power constraint
            if self.enforce_grid_constraints:
                G_i = self.grid_manager.compute_G_i(i)
            else:
                G_i = 0

            self.P_mat[i, :] = agent.solve_local(
                P_i=self.P_mat[i, :],
                pi_row=self.pi_mat[i, :],
                lambda_row=self.lambda_mat[i, :],
                rho=self.rho,
                G_i=G_i
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
        self.lambda_mat = np.clip(self.lambda_mat, 5, 15)
        