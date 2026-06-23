import numpy as np
from src.admm.AdaptivePenaltyADMM import AdaptivePenaltyADMM
from src.admm.SimpleADMM import SimpleADMM


class FastADMM(AdaptivePenaltyADMM):
    def __init__(self, mu=1, **kwargs):
        super().__init__(**kwargs)
        self.mu_k = mu
        self.lambda_tilde_prev = np.zeros((self.N, self.N))

    # Fast-ADMM specific dual update
    def update_dual_lambda(self):
        # normal admm update term
        lambda_tilde = self.lambda_mat + self.rho * (self.pi_mat - self.P_mat)

        # fast admm accelerator term
        mu_k_plus_1 = (1 + np.sqrt(1 + 4 * (self.mu_k**2))) / 2

        # total update
        self.lambda_mat = lambda_tilde + ((self.mu_k - 1) / mu_k_plus_1) * (lambda_tilde - self.lambda_tilde_prev)

        # update mu_k and lambda_tile_prev values for next iteration
        self.mu_k = mu_k_plus_1
        self.lambda_tilde_prev = lambda_tilde