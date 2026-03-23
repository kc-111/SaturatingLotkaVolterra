import torch
from .ode_solver import Tsit5SolverTorch

class SLV:
    def __init__(self,
        n_species: int,
        mu_scale: float = 1.0,
        A_off_diag_scale: float = 1.0,
        A_diag_scale: float = 1.0,
        K1_scale: float = 1.0,
        K2_scale: float = 1.0,
        K2_exist_prob: float = 0.2,
        interaction_prob: float = 0.8,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
        n_species (int): Number of species
        mu_scale (float): Scale of the mu parameter
        A_off_diag_scale (float): Scale of the off-diagonal A parameter
        A_diag_scale (float): Scale of the diagonal A parameter
        K1_scale (float): Scale of the K1 parameter
        K2_scale (float): Scale of the K2 parameter
        K2_exist_prob (float): Probability of K2 existing
        interaction_prob (float): Probability of an off-diagonal interaction existing
        device (torch.device): Device to use

        Returns:
        None

        Initializes the Saturating Lotka Volterra (SLV) model.
        """
        self.n_species = n_species
        self.A_diag = torch.randn(n_species, device=device).abs() * A_diag_scale + 0.1
        self.mu = torch.rand(n_species, device=device) * mu_scale

        # Sparse interaction mask (shared across A, K1, K2)
        interaction_mask = (torch.rand(n_species, n_species, device=device) < interaction_prob).float()
        interaction_mask.fill_diagonal_(0.0)  # no self-interaction in off-diagonal terms
        self.interaction_mask = interaction_mask

        self.A = torch.randn(n_species, n_species, device=device) * A_off_diag_scale * interaction_mask
        # Nonmonotonic saturating interaction is x/(x + K1 + x^2/K2)
        # K1 always exists, K2 has K2_exist_prob chance of existing, if it does
        # not exist, K2 is set to 1e6
        self.K1 = torch.randn(n_species, n_species, device=device).abs() * K1_scale + 1e-6
        self.K2 = torch.randn(n_species, n_species, device=device).abs() * K2_scale + 1e-6
        k2_mask = (torch.rand(n_species, n_species, device=device) < K2_exist_prob).float()
        self.K2 = self.K2 * k2_mask + 1e6 * (1 - k2_mask)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x (torch.Tensor): (n_samples, n_species)

        Returns:
        torch.Tensor: (n_samples, n_species)
        """
        # Calculate the saturating interaction
        ints = x[:, None, :] + self.K1[None, :, :] + x[:, None, :]**2 / self.K2[None, :, :]
        ints = self.A[None, :, :] * 1.0 / ints
        return x * (torch.einsum("ijk,ik->ij", ints, x) + self.mu - self.A_diag[None, :] * x)

    def solve(self, x0: torch.Tensor, t_span: tuple[float, float], t_eval: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x0 (torch.Tensor): (n_samples, n_species)
        t_span (tuple[float, float]): (t_start, t_end)
        t_eval (torch.Tensor): (n_eval)

        Returns:
        torch.Tensor: (n_samples, n_eval, n_species)
        """
        solver = Tsit5SolverTorch()
        return solver.solve(lambda t, y, args: self.__call__(y), x0, t_span, t_eval)
