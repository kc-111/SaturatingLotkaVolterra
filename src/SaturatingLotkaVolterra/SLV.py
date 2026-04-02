import torch
from safetensors.torch import save_file, load_file
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
        negative_interaction_prob: float = 0.5,
        A_sign: torch.Tensor | None = None,
        K2_mask: torch.Tensor | None = None,
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
        K2_exist_prob (float): Probability of K2 existing (ignored if K2_mask is provided)
        interaction_prob (float): Probability of an off-diagonal interaction existing
        negative_interaction_prob (float): Probability that an off-diagonal interaction
            is negative/inhibitory (ignored if A_sign is provided)
        A_sign (torch.Tensor | None): Optional (n_species, n_species) tensor with entries
            in {-1, 0, +1} for deterministic sign control of interactions.
            Overrides negative_interaction_prob when provided.
        K2_mask (torch.Tensor | None): Optional (n_species, n_species) float tensor (0 or 1).
            1 = finite K2 (non-monotone), 0 = K2 set to 1e6 (monotone).
            Overrides K2_exist_prob when provided.
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

        # Off-diagonal interaction matrix with sign control
        if A_sign is not None:
            self.A = torch.randn(n_species, n_species, device=device).abs() * A_off_diag_scale * interaction_mask * A_sign.to(device)
        else:
            neg_mask = (torch.rand(n_species, n_species, device=device) < negative_interaction_prob).float()
            sign = 1.0 - 2.0 * neg_mask  # +1 where not negative, -1 where negative
            self.A = torch.randn(n_species, n_species, device=device).abs() * A_off_diag_scale * interaction_mask * sign

        # Nonmonotonic saturating interaction is x/(x + K1 + x^2/K2)
        # K1 always exists, K2 has K2_exist_prob chance of existing, if it does
        # not exist, K2 is set to 1e6
        self.K1 = torch.randn(n_species, n_species, device=device).abs() * K1_scale + 1e-6
        self.K2 = torch.randn(n_species, n_species, device=device).abs() * K2_scale + 1e-6
        if K2_mask is not None:
            k2_m = K2_mask.to(device).float()
        else:
            k2_m = (torch.rand(n_species, n_species, device=device) < K2_exist_prob).float()
        self.K2 = self.K2 * k2_m + 1e6 * (1.0 - k2_m)

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

    def save(self, path: str) -> None:
        """Save model parameters to a safetensors file."""
        tensors = {
            "mu": self.mu,
            "A_diag": self.A_diag,
            "A": self.A,
            "K1": self.K1,
            "K2": self.K2,
            "interaction_mask": self.interaction_mask,
        }
        metadata = {"n_species": str(self.n_species)}
        save_file(tensors, path, metadata=metadata)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "SLV":
        """Load model parameters from a safetensors file."""
        tensors = load_file(path, device=str(device))
        # Read metadata to get n_species
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
        obj = object.__new__(cls)
        obj.n_species = int(metadata["n_species"])
        obj.mu = tensors["mu"]
        obj.A_diag = tensors["A_diag"]
        obj.A = tensors["A"]
        obj.K1 = tensors["K1"]
        obj.K2 = tensors["K2"]
        obj.interaction_mask = tensors["interaction_mask"]
        return obj

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
