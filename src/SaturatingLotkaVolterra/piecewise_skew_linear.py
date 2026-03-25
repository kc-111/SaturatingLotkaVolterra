import torch
import math


class PiecewiseSkewLinear:
    """
    Piecewise linear dynamical system where each piece is governed by a
    skew-symmetric (antisymmetric) matrix:

        dx/dt = A_i x,   A_i^T = -A_i

    Time is divided into repeating blocks of duration ``block_t``.  Within
    each block the pieces are applied sequentially for fractions
    tau_1, tau_2, ..., tau_k of the block (sum(tau) = 1).

    When ``random_taus=True``, fresh fractions are sampled from a
    Dirichlet distribution independently for every block, producing a
    stochastic switching schedule.

    Because every A_i is skew-symmetric, the flow of each piece is a
    rotation (orthogonal map), so ||x(t)|| = ||x(0)|| is preserved exactly.
    The solver exploits this by computing exact matrix exponentials rather
    than using a numerical ODE integrator.
    """

    def __init__(
        self,
        n_species: int,
        n_pieces: int,
        taus: list[float] | None = None,
        block_t: float = 4.0,
        scale: float = 1.0,
        random_taus: bool = False,
        dirichlet_alpha: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            n_species:       Dimension of the state vector.
            n_pieces:        Number of piecewise linear sub-systems.
            taus:            Time fractions for each piece within a block
                             (must sum to 1).  Defaults to equal fractions.
                             Ignored when ``random_taus=True``.
            block_t:         Duration of one block.
            scale:           Scale of the random skew-symmetric entries.
            random_taus:     If True, sample new taus from a Dirichlet
                             distribution for every block.
            dirichlet_alpha: Concentration parameter for the Dirichlet
                             distribution (used only when ``random_taus=True``).
                             alpha < 1 pushes mass to corners (sparse),
                             alpha = 1 is uniform on the simplex,
                             alpha > 1 concentrates toward equal fractions.
            device:          Torch device.
        """
        self.n_species = n_species
        self.n_pieces = n_pieces
        self.block_t = block_t
        self.device = device
        self.random_taus = random_taus
        self.dirichlet_alpha = dirichlet_alpha

        if not random_taus:
            if taus is None:
                taus = [1.0 / n_pieces] * n_pieces
            assert len(taus) == n_pieces
            assert abs(sum(taus) - 1.0) < 1e-10, f"taus must sum to 1, got {sum(taus)}"
        self.taus = taus

        # Per-block taus schedule, populated by solve()
        self._block_taus: torch.Tensor | None = None

        # Generate random skew-symmetric matrices
        self.As = []
        for _ in range(n_pieces):
            M = torch.randn(n_species, n_species, device=device) * scale
            A = (M - M.T) / 2  # skew-symmetric
            self.As.append(A)

    def get_active_phase(self, t: float, t_start: float = 0.0) -> tuple[int, torch.Tensor]:
        """Return (phase_index, A_i) for the phase active at time *t*.

        If ``random_taus`` is enabled, ``_block_taus`` must have been
        populated by a prior call to :meth:`solve`.
        """
        block_idx = int((t - t_start) // self.block_t)
        t_in_block = (t - t_start) - block_idx * self.block_t

        if self._block_taus is not None:
            block_idx = min(block_idx, self._block_taus.shape[0] - 1)
            taus = self._block_taus[block_idx]
        else:
            taus = self.taus

        cumulative = 0.0
        for i in range(self.n_pieces):
            tau_i = taus[i].item() if isinstance(taus, torch.Tensor) else taus[i]
            cumulative += tau_i * self.block_t
            if t_in_block < cumulative + 1e-12:
                return i, self.As[i]
        return self.n_pieces - 1, self.As[-1]

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate dx/dt = A(t) x.

        Args:
            t: Current time (scalar).
            x: State tensor of shape ``(n_samples, n_species)``.

        Returns:
            dx/dt of the same shape.
        """
        _, A = self.get_active_phase(t)
        return x @ A.T

    def solve(
        self,
        x0: torch.Tensor,
        t_span: tuple[float, float],
        t_eval: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact solve via matrix exponentials.

        Args:
            x0:     Initial state, shape ``(n_samples, n_species)``.
            t_span: ``(t_start, t_end)``.
            t_eval: Sorted 1-D tensor of output times.

        Returns:
            Solution tensor of shape ``(n_samples, len(t_eval), n_species)``.
        """
        device, dtype = x0.device, x0.dtype
        n_samples, n_species = x0.shape
        n_eval = len(t_eval)
        t_eval = torch.as_tensor(t_eval, device=device, dtype=dtype)

        t_start, t_end = float(t_span[0]), float(t_span[1])
        n_blocks = math.ceil((t_end - t_start) / self.block_t)

        # Build per-block taus schedule: (n_blocks, n_pieces)
        if self.random_taus:
            alpha = torch.full(
                (self.n_pieces,), self.dirichlet_alpha, device=device, dtype=dtype,
            )
            block_taus = torch.distributions.Dirichlet(alpha).sample((n_blocks,))
        else:
            block_taus = (
                torch.tensor(self.taus, device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(n_blocks, -1)
            )
        self._block_taus = block_taus

        # Precompute full-phase propagators per block: list of (n_pieces,) per block
        block_phase_props = []
        block_phase_durs = []
        for b in range(n_blocks):
            durs = (block_taus[b] * self.block_t).tolist()
            props = [
                torch.linalg.matrix_exp(A * dur)
                for A, dur in zip(self.As, durs)
            ]
            block_phase_durs.append(durs)
            block_phase_props.append(props)

        results = torch.zeros(n_samples, n_eval, n_species, dtype=dtype, device=device)

        x = x0.clone()
        eval_idx = 0
        current_t = t_start

        for b in range(n_blocks):
            phase_durs = block_phase_durs[b]
            phase_props = block_phase_props[b]

            for phase_idx in range(self.n_pieces):
                phase_dur = phase_durs[phase_idx]
                phase_end = min(current_t + phase_dur, t_end)

                # Record t_eval points that fall within this phase
                while (
                    eval_idx < n_eval
                    and t_eval[eval_idx].item() <= phase_end + 1e-12
                ):
                    dt = t_eval[eval_idx].item() - current_t
                    if dt < 1e-15:
                        results[:, eval_idx, :] = x
                    else:
                        prop = torch.linalg.matrix_exp(self.As[phase_idx] * dt)
                        results[:, eval_idx, :] = x @ prop.T
                    eval_idx += 1

                # Advance state to end of phase
                actual_dur = phase_end - current_t
                if actual_dur < 1e-15:
                    pass
                elif abs(actual_dur - phase_dur) < 1e-12:
                    x = x @ phase_props[phase_idx].T
                else:
                    prop = torch.linalg.matrix_exp(self.As[phase_idx] * actual_dur)
                    x = x @ prop.T

                current_t = phase_end
                if current_t >= t_end - 1e-12:
                    break

            if current_t >= t_end - 1e-12:
                break

        return results
