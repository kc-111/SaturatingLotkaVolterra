import torch

class Tsit5SolverTorch:
    """
    An adaptive step-size Tsitouras 5(4) Runge-Kutta solver with dense output
    via cubic Hermite interpolation, implemented in PyTorch.

    Dense output: The solver takes natural adaptive steps and interpolates the
    solution at requested output times using the FSAL (First Same As Last)
    property, which provides derivatives at both endpoints of each step for free.

    All intermediate buffers are preallocated to eliminate per-step memory
    allocation overhead — critical for large spatial systems.

    Tsit5 Solver:
    Tsitouras, C. (2011). Runge-Kutta pairs of order 5 (4) satisfying only the
    first column simplifying assumption. Computers & mathematics with applications,
    62(2), 770-775.

    Error Control:
    Hairer, E., Wanner, G., & Norsett, S. P. (1993). Solving ordinary
    differential equations I: Nonstiff problems. Berlin, Heidelberg:
    Springer Berlin Heidelberg.
    """
    def __init__(self, atol=1e-6, rtol=1e-6, h_min=1e-8, h_max=10.0, maxiters=1000000):
        self.atol = atol
        self.rtol = rtol
        self.h_min = h_min
        self.h_max = h_max
        self.maxiters = maxiters
        self.p = 4
        self.safety_factor = 0.9

        # Tsitouras 5(4) Butcher Tableau — store as Python floats to avoid
        # repeated .item() calls in the hot loop
        self.A = [
            [],
            [0.2],
            [0.075, 0.225],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
        ]
        self.c = [0.0, 0.2, 0.3, 0.8, 8/9, 1.0, 1.0]

        # b and e weights as Python float lists (b[1]=0, b[6]=0, e[1]=0)
        self.b = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]
        self.e = [71/57600, 0.0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]

        # Precompute nonzero indices for b and e to skip zero terms
        self.b_nz = [(i, self.b[i]) for i in range(7) if self.b[i] != 0.0]
        self.e_nz = [(i, self.e[i]) for i in range(7) if self.e[i] != 0.0]

    def solve(self, fun, y0, t_span, t_eval, args=None, h0=0.1):
        """
        Solves a batch of ODEs using dense output with preallocated buffers.

        Args:
            fun (callable): fun(t, y, args) -> dy/dt
            y0 (torch.Tensor): Initial state, (num_samples, num_vars).
            t_span (tuple): (t_start, t_end).
            t_eval (torch.Tensor): Output time points.
            args: Additional arguments for fun.
            h0 (float): Initial step size.

        Returns:
            torch.Tensor: (num_samples, len(t_eval), num_vars).
        """
        device, dtype = y0.device, y0.dtype
        num_samples, num_vars = y0.shape
        A, c, b_nz, e_nz = self.A, self.c, self.b_nz, self.e_nz
        atol, rtol = self.atol, self.rtol

        t_eval = torch.as_tensor(t_eval, device=device, dtype=torch.float64)
        t_end = t_eval[-1].item()
        n_eval = len(t_eval)

        # ---- Preallocate ALL buffers ----
        ks = torch.zeros((7, num_samples, num_vars), dtype=dtype, device=device)
        dy = torch.empty((num_samples, num_vars), dtype=dtype, device=device)
        y_stage = torch.empty_like(dy)
        y_new = torch.empty_like(dy)
        error_estimate = torch.empty_like(dy)
        abs_max = torch.empty_like(dy)
        scaled_err = torch.empty_like(dy)
        results_y = torch.zeros((num_samples, n_eval, num_vars), dtype=dtype, device=device)

        inv_sqrt_nv = 1.0 / (num_vars ** 0.5)

        # ---- State ----
        y = y0.clone()
        t = float(t_span[0])
        h = h0

        # Preallocate FSAL buffer (own memory, never aliases ks)
        f_current = torch.empty((num_samples, num_vars), dtype=dtype, device=device)
        f_current.copy_(fun(t, y, args))

        # Store initial point
        eval_idx = 0
        if t_eval[0].item() <= t + 1e-12:
            results_y[:, 0, :] = y
            eval_idx = 1

        iters = 0
        while eval_idx < n_eval and iters < self.maxiters:
            iters += 1
            h_current = min(h, t_end - t)
            if h_current < self.h_min:
                h_current = self.h_min

            # ---- RK stages (FSAL: ks[0] = f_current) ----
            ks[0].copy_(f_current)
            for j in range(1, 7):
                Aj = A[j]
                # dy = sum_k A[j,k] * ks[k] for k < j
                dy.copy_(ks[0]).mul_(Aj[0])
                for k in range(1, j):
                    dy.add_(ks[k], alpha=Aj[k])
                # y_stage = y + h * dy
                torch.add(y, dy, alpha=h_current, out=y_stage)
                ks[j] = fun(t + h_current * c[j], y_stage, args)

            # ---- y_new = y + h * sum(b[i] * ks[i]) ----
            i0, b0 = b_nz[0]
            y_new.copy_(ks[i0]).mul_(b0)
            for i, bi in b_nz[1:]:
                y_new.add_(ks[i], alpha=bi)
            y_new.mul_(h_current).add_(y)

            # ---- error = h * sum(e[i] * ks[i]) ----
            i0, e0 = e_nz[0]
            error_estimate.copy_(ks[i0]).mul_(e0)
            for i, ei in e_nz[1:]:
                error_estimate.add_(ks[i], alpha=ei)
            error_estimate.mul_(h_current)

            # ---- Adaptive step-size control (fused, in-place) ----
            torch.maximum(y.abs(), y_new.abs(), out=abs_max)
            scaled_err.copy_(abs_max).mul_(rtol).add_(atol + 1e-9)
            torch.div(error_estimate, scaled_err, out=scaled_err)
            scaled_err.square_()
            err = scaled_err.sum(dim=1).max().item() ** 0.5 * inv_sqrt_nv

            if err <= 1.0:  # Accept
                t_new = t + h_current
                y_new.clamp_(min=0.0)

                # Interpolate at t_eval points within [t, t_new]
                # Use ks[0] for f_start (safe copy), ks[6] for f_end
                while eval_idx < n_eval and t_eval[eval_idx].item() <= t_new + 1e-12:
                    t_target = t_eval[eval_idx].item()
                    theta = (t_target - t) / h_current if h_current > 1e-15 else 1.0
                    theta = max(0.0, min(1.0, theta))
                    _hermite_interp_out(theta, y, y_new, ks[0], ks[6],
                                        h_current, results_y[:, eval_idx, :])
                    results_y[:, eval_idx, :].clamp_(min=0.0)
                    eval_idx += 1

                t = t_new
                y.copy_(y_new)
                f_current.copy_(ks[6])  # FSAL: copy into own buffer

                q = (1.0 / (err + 1e-9)) ** 0.2  # 1/(p+1) = 1/5
                h = min(self.h_max, h_current * self.safety_factor * q)
            else:
                q = (1.0 / err) ** 0.2
                h = max(self.h_min, h * self.safety_factor * q)

        if eval_idx < n_eval:
            raise ValueError(
                f"Solver reached maxiters ({self.maxiters}) with {n_eval - eval_idx} "
                f"output points remaining at t={t:.6f}."
            )

        return results_y


def _hermite_interp_out(theta, y0, y1, f0, f1, h, out):
    """Cubic Hermite interpolation, writing result into `out` buffer."""
    h00 = 2 * theta**3 - 3 * theta**2 + 1
    h10 = theta**3 - 2 * theta**2 + theta
    h01 = -2 * theta**3 + 3 * theta**2
    h11 = theta**3 - theta**2
    # out = h00*y0 + h01*y1 + h10*h*f0 + h11*h*f1
    out.copy_(y0).mul_(h00)
    out.add_(y1, alpha=h01)
    out.add_(f0, alpha=h10 * h)
    out.add_(f1, alpha=h11 * h)
