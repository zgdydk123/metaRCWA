import functools
import numpy as np
import torch
from scipy.interpolate import interp1d


def _load_table(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 2]  # wavelength, n, k


@functools.lru_cache(maxsize=None)
def _interp_functions(path):
    wavelength, n, k = _load_table(path)
    n_interp = interp1d(wavelength, n, kind="cubic", fill_value="extrapolate")
    k_interp = interp1d(wavelength, k, kind="cubic", fill_value="extrapolate")
    return wavelength, n_interp, k_interp


def _sample_complex_index(path, wavelength_np, delta):
    w_grid, n_fn, k_fn = _interp_functions(path)

    def clamp_eval(val):
        if val <= w_grid[0]:
            return n_fn(w_grid[0]) + 1j * k_fn(w_grid[0])
        if val >= w_grid[-1]:
            return n_fn(w_grid[-1]) + 1j * k_fn(w_grid[-1])
        return n_fn(val) + 1j * k_fn(val)

    center = clamp_eval(wavelength_np)
    minus = clamp_eval(wavelength_np - delta)
    plus = clamp_eval(wavelength_np + delta)
    grad = (plus - minus) / (2 * delta)
    return center, grad


def _ensure_complex_dtype(x):
    return torch.complex128 if x.dtype in (torch.float64, torch.complex128) else torch.complex64


def make_material_function(table_path):
    class Dispersion(torch.autograd.Function):
        @staticmethod
        def forward(ctx, wavelength, dl=0.005):
            w_np = wavelength.detach().cpu().numpy()
            nk_value, dnk = _sample_complex_index(table_path, w_np, dl)
            ctx.dnk_dl = dnk
            return torch.tensor(nk_value, dtype=_ensure_complex_dtype(wavelength), device=wavelength.device)

        @staticmethod
        def backward(ctx, grad_output):
            grad = 2 * torch.real(torch.conj(grad_output) * ctx.dnk_dl)
            return grad, None

    return Dispersion


# Material-specific autograd Functions
aSiH = make_material_function("Materials_data/aSiH.txt")
gold = make_material_function("Materials_data/gold.txt")
