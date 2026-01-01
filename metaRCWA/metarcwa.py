import warnings
import torch
import numpy as np

from .solve import (
    EgienSolve,
    solve_uniform_modes,
    solve_patterned_modes,
    solve_layer_scattering_matrix,
    assemble_global_s_matrix,
)
from .field_sampling import (
    field_xz_slice,
    field_yz_slice,
    field_xy_slice,
    dispatch_field_slice,
)
from .internal_ops import (
    material_uniform,
    identity_conv,
    find_order_indices,
    build_reciprocal_basis,
    material_convolution,
    cascade_scattering,
    safe_kz,
    select_kz,
    select_directional_params,
)

PI = np.pi


def _is_complex_dtype(value: torch.dtype) -> bool:
    return value in (torch.complex64, torch.complex128)


class MetaRCWA:
    """RCWA solver with decoupled helper modules for readability."""

    def __init__(self, optical, fourier_order, cell_period, *, precision=None, control=None):
        """
        Parameters
        - optical: operating frequency (length^-1)
        - fourier_order: Fourier order [Nx, Ny]
        - cell_period: lattice constants [Lx, Ly]
        - precision: dict with optional keys {'dtype', 'device'}
        - control: reserved for future controls
        """
        precision = precision or {}
        control = control or {}

        dtype = precision.get("dtype", torch.complex64)
        device = precision.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._dtype = dtype if _is_complex_dtype(dtype) else torch.complex64
        if not _is_complex_dtype(dtype):
            warnings.warn("Invalid dtype requested; defaulting to torch.complex64.", UserWarning)
        self._device = device

        self.eig_smoothing_enabled = True

        self.base_freq = torch.as_tensor(optical, dtype=self._dtype, device=self._device)
        self.angular_freq = 2 * PI * self.base_freq
        self.lattice_periods = torch.as_tensor(cell_period, dtype=self._dtype, device=self._device)

        self.fourier_order = fourier_order
        self.fourier_gx_indices = torch.arange(-fourier_order[0], fourier_order[0] + 1, dtype=torch.int64, device=self._device)
        self.fourier_gy_indices = torch.arange(-fourier_order[1], fourier_order[1] + 1, dtype=torch.int64, device=self._device)
        self.mode_count = len(self.fourier_gx_indices) * len(self.fourier_gy_indices)

        self.recip_gx = 1 / (cell_period[0] * self.base_freq)
        self.recip_gy = 1 / (cell_period[1] * self.base_freq)

        self.permittivity_input = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.permeability_input = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.permittivity_output = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.permeability_output = torch.tensor(1.0, dtype=self._dtype, device=self._device)

        self.stack_depth = 0
        self.layer_thicknesses = []
        self.permittivity_conv = []
        self.permeability_conv = []

        self.p_mappers = []
        self.q_mappers = []
        self.kz_eigenvalues = []
        self.e_mode_vectors = []
        self.h_mode_vectors = []

        self.coupling_forward = []
        self.coupling_backward = []

        self.s11_layers = []
        self.s21_layers = []
        self.s12_layers = []
        self.s22_layers = []

        self._angles_defined = False

    # Boundary configuration -------------------------------------------------
    def configure_input_layer(self, eps=1.0, mu=1.0):
        self.permittivity_input = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        self.permeability_input = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        self.input_scattering = []

    def configure_output_layer(self, eps=1.0, mu=1.0):
        self.permittivity_output = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        self.permeability_output = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        self.output_scattering = []

    # Angles -----------------------------------------------------------------
    def set_incident_angles(self, incident_theta, incident_phi, reference_medium="input"):
        self.incident_theta = torch.as_tensor(incident_theta, dtype=self._dtype, device=self._device)
        self.incident_phi = torch.as_tensor(incident_phi, dtype=self._dtype, device=self._device)

        if reference_medium in ["i", "in", "input"]:
            self.incident_reference = "input"
        elif reference_medium in ["o", "out", "output"]:
            self.incident_reference = "output"
        else:
            warnings.warn("Invalid angle layer; using input.", UserWarning)
            self.incident_reference = "input"

        build_reciprocal_basis(self)
        self._angles_defined = True

    # Layers -----------------------------------------------------------------
    def add_structure_layer(self, thickness, eps=1.0, mu=1.0):
        eps_hom = material_uniform(self, eps)
        mu_hom = material_uniform(self, mu)

        self.permittivity_conv.append(identity_conv(self, eps) if eps_hom else material_convolution(self, eps))
        self.permeability_conv.append(identity_conv(self, mu) if mu_hom else material_convolution(self, mu))

        self.stack_depth += 1
        self.layer_thicknesses.append(thickness)

        if eps_hom and mu_hom:
            solve_uniform_modes(self, eps, mu)
        else:
            solve_patterned_modes(self)

        solve_layer_scattering_matrix(self)

    # Solvers ----------------------------------------------------------------
    def solve_global_s_matrix(self):
        return assemble_global_s_matrix(self)

    # Results ----------------------------------------------------------------
    def diffraction_angles(self, orders, *, layer="output", unit="radian"):
        orders = torch.as_tensor(orders, dtype=torch.int64, device=self._device).reshape([-1, 2])

        if layer in ["i", "in", "input"]:
            layer = "input"
        elif layer in ["o", "out", "output"]:
            layer = "output"
        else:
            warnings.warn("Invalid layer; using output.", UserWarning)
            layer = "output"

        unit = "radian" if unit in ["r", "rad", "radian"] else "degree"

        indices = find_order_indices(self, orders)
        eps = self.permittivity_input if layer == "input" else self.permittivity_output
        mu = self.permeability_input if layer == "input" else self.permeability_output

        kx_norm = self.kx_vector[indices]
        ky_norm = self.ky_vector[indices]
        kt_norm = torch.sqrt(kx_norm ** 2 + ky_norm ** 2)
        kz_norm = torch.sqrt(eps * mu - kx_norm ** 2 - ky_norm ** 2)
        theta_out = torch.atan2(torch.real(kt_norm), torch.real(kz_norm))
        phi_out = torch.atan2(torch.real(ky_norm), torch.real(kx_norm))

        if unit == "degree":
            theta_out = (180.0 / PI) * theta_out
            phi_out = (180.0 / PI) * phi_out

        return theta_out, phi_out

    def recover_layer_profile(self, layer_num, nx=100, ny=100):
        eps_fft = torch.zeros([nx, ny], dtype=self._dtype, device=self._device)
        mu_fft = torch.zeros([nx, ny], dtype=self._dtype, device=self._device)

        for i in range(-2 * self.fourier_order[0], 2 * self.fourier_order[0] + 1):
            for j in range(-2 * self.fourier_order[1], 2 * self.fourier_order[1] + 1):
                row_shift = abs(i) * (2 * self.fourier_order[1] + 1)
                if i >= 0 and j >= 0:
                    eps_fft[i, j] = self.permittivity_conv[layer_num][row_shift + j, 0]
                    mu_fft[i, j] = self.permeability_conv[layer_num][row_shift + j, 0]
                elif i >= 0 and j < 0:
                    eps_fft[i, j] = self.permittivity_conv[layer_num][row_shift, -j]
                    mu_fft[i, j] = self.permeability_conv[layer_num][row_shift, -j]
                elif i < 0 and j >= 0:
                    eps_fft[i, j] = self.permittivity_conv[layer_num][j, -i * (2 * self.fourier_order[1] + 1)]
                    mu_fft[i, j] = self.permeability_conv[layer_num][j, -i * (2 * self.fourier_order[1] + 1)]
                else:
                    eps_fft[i, j] = self.permittivity_conv[layer_num][0, -i * (2 * self.fourier_order[1] + 1) - j]
                    mu_fft[i, j] = self.permeability_conv[layer_num][0, -i * (2 * self.fourier_order[1] + 1) - j]

        eps_recover = torch.fft.ifftn(eps_fft) * nx * ny
        mu_recover = torch.fft.ifftn(mu_fft) * nx * ny
        return eps_recover, mu_recover

    def scattering_parameters(self, orders, *, direction="forward", port="transmission", polarization="xx", ref_order=[0, 0], power_norm=True, evanscent=1e-3):
        orders = torch.as_tensor(orders, dtype=torch.int64, device=self._device).reshape([-1, 2])

        direction = "forward" if direction in ["f", "forward"] else "backward"
        port = "transmission" if port in ["t", "transmission"] else "reflection"

        if polarization not in ["xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"]:
            warnings.warn("Invalid polarization; defaulting to xx.", UserWarning)
            polarization = "xx"

        ref_order = torch.as_tensor(ref_order, dtype=torch.int64, device=self._device).reshape([1, 2])

        order_indices = find_order_indices(self, orders)
        ref_order_index = find_order_indices(self, ref_order)

        if polarization in ["xx", "yx", "xy", "yy"]:
            order_indices = order_indices + (self.mode_count if polarization in ["yx", "yy"] else 0)
            ref_order_index = ref_order_index + (self.mode_count if polarization in ["xy", "yy"] else 0)

            if power_norm:
                kz_in = safe_kz(self, self.permittivity_input, self.permeability_input, evanscent)
                kz_out = safe_kz(self, self.permittivity_output, self.permeability_output, evanscent)
                kx = torch.hstack((torch.real(self.kx_vector), torch.real(self.kx_vector)))
                ky = torch.hstack((torch.real(self.ky_vector), torch.real(self.ky_vector)))

                pol_num = kx if polarization in ["xx", "xy"] else ky
                pol_den = kx if polarization in ["xx", "yx"] else ky

                kz_num, kz_den = select_kz(direction, port, kz_in, kz_out)
                normalization = torch.sqrt((1 + (pol_num[order_indices] / kz_num[order_indices]) ** 2) / (1 + (pol_den[ref_order_index] / kz_den[ref_order_index]) ** 2))
                normalization = normalization * torch.sqrt(kz_num[order_indices] / kz_den[ref_order_index])
            else:
                normalization = 1.0

            if direction == "forward" and port == "transmission":
                S = self.global_scattering[0][order_indices, ref_order_index] * normalization
            elif direction == "forward" and port == "reflection":
                S = self.global_scattering[1][order_indices, ref_order_index] * normalization
            elif direction == "backward" and port == "reflection":
                S = self.global_scattering[2][order_indices, ref_order_index] * normalization
            else:
                S = self.global_scattering[3][order_indices, ref_order_index] * normalization

            S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
            return S

        idx, order_sign, ref_sign, order_k0, ref_k0 = select_directional_params(self, direction, port)
        order_idx = order_indices
        ref_idx = ref_order_index

        kx_o, ky_o = self.kx_vector[order_idx], self.ky_vector[order_idx]
        kx_r, ky_r = self.kx_vector[ref_idx], self.ky_vector[ref_idx]

        kt_o = torch.sqrt(kx_o ** 2 + ky_o ** 2)
        kt_r = torch.sqrt(kx_r ** 2 + ky_r ** 2)
        kz_o_complex = torch.sqrt(order_k0 - kx_o ** 2 - ky_o ** 2)
        kz_r_complex = torch.sqrt(ref_k0 - kx_r ** 2 - ky_r ** 2)

        kz_o = order_sign * torch.abs(torch.real(kz_o_complex))
        kz_r = ref_sign * torch.abs(torch.real(kz_r_complex))

        evan_o = torch.abs(torch.real(kz_o_complex) / torch.imag(kz_o_complex)) < evanscent
        evan_r = torch.abs(torch.real(kz_r_complex) / torch.imag(kz_r_complex)) < evanscent

        theta_o = torch.atan2(torch.real(kt_o), kz_o)
        theta_r = torch.atan2(torch.real(kt_r), kz_r)
        phi_o = torch.atan2(torch.real(ky_o), torch.real(kx_o))
        phi_r = torch.atan2(torch.real(ky_r), torch.real(kx_r))

        xx = self.global_scattering[idx][order_idx, ref_idx]
        xy = self.global_scattering[idx][order_idx, ref_idx + self.mode_count]
        yx = self.global_scattering[idx][order_idx + self.mode_count, ref_idx]
        yy = self.global_scattering[idx][order_idx + self.mode_count, ref_idx + self.mode_count]

        xx = torch.where(evan_o, torch.zeros_like(xx), xx)
        xy = torch.where(evan_o, torch.zeros_like(xy), xy)
        yx = torch.where(evan_o, torch.zeros_like(yx), yx)
        yy = torch.where(evan_o, torch.zeros_like(yy), yy)

        if evan_r:
            return torch.zeros_like(xx)

        if polarization == "pp":
            S = (
                torch.cos(phi_o) / torch.cos(theta_o) * torch.cos(theta_r) * torch.cos(phi_r) * xx
                + torch.sin(phi_o) / torch.cos(theta_o) * torch.cos(theta_r) * torch.cos(phi_r) * yx
                + torch.cos(phi_o) / torch.cos(theta_o) * torch.cos(theta_r) * torch.sin(phi_r) * xy
                + torch.sin(phi_o) / torch.cos(theta_o) * torch.cos(theta_r) * torch.sin(phi_r) * yy
            )
        elif polarization == "ps":
            S = (
                torch.cos(phi_o) / torch.cos(theta_o) * (-torch.sin(phi_r)) * xx
                + torch.sin(phi_o) / torch.cos(theta_o) * (-torch.sin(phi_r)) * yx
                + torch.cos(phi_o) / torch.cos(theta_o) * torch.cos(phi_r) * xy
                + torch.sin(phi_o) / torch.cos(theta_o) * torch.cos(phi_r) * yy
            )
        elif polarization == "sp":
            S = (
                -torch.sin(phi_o) * torch.cos(theta_r) * torch.cos(phi_r) * xx
                + torch.cos(phi_o) * torch.cos(theta_r) * torch.cos(phi_r) * yx
                - torch.sin(phi_o) * torch.cos(theta_r) * torch.sin(phi_r) * xy
                + torch.cos(phi_o) * torch.cos(theta_r) * torch.sin(phi_r) * yy
            )
        else:  # ss
            S = (
                -torch.sin(phi_o) * (-torch.sin(phi_r)) * xx
                + torch.cos(phi_o) * (-torch.sin(phi_r)) * yx
                - torch.sin(phi_o) * torch.cos(phi_r) * xy
                + torch.cos(phi_o) * torch.cos(phi_r) * yy
            )

        if power_norm:
            kz_in = safe_kz(self, self.permittivity_input, self.permeability_input, evanscent)
            kz_out = safe_kz(self, self.permittivity_output, self.permeability_output, evanscent)
            kz_num, kz_den = select_kz(direction, port, kz_in, kz_out)
            normalization = torch.sqrt(kz_num[order_idx] / kz_den[ref_idx])
        else:
            normalization = 1.0

        S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
        return S * normalization

    # Sources ----------------------------------------------------------------
    def set_plane_wave_source(self, *, amplitude=[1.0, 0.0], direction="forward", notation="xy"):
        self.set_fourier_source(amplitude=amplitude, orders=[0, 0], direction=direction, notation=notation)

    def set_fourier_source(self, *, amplitude, orders, direction="forward", notation="xy"):
        amplitude = torch.as_tensor(amplitude, dtype=self._dtype, device=self._device).reshape([-1, 2])
        orders = torch.as_tensor(orders, dtype=torch.int64, device=self._device).reshape([-1, 2])

        direction = "forward" if direction in ["f", "forward"] else "backward"
        if notation not in ["xy", "ps"]:
            warnings.warn("Invalid notation; defaulting to xy.", UserWarning)
            notation = "xy"

        order_indices = find_order_indices(self, orders)
        self.source_direction = direction

        E_i = torch.zeros([2 * self.mode_count, 1], dtype=self._dtype, device=self._device)
        E_i[order_indices, 0] = amplitude[:, 0]
        E_i[order_indices + self.mode_count, 0] = amplitude[:, 1]

        if notation == "ps":
            eps, mu = (self.permittivity_input, self.permeability_input) if direction == "forward" else (self.permittivity_output, self.permeability_output)
            sign = 1 if direction == "forward" else -1

            Kt_norm_dn = torch.sqrt(self.kx_vector ** 2 + self.ky_vector ** 2)
            Kz_norm_dn = sign * torch.abs(torch.real(torch.sqrt(eps * mu - self.kx_vector ** 2 - self.ky_vector ** 2)))

            incident_theta = torch.atan2(torch.real(Kt_norm_dn), Kz_norm_dn)
            incident_phi = torch.atan2(torch.real(self.ky_vector), torch.real(self.kx_vector))

            tmp1 = torch.vstack(
                (
                    torch.diag(torch.cos(incident_theta) * torch.cos(incident_phi)),
                    torch.diag(torch.cos(incident_theta) * torch.sin(incident_phi)),
                )
            )
            tmp2 = torch.vstack((torch.diag(-torch.sin(incident_phi)), torch.diag(torch.cos(incident_phi))))
            ps2xy = torch.hstack((tmp1, tmp2))

            E_i = torch.matmul(ps2xy.to(self._dtype), E_i)

        self.source_field = E_i

    # Field probes -----------------------------------------------------------
    def field_xz_profile(self, x_axis, z_axis, y):
        return field_xz_slice(self, x_axis, z_axis, y)

    def field_yz_profile(self, y_axis, z_axis, x):
        return field_yz_slice(self, y_axis, z_axis, x)

    def field_xy_profile(self, layer_num, x_axis, y_axis, z_prop=0.):
        return field_xy_slice(self, layer_num, x_axis, y_axis, z_prop=z_prop)

    def field_slice(self, plane, *, x_axis=None, y_axis=None, z_axis=None, layer_index=None, plane_position=0.0, z_offset=0.0):
        return dispatch_field_slice(
            self,
            plane,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            layer_index=layer_index,
            plane_position=plane_position,
            z_offset=z_offset,
        )
