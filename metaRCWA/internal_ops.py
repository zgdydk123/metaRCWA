import torch

def material_uniform(sim, mat):
    return (
        isinstance(mat, (float, complex))
        or (isinstance(mat, torch.Tensor) and (mat.dim() == 0 or (mat.dim() == 1 and mat.shape[0] == 1)))
    )


def identity_conv(sim, value):
    return value * torch.eye(sim.mode_count, dtype=sim._dtype, device=sim._device)


def find_order_indices(sim, orders):
    orders[orders[:, 0] < -sim.fourier_order[0], 0] = int(-sim.fourier_order[0])
    orders[orders[:, 0] > sim.fourier_order[0], 0] = int(sim.fourier_order[0])
    orders[orders[:, 1] < -sim.fourier_order[1], 1] = int(-sim.fourier_order[1])
    orders[orders[:, 1] > sim.fourier_order[1], 1] = int(sim.fourier_order[1])
    return len(sim.fourier_gy_indices) * (orders[:, 0] + int(sim.fourier_order[0])) + orders[:, 1] + int(sim.fourier_order[1])


def build_reciprocal_basis(sim):
    base_eps = sim.permittivity_input if sim.incident_reference == 'input' else sim.permittivity_output
    base_mu = sim.permeability_input if sim.incident_reference == 'input' else sim.permeability_output

    sim.base_kx = torch.real(torch.sqrt(base_eps * base_mu)) * torch.sin(sim.incident_theta) * torch.cos(sim.incident_phi)
    sim.base_ky = torch.real(torch.sqrt(base_eps * base_mu)) * torch.sin(sim.incident_theta) * torch.sin(sim.incident_phi)

    sim.kx_modes = sim.base_kx + sim.fourier_gx_indices * sim.recip_gx
    sim.ky_modes = sim.base_ky + sim.fourier_gy_indices * sim.recip_gy

    kx_grid, ky_grid = torch.meshgrid(sim.kx_modes, sim.ky_modes, indexing='ij')
    sim.kx_vector = torch.reshape(kx_grid, (-1,))
    sim.ky_vector = torch.reshape(ky_grid, (-1,))
    sim.kx_diag = torch.diag(sim.kx_vector)
    sim.ky_diag = torch.diag(sim.ky_vector)

    kz_norm = torch.sqrt(1. - sim.kx_vector ** 2 - sim.ky_vector ** 2)
    kz_norm = torch.where(torch.imag(kz_norm) < 0, torch.conj(kz_norm), kz_norm)
    tmp1 = torch.vstack((torch.diag(-sim.ky_vector * sim.kx_vector / kz_norm),
                         torch.diag(kz_norm + sim.kx_vector ** 2 / kz_norm)))
    tmp2 = torch.vstack((torch.diag(-kz_norm - sim.ky_vector ** 2 / kz_norm),
                         torch.diag(sim.kx_vector * sim.ky_vector / kz_norm)))
    sim.forward_transform = torch.hstack((tmp1, tmp2))

    if hasattr(sim, 'input_scattering'):
        kz_in = torch.sqrt(sim.permittivity_input * sim.permeability_input - sim.kx_vector ** 2 - sim.ky_vector ** 2)
        kz_in = torch.where(torch.imag(kz_in) < 0, torch.conj(kz_in), kz_in)
        tmp1 = torch.vstack((torch.diag(-sim.ky_vector * sim.kx_vector / kz_in),
                             torch.diag(kz_in + sim.kx_vector ** 2 / kz_in)))
        tmp2 = torch.vstack((torch.diag(-kz_in - sim.ky_vector ** 2 / kz_in),
                             torch.diag(sim.kx_vector * sim.ky_vector / kz_in)))
        sim.input_transform = torch.hstack((tmp1, tmp2))

        vtmp1 = torch.linalg.inv(sim.forward_transform + sim.input_transform)
        vtmp2 = sim.forward_transform - sim.input_transform

        sim.input_scattering.append(2 * torch.matmul(vtmp1, sim.input_transform))
        sim.input_scattering.append(-torch.matmul(vtmp1, vtmp2))
        sim.input_scattering.append(torch.matmul(vtmp1, vtmp2))
        sim.input_scattering.append(2 * torch.matmul(vtmp1, sim.forward_transform))

    if hasattr(sim, 'output_scattering'):
        kz_out = torch.sqrt(sim.permittivity_output * sim.permeability_output - sim.kx_vector ** 2 - sim.ky_vector ** 2)
        kz_out = torch.where(torch.imag(kz_out) < 0, torch.conj(kz_out), kz_out)
        tmp1 = torch.vstack((torch.diag(-sim.ky_vector * sim.kx_vector / kz_out),
                             torch.diag(kz_out + sim.kx_vector ** 2 / kz_out)))
        tmp2 = torch.vstack((torch.diag(-kz_out - sim.ky_vector ** 2 / kz_out),
                             torch.diag(sim.kx_vector * sim.ky_vector / kz_out)))
        sim.output_transform = torch.hstack((tmp1, tmp2))

        vtmp1 = torch.linalg.inv(sim.forward_transform + sim.output_transform)
        vtmp2 = sim.forward_transform - sim.output_transform

        sim.output_scattering.append(2 * torch.matmul(vtmp1, sim.forward_transform))
        sim.output_scattering.append(torch.matmul(vtmp1, vtmp2))
        sim.output_scattering.append(-torch.matmul(vtmp1, vtmp2))
        sim.output_scattering.append(2 * torch.matmul(vtmp1, sim.output_transform))


def material_convolution(sim, material):
    material_N = material.shape[0] * material.shape[1]
    gx_grid, gy_grid = torch.meshgrid(sim.fourier_gx_indices, sim.fourier_gy_indices, indexing='ij')
    ox = gx_grid.to(torch.int64).reshape([-1])
    oy = gy_grid.to(torch.int64).reshape([-1])

    ind = torch.arange(len(sim.fourier_gx_indices) * len(sim.fourier_gy_indices), device=sim._device)
    indx, indy = torch.meshgrid(ind.to(torch.int64), ind.to(torch.int64), indexing='ij')

    material_fft = torch.fft.fft2(material) / material_N
    material_fft_real = torch.real(material_fft)
    material_fft_imag = torch.imag(material_fft)

    mat_real = material_fft_real[ox[indx] - ox[indy], oy[indx] - oy[indy]]
    mat_imag = material_fft_imag[ox[indx] - ox[indy], oy[indx] - oy[indy]]
    return torch.complex(mat_real, mat_imag)


def cascade_scattering(sim, Sm, Sn, Cm, Cn):
    tmp1 = torch.linalg.inv(torch.eye(2 * sim.mode_count, dtype=sim._dtype, device=sim._device) - torch.matmul(Sm[2], Sn[1]))
    tmp2 = torch.linalg.inv(torch.eye(2 * sim.mode_count, dtype=sim._dtype, device=sim._device) - torch.matmul(Sn[1], Sm[2]))

    S11 = torch.matmul(Sn[0], torch.matmul(tmp1, Sm[0]))
    S21 = Sm[1] + torch.matmul(Sm[3], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0])))
    S12 = Sn[2] + torch.matmul(Sn[0], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3])))
    S22 = torch.matmul(Sm[3], torch.matmul(tmp2, Sn[3]))

    C = [[], []]
    for m in range(len(Cm[0])):
        C[0].append(Cm[0][m] + torch.matmul(Cm[1][m], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0]))))
        C[1].append(torch.matmul(Cm[1][m], torch.matmul(tmp2, Sn[3])))

    for n in range(len(Cn[0])):
        C[0].append(torch.matmul(Cn[0][n], torch.matmul(tmp1, Sm[0])))
        C[1].append(Cn[1][n] + torch.matmul(Cn[0][n], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3]))))

    return [S11, S21, S12, S22], C


def safe_kz(sim, eps, mu, evanescent_threshold):
    kz_complex = torch.sqrt(eps * mu - sim.kx_vector ** 2 - sim.ky_vector ** 2)
    evan_mask = torch.abs(torch.real(kz_complex) / torch.imag(kz_complex)) < evanescent_threshold
    kz = torch.where(evan_mask, torch.real(torch.zeros_like(kz_complex)), torch.real(kz_complex))
    return torch.hstack((kz, kz))


def select_kz(direction, port, kz_in, kz_out):
    if direction == 'forward' and port == 'transmission':
        return kz_out, kz_in
    if direction == 'forward' and port == 'reflection':
        return kz_in, kz_in
    if direction == 'backward' and port == 'reflection':
        return kz_out, kz_out
    return kz_in, kz_out


def select_directional_params(sim, direction, port):
    if direction == 'forward' and port == 'transmission':
        return 0, 1, 1, sim.permittivity_output * sim.permeability_output, sim.permittivity_input * sim.permeability_input
    if direction == 'forward' and port == 'reflection':
        return 1, -1, 1, sim.permittivity_input * sim.permeability_input, sim.permittivity_input * sim.permeability_input
    if direction == 'backward' and port == 'reflection':
        return 2, 1, -1, sim.permittivity_output * sim.permeability_output, sim.permittivity_output * sim.permeability_output
    return 3, -1, -1, sim.permittivity_input * sim.permeability_input, sim.permittivity_output * sim.permeability_output
