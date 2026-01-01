import torch

from .internal_ops import cascade_scattering

"""
Autograd-safe eigen-decomposition with optional Lorentzian broadening.
"""


class EgienSolve(torch.autograd.Function):
    """Complex-domain eigendecomposition with stabilized gradients."""

    broadening_parameter = 1e-10

    @staticmethod
    def forward(ctx, matrix):
        eigval, eigvec = torch.linalg.eig(matrix)
        ctx.input = matrix
        ctx.eigval = eigval.detach().cpu()
        ctx.eigvec = eigvec.detach().cpu()
        return eigval, eigvec

    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvec):
        eigval = ctx.eigval.to(grad_eigval)
        eigvec = ctx.eigvec.to(grad_eigvec)

        grad_eigval = torch.diag(grad_eigval)
        spacing = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

        if EgienSolve.broadening_parameter is not None:
            F = torch.conj(spacing) / (torch.abs(spacing) ** 2 + EgienSolve.broadening_parameter)
        elif spacing.dtype == torch.complex64:
            F = torch.conj(spacing) / (torch.abs(spacing) ** 2 + 1.4e-45)
        else:
            F = torch.conj(spacing) / (torch.abs(spacing) ** 2 + 4.9e-324)

        diag_idx = torch.arange(F.shape[-1], dtype=torch.int64)
        F[diag_idx, diag_idx] = 0.

        eigvec_h = torch.transpose(torch.conj(eigvec), -2, -1)
        tmp = torch.conj(F) * torch.matmul(eigvec_h, grad_eigvec)

        grad = torch.matmul(torch.matmul(torch.inverse(eigvec_h), grad_eigval + tmp), eigvec_h)
        if not torch.is_complex(ctx.input):
            grad = torch.real(grad)

        return grad


def solve_uniform_modes(sim, eps, mu):
    P = torch.hstack((torch.vstack((torch.zeros_like(sim.permeability_conv[-1]), -sim.permeability_conv[-1])),
                      torch.vstack((sim.permeability_conv[-1], torch.zeros_like(sim.permeability_conv[-1]))))) + \
        1 / eps * torch.matmul(torch.vstack((sim.kx_diag, sim.ky_diag)),
                               torch.hstack((sim.ky_diag, -sim.kx_diag)))
    Q = torch.hstack((torch.vstack((torch.zeros_like(sim.permittivity_conv[-1]), sim.permittivity_conv[-1])),
                      torch.vstack((-sim.permittivity_conv[-1], torch.zeros_like(sim.permittivity_conv[-1]))))) + \
        1 / mu * torch.matmul(torch.vstack((sim.kx_diag, sim.ky_diag)),
                               torch.hstack((-sim.ky_diag, sim.kx_diag)))
    sim.p_mappers.append(P)
    sim.q_mappers.append(Q)

    e_modes = torch.eye(sim.p_mappers[-1].shape[-1], dtype=sim._dtype, device=sim._device)
    kz = torch.sqrt(eps * mu - sim.kx_vector ** 2 - sim.ky_vector ** 2)
    kz = torch.where(torch.imag(kz) < 0, torch.conj(kz), kz)
    kz = torch.cat((kz, kz))

    sim.kz_eigenvalues.append(kz)
    sim.e_mode_vectors.append(e_modes)


def solve_patterned_modes(sim):
    p_tmp = torch.matmul(torch.vstack((sim.kx_diag, sim.ky_diag)), torch.linalg.inv(sim.permittivity_conv[-1]))
    P = torch.hstack((torch.vstack((torch.zeros_like(sim.permeability_conv[-1]), -sim.permeability_conv[-1])),
                      torch.vstack((sim.permeability_conv[-1], torch.zeros_like(sim.permeability_conv[-1]))))) + \
        torch.matmul(p_tmp, torch.hstack((sim.ky_diag, -sim.kx_diag)))
    sim.p_mappers.append(P)

    q_tmp = torch.matmul(torch.vstack((sim.kx_diag, sim.ky_diag)), torch.linalg.inv(sim.permeability_conv[-1]))
    Q = torch.hstack((torch.vstack((torch.zeros_like(sim.permittivity_conv[-1]), sim.permittivity_conv[-1])),
                      torch.vstack((-sim.permittivity_conv[-1], torch.zeros_like(sim.permittivity_conv[-1]))))) + \
        torch.matmul(q_tmp, torch.hstack((-sim.ky_diag, sim.kx_diag)))
    sim.q_mappers.append(Q)

    if sim.eig_smoothing_enabled:
        kz_norm, e_vec = EgienSolve.apply(torch.matmul(sim.p_mappers[-1], sim.q_mappers[-1]))
    else:
        kz_norm, e_vec = torch.linalg.eig(torch.matmul(sim.p_mappers[-1], sim.q_mappers[-1]))

    kz_norm = torch.sqrt(kz_norm)
    kz_norm = torch.where(torch.imag(kz_norm) < 0, -kz_norm, kz_norm)
    sim.kz_eigenvalues.append(kz_norm)
    sim.e_mode_vectors.append(e_vec)


def solve_layer_scattering_matrix(sim):
    Kz_norm = torch.diag(sim.kz_eigenvalues[-1])
    phase = torch.diag(torch.exp(1.j * sim.angular_freq * sim.kz_eigenvalues[-1] * sim.layer_thicknesses[-1]))

    Pinv_tmp = torch.linalg.inv(sim.p_mappers[-1])
    H_eigvec = torch.matmul(Pinv_tmp, torch.matmul(sim.e_mode_vectors[-1], Kz_norm))
    sim.h_mode_vectors.append(H_eigvec)

    Ctmp1 = torch.vstack((sim.e_mode_vectors[-1] + torch.matmul(torch.linalg.inv(sim.forward_transform), sim.h_mode_vectors[-1]),
                          torch.matmul(sim.e_mode_vectors[-1] - torch.matmul(torch.linalg.inv(sim.forward_transform), sim.h_mode_vectors[-1]), phase)))
    Ctmp2 = torch.vstack((torch.matmul(sim.e_mode_vectors[-1] - torch.matmul(torch.linalg.inv(sim.forward_transform), sim.h_mode_vectors[-1]), phase),
                          sim.e_mode_vectors[-1] + torch.matmul(torch.linalg.inv(sim.forward_transform), sim.h_mode_vectors[-1])))
    Ctmp = torch.hstack((Ctmp1, Ctmp2))

    identity = torch.eye(2 * sim.mode_count, dtype=sim._dtype, device=sim._device)
    Cinv = torch.linalg.inv(Ctmp)
    sim.coupling_forward.append(torch.matmul(Cinv, torch.vstack((2 * identity,
                                                                 torch.zeros([2 * sim.mode_count, 2 * sim.mode_count], dtype=sim._dtype, device=sim._device)))))
    sim.coupling_backward.append(torch.matmul(Cinv, torch.vstack((torch.zeros([2 * sim.mode_count, 2 * sim.mode_count], dtype=sim._dtype, device=sim._device),
                                                                   2 * identity))))

    sim.s11_layers.append(torch.matmul(torch.matmul(sim.e_mode_vectors[-1], phase), sim.coupling_forward[-1][:2 * sim.mode_count, :]) +
                          torch.matmul(sim.e_mode_vectors[-1], sim.coupling_forward[-1][2 * sim.mode_count:, :]))
    sim.s21_layers.append(torch.matmul(sim.e_mode_vectors[-1], sim.coupling_forward[-1][:2 * sim.mode_count, :]) +
                          torch.matmul(torch.matmul(sim.e_mode_vectors[-1], phase), sim.coupling_forward[-1][2 * sim.mode_count:, :]) -
                          identity)
    sim.s12_layers.append(torch.matmul(torch.matmul(sim.e_mode_vectors[-1], phase), sim.coupling_backward[-1][:2 * sim.mode_count, :]) +
                          torch.matmul(sim.e_mode_vectors[-1], sim.coupling_backward[-1][2 * sim.mode_count:, :]) -
                          identity)
    sim.s22_layers.append(torch.matmul(sim.e_mode_vectors[-1], sim.coupling_backward[-1][:2 * sim.mode_count, :]) +
                          torch.matmul(torch.matmul(sim.e_mode_vectors[-1], phase), sim.coupling_backward[-1][2 * sim.mode_count:, :]))


def assemble_global_s_matrix(sim):
    if sim.stack_depth > 0:
        S11, S21, S12, S22 = (
            sim.s11_layers[0],
            sim.s21_layers[0],
            sim.s12_layers[0],
            sim.s22_layers[0],
        )
        coupling = [[sim.coupling_forward[0]], [sim.coupling_backward[0]]]
    else:
        S11 = torch.eye(2 * sim.mode_count, dtype=sim._dtype, device=sim._device)
        S21 = torch.zeros(2 * sim.mode_count, dtype=sim._dtype, device=sim._device)
        S12 = torch.zeros(2 * sim.mode_count, dtype=sim._dtype, device=sim._device)
        S22 = torch.eye(2 * sim.mode_count, dtype=sim._dtype, device=sim._device)
        coupling = [[], []]

    for idx in range(sim.stack_depth - 1):
        [S11, S21, S12, S22], coupling = cascade_scattering(
            sim,
            Sm=[S11, S21, S12, S22],
            Sn=[sim.s11_layers[idx + 1], sim.s21_layers[idx + 1], sim.s12_layers[idx + 1], sim.s22_layers[idx + 1]],
            Cm=coupling,
            Cn=[[sim.coupling_forward[idx + 1]], [sim.coupling_backward[idx + 1]]],
        )

    if hasattr(sim, "input_scattering"):
        [S11, S21, S12, S22], coupling = cascade_scattering(
            sim,
            Sm=[sim.input_scattering[0], sim.input_scattering[1], sim.input_scattering[2], sim.input_scattering[3]],
            Sn=[S11, S21, S12, S22],
            Cm=[[], []],
            Cn=coupling,
        )

    if hasattr(sim, "output_scattering"):
        [S11, S21, S12, S22], coupling = cascade_scattering(
            sim,
            Sm=[S11, S21, S12, S22],
            Sn=[sim.output_scattering[0], sim.output_scattering[1], sim.output_scattering[2], sim.output_scattering[3]],
            Cm=coupling,
            Cn=[[], []],
        )

    sim.global_scattering = [S11, S21, S12, S22]
    sim.cascaded_couplings = coupling
    return sim.global_scattering
