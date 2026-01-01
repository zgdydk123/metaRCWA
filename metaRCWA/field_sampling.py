import torch
import warnings


def field_xz_slice(sim, x_axis, z_axis, y):
    if not isinstance(x_axis, torch.Tensor) or not isinstance(z_axis, torch.Tensor):
        warnings.warn("x and z axis must be torch.Tensor type. Return None.", UserWarning)
        return None

    x_axis = x_axis.reshape([-1, 1, 1])
    Kx_norm, Ky_norm = sim.kx_diag, sim.ky_diag

    Ex_split, Ey_split, Ez_split = [], [], []
    Hx_split, Hy_split, Hz_split = [], [], []

    cumulative_top = torch.zeros(len(sim.layer_thicknesses), device=sim._device)
    cumulative_bottom = torch.zeros(len(sim.layer_thicknesses), device=sim._device)
    layer_num = torch.zeros([len(z_axis)], dtype=torch.int64, device=sim._device)
    layer_num[z_axis < 0.] = -1

    for idx in range(len(sim.layer_thicknesses)):
        cumulative_top[idx:] += sim.layer_thicknesses[idx]
    cumulative_bottom[1:] = cumulative_top[0:-1]
    for idx in range(len(cumulative_top)):
        layer_num[z_axis > cumulative_top[idx]] += 1

    prev_layer_num = -2
    for zi in range(len(z_axis)):
        if layer_num[zi] in (-1, sim.stack_depth):
            Kx_norm_dn = sim.kx_vector
            Ky_norm_dn = sim.ky_vector

            if layer_num[zi] == -1:
                z_prop = z_axis[zi] if z_axis[zi] <= 0. else 0.
                if layer_num[zi] != prev_layer_num:
                    eps = sim.permittivity_input
                    mu = sim.permeability_input
                    Vi = sim.input_transform if hasattr(sim, 'input_transform') else sim.forward_transform
                    Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn ** 2 - Ky_norm_dn ** 2)
                    Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn) > 0, torch.conj(Kz_norm_dn), Kz_norm_dn).reshape([-1, 1])
                    Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))
            else:
                z_prop = z_axis[zi] - (cumulative_top[-1] if len(cumulative_top) > 0 else 0.)
                z_prop = z_prop if z_prop >= 0. else 0.
                if layer_num[zi] != prev_layer_num:
                    eps = sim.permittivity_output
                    mu = sim.permeability_output
                    Vo = sim.output_transform if hasattr(sim, 'output_transform') else sim.forward_transform
                    Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn ** 2 - Ky_norm_dn ** 2)
                    Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn).reshape([-1, 1])
                    Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

            z_phase = torch.exp(1.j * sim.angular_freq * Kz_norm_dn * z_prop)

            if layer_num[zi] == -1 and sim.source_direction == 'forward':
                Exy_p = sim.source_field * z_phase
                Hxy_p = torch.matmul(Vi, Exy_p)
                Exy_m = torch.matmul(sim.global_scattering[1], sim.source_field) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif layer_num[zi] == -1 and sim.source_direction == 'backward':
                Exy_p = torch.zeros_like(sim.source_field)
                Hxy_p = torch.zeros_like(sim.source_field)
                Exy_m = torch.matmul(sim.global_scattering[3], sim.source_field) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif layer_num[zi] == sim.stack_depth and sim.source_direction == 'forward':
                Exy_p = torch.matmul(sim.global_scattering[0], sim.source_field) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = torch.zeros_like(sim.source_field)
                Hxy_m = torch.zeros_like(sim.source_field)
            else:
                Exy_p = torch.matmul(sim.global_scattering[2], sim.source_field) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = sim.source_field * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vo, Exy_m)

            Ex_mn = Exy_p[:sim.mode_count] + Exy_m[:sim.mode_count]
            Ey_mn = Exy_p[sim.mode_count:] + Exy_m[sim.mode_count:]
            Hz_mn = torch.matmul(Kx_norm, Ey_mn) / mu - torch.matmul(Ky_norm, Ex_mn) / mu
            Hx_mn = Hxy_p[:sim.mode_count] + Hxy_m[:sim.mode_count]
            Hy_mn = Hxy_p[sim.mode_count:] + Hxy_m[sim.mode_count:]
            Ez_mn = torch.matmul(Ky_norm, Hx_mn) / eps - torch.matmul(Kx_norm, Hy_mn) / eps

            xy_phase = torch.exp(1.j * sim.angular_freq * (sim.kx_vector * x_axis + sim.ky_vector * y))
            Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

        else:
            z_prop = z_axis[zi] - cumulative_bottom[layer_num[zi]]

            if layer_num[zi] != prev_layer_num:
                C = torch.matmul(sim.cascaded_couplings[0][layer_num[zi]], sim.source_field) if sim.source_direction == 'forward' else torch.matmul(sim.cascaded_couplings[1][layer_num[zi]], sim.source_field)

                kz_norm = sim.kz_eigenvalues[layer_num[zi]]
                E_vec = sim.e_mode_vectors[layer_num[zi]]
                H_vec = sim.h_mode_vectors[layer_num[zi]]

                Cp = torch.diag(C[:2 * sim.mode_count, 0])
                Cm = torch.diag(C[2 * sim.mode_count:, 0])

                eps_inv = torch.linalg.inv(sim.permittivity_conv[layer_num[zi]])
                mu_inv = torch.linalg.inv(sim.permeability_conv[layer_num[zi]])

            z_phase_p = torch.diag(torch.exp(1.j * sim.angular_freq * kz_norm * z_prop))
            z_phase_m = torch.diag(torch.exp(1.j * sim.angular_freq * kz_norm * (sim.layer_thicknesses[layer_num[zi]] - z_prop)))

            Exy_p = torch.matmul(E_vec, z_phase_p)
            Ex_p = Exy_p[:sim.mode_count, :]
            Ey_p = Exy_p[sim.mode_count:, :]
            Hz_p = torch.matmul(mu_inv, torch.matmul(Kx_norm, Ey_p)) - torch.matmul(mu_inv, torch.matmul(Ky_norm, Ex_p))
            Exy_m = torch.matmul(E_vec, z_phase_m)
            Ex_m = Exy_m[:sim.mode_count, :]
            Ey_m = Exy_m[sim.mode_count:, :]
            Hz_m = torch.matmul(mu_inv, torch.matmul(Kx_norm, Ey_m)) - torch.matmul(mu_inv, torch.matmul(Ky_norm, Ex_m))

            Hxy_p = torch.matmul(H_vec, z_phase_p)
            Hx_p = Hxy_p[:sim.mode_count, :]
            Hy_p = Hxy_p[sim.mode_count:, :]
            Ez_p = torch.matmul(eps_inv, torch.matmul(Ky_norm, Hx_p)) - torch.matmul(eps_inv, torch.matmul(Kx_norm, Hy_p))
            Hxy_m = torch.matmul(-H_vec, z_phase_m)
            Hx_m = Hxy_m[:sim.mode_count, :]
            Hy_m = Hxy_m[sim.mode_count:, :]
            Ez_m = torch.matmul(eps_inv, torch.matmul(Ky_norm, Hx_m)) - torch.matmul(eps_inv, torch.matmul(Kx_norm, Hy_m))

            Ex_mn = torch.sum(torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1)
            Ey_mn = torch.sum(torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1)
            Ez_mn = torch.sum(torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1)
            Hx_mn = torch.sum(torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1)
            Hy_mn = torch.sum(torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1)
            Hz_mn = torch.sum(torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1)

            xy_phase = torch.exp(1.j * sim.angular_freq * (sim.kx_vector * x_axis + sim.ky_vector * y))
            Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
            Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

        prev_layer_num = layer_num[zi]

    Ex = torch.cat(Ex_split, dim=1)
    Ey = torch.cat(Ey_split, dim=1)
    Ez = torch.cat(Ez_split, dim=1)
    Hx = torch.cat(Hx_split, dim=1)
    Hy = torch.cat(Hy_split, dim=1)
    Hz = torch.cat(Hz_split, dim=1)
    return [Ex, Ey, Ez], [Hx, Hy, Hz]


def field_yz_slice(sim, y_axis, z_axis, x):
    if not isinstance(y_axis, torch.Tensor) or not isinstance(z_axis, torch.Tensor):
        warnings.warn("y and z axis must be torch.Tensor type. Return None.", UserWarning)
        return None

    y_axis = y_axis.reshape([-1, 1, 1])
    Kx_norm, Ky_norm = sim.kx_diag, sim.ky_diag

    Ex_split, Ey_split, Ez_split = [], [], []
    Hx_split, Hy_split, Hz_split = [], [], []

    cumulative_top = torch.zeros(len(sim.layer_thicknesses), device=sim._device)
    cumulative_bottom = torch.zeros(len(sim.layer_thicknesses), device=sim._device)
    layer_num = torch.zeros([len(z_axis)], dtype=torch.int64, device=sim._device)
    layer_num[z_axis < 0.] = -1

    for idx in range(len(sim.layer_thicknesses)):
        cumulative_top[idx:] += sim.layer_thicknesses[idx]
    cumulative_bottom[1:] = cumulative_top[0:-1]
    for idx in range(len(cumulative_top)):
        layer_num[z_axis > cumulative_top[idx]] += 1

    prev_layer_num = -2
    for zi in range(len(z_axis)):
        if layer_num[zi] in (-1, sim.stack_depth):
            Kx_norm_dn = sim.kx_vector
            Ky_norm_dn = sim.ky_vector

            if layer_num[zi] == -1:
                z_prop = z_axis[zi] if z_axis[zi] <= 0. else 0.
                if layer_num[zi] != prev_layer_num:
                    eps = sim.permittivity_input
                    mu = sim.permeability_input
                    Vi = sim.input_transform if hasattr(sim, 'input_transform') else sim.forward_transform
                    Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn ** 2 - Ky_norm_dn ** 2)
                    Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn) > 0, torch.conj(Kz_norm_dn), Kz_norm_dn).reshape([-1, 1])
                    Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))
            else:
                z_prop = z_axis[zi] - (cumulative_top[-1] if len(cumulative_top) > 0 else 0.)
                z_prop = z_prop if z_prop >= 0. else 0.
                if layer_num[zi] != prev_layer_num:
                    eps = sim.permittivity_output
                    mu = sim.permeability_output
                    Vo = sim.output_transform if hasattr(sim, 'output_transform') else sim.forward_transform
                    Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn ** 2 - Ky_norm_dn ** 2)
                    Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn).reshape([-1, 1])
                    Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

            z_phase = torch.exp(1.j * sim.angular_freq * Kz_norm_dn * z_prop)

            if layer_num[zi] == -1 and sim.source_direction == 'forward':
                Exy_p = sim.source_field * z_phase
                Hxy_p = torch.matmul(Vi, Exy_p)
                Exy_m = torch.matmul(sim.global_scattering[1], sim.source_field) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif layer_num[zi] == -1 and sim.source_direction == 'backward':
                Exy_p = torch.zeros_like(sim.source_field)
                Hxy_p = torch.zeros_like(sim.source_field)
                Exy_m = torch.matmul(sim.global_scattering[3], sim.source_field) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif layer_num[zi] == sim.stack_depth and sim.source_direction == 'forward':
                Exy_p = torch.matmul(sim.global_scattering[0], sim.source_field) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = torch.zeros_like(sim.source_field)
                Hxy_m = torch.zeros_like(sim.source_field)
            else:
                Exy_p = torch.matmul(sim.global_scattering[2], sim.source_field) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = sim.source_field * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vo, Exy_m)

            Ex_mn = Exy_p[:sim.mode_count] + Exy_m[:sim.mode_count]
            Ey_mn = Exy_p[sim.mode_count:] + Exy_m[sim.mode_count:]
            Hz_mn = torch.matmul(Kx_norm, Ey_mn) / mu - torch.matmul(Ky_norm, Ex_mn) / mu
            Hx_mn = Hxy_p[:sim.mode_count] + Hxy_m[:sim.mode_count]
            Hy_mn = Hxy_p[sim.mode_count:] + Hxy_m[sim.mode_count:]
            Ez_mn = torch.matmul(Ky_norm, Hx_mn) / eps - torch.matmul(Kx_norm, Hy_mn) / eps

            yz_phase = torch.exp(1.j * sim.angular_freq * (sim.kx_vector * x + sim.ky_vector * y_axis))
            Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * yz_phase, dim=2))

        else:
            z_prop = z_axis[zi] - cumulative_bottom[layer_num[zi]]

            if layer_num[zi] != prev_layer_num:
                C = torch.matmul(sim.cascaded_couplings[0][layer_num[zi]], sim.source_field) if sim.source_direction == 'forward' else torch.matmul(sim.cascaded_couplings[1][layer_num[zi]], sim.source_field)

                kz_norm = sim.kz_eigenvalues[layer_num[zi]]
                E_vec = sim.e_mode_vectors[layer_num[zi]]
                H_vec = sim.h_mode_vectors[layer_num[zi]]

                Cp = torch.diag(C[:2 * sim.mode_count, 0])
                Cm = torch.diag(C[2 * sim.mode_count:, 0])

                eps_inv = torch.linalg.inv(sim.permittivity_conv[layer_num[zi]])
                mu_inv = torch.linalg.inv(sim.permeability_conv[layer_num[zi]])

            z_phase_p = torch.diag(torch.exp(1.j * sim.angular_freq * kz_norm * z_prop))
            z_phase_m = torch.diag(torch.exp(1.j * sim.angular_freq * kz_norm * (sim.layer_thicknesses[layer_num[zi]] - z_prop)))

            Exy_p = torch.matmul(E_vec, z_phase_p)
            Ex_p = Exy_p[:sim.mode_count, :]
            Ey_p = Exy_p[sim.mode_count:, :]
            Hz_p = torch.matmul(mu_inv, torch.matmul(Kx_norm, Ey_p)) - torch.matmul(mu_inv, torch.matmul(Ky_norm, Ex_p))
            Exy_m = torch.matmul(E_vec, z_phase_m)
            Ex_m = Exy_m[:sim.mode_count, :]
            Ey_m = Exy_m[sim.mode_count:, :]
            Hz_m = torch.matmul(mu_inv, torch.matmul(Kx_norm, Ey_m)) - torch.matmul(mu_inv, torch.matmul(Ky_norm, Ex_m))

            Hxy_p = torch.matmul(H_vec, z_phase_p)
            Hx_p = Hxy_p[:sim.mode_count, :]
            Hy_p = Hxy_p[sim.mode_count:, :]
            Ez_p = torch.matmul(eps_inv, torch.matmul(Ky_norm, Hx_p)) - torch.matmul(eps_inv, torch.matmul(Kx_norm, Hy_p))
            Hxy_m = torch.matmul(-H_vec, z_phase_m)
            Hx_m = Hxy_m[:sim.mode_count, :]
            Hy_m = Hxy_m[sim.mode_count:, :]
            Ez_m = torch.matmul(eps_inv, torch.matmul(Ky_norm, Hx_m)) - torch.matmul(eps_inv, torch.matmul(Kx_norm, Hy_m))

            Ex_mn = torch.sum(torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1)
            Ey_mn = torch.sum(torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1)
            Ez_mn = torch.sum(torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1)
            Hx_mn = torch.sum(torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1)
            Hy_mn = torch.sum(torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1)
            Hz_mn = torch.sum(torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1)

            yz_phase = torch.exp(1.j * sim.angular_freq * (sim.kx_vector * x + sim.ky_vector * y_axis))
            Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * yz_phase, dim=2))
            Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * yz_phase, dim=2))

        prev_layer_num = layer_num[zi]

    Ex = torch.cat(Ex_split, dim=1)
    Ey = torch.cat(Ey_split, dim=1)
    Ez = torch.cat(Ez_split, dim=1)
    Hx = torch.cat(Hx_split, dim=1)
    Hy = torch.cat(Hy_split, dim=1)
    Hz = torch.cat(Hz_split, dim=1)
    return [Ex, Ey, Ez], [Hx, Hy, Hz]


def field_xy_slice(sim, layer_num, x_axis, y_axis, z_prop=0.):
    if not isinstance(layer_num, int):
        warnings.warn('Parameter \"layer_num\" must be int type. Return None.', UserWarning)
        return None

    if layer_num < -1 or layer_num > sim.stack_depth:
        warnings.warn('Layer number is out of range. Return None.', UserWarning)
        return None

    if not isinstance(x_axis, torch.Tensor) or not isinstance(y_axis, torch.Tensor):
        warnings.warn('x and y axis must be torch.Tensor type. Return None.', UserWarning)
        return None

    x_axis = x_axis.reshape([-1, 1, 1])
    y_axis = y_axis.reshape([-1, 1, 1])
    Kx_norm, Ky_norm = sim.kx_diag, sim.ky_diag

    if layer_num == -1:
        eps = sim.permittivity_input
        mu = sim.permeability_input
        z_prop = z_prop if z_prop <= 0 else 0.
        Kz_norm_dn = torch.sqrt(eps * mu - sim.kx_vector ** 2 - sim.ky_vector ** 2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn) > 0, torch.conj(Kz_norm_dn), Kz_norm_dn).reshape([-1, 1])
        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

        z_phase = torch.exp(1.j * sim.angular_freq * Kz_norm_dn * z_prop)
        if sim.source_direction == 'forward':
            Exy_p = sim.source_field * z_phase
            Hxy_p = torch.matmul(sim.input_transform, Exy_p)
            Exy_m = torch.matmul(sim.global_scattering[1], sim.source_field) * torch.conj(z_phase)
            Hxy_m = torch.matmul(-sim.input_transform, Exy_m)
        else:
            Exy_p = torch.zeros_like(sim.source_field)
            Hxy_p = torch.zeros_like(sim.source_field)
            Exy_m = torch.matmul(sim.global_scattering[3], sim.source_field) * torch.conj(z_phase)
            Hxy_m = torch.matmul(-sim.input_transform, Exy_m)

        Ex_mn = Exy_p[:sim.mode_count] + Exy_m[:sim.mode_count]
        Ey_mn = Exy_p[sim.mode_count:] + Exy_m[sim.mode_count:]
        Hz_mn = torch.matmul(Kx_norm, Ey_mn) / mu - torch.matmul(Ky_norm, Ex_mn) / mu
        Hx_mn = Hxy_p[:sim.mode_count] + Hxy_m[:sim.mode_count]
        Hy_mn = Hxy_p[sim.mode_count:] + Hxy_m[sim.mode_count:]
        Ez_mn = torch.matmul(Ky_norm, Hx_mn) / eps - torch.matmul(Kx_norm, Hy_mn) / eps

    elif layer_num == sim.stack_depth:
        eps = sim.permittivity_output
        mu = sim.permeability_output
        z_prop = z_prop if z_prop >= 0. else 0.
        Kz_norm_dn = torch.sqrt(eps * mu - sim.kx_vector ** 2 - sim.ky_vector ** 2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn).reshape([-1, 1])
        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

        z_phase = torch.exp(1.j * sim.angular_freq * Kz_norm_dn * z_prop)
        if sim.source_direction == 'forward':
            Exy_p = torch.matmul(sim.global_scattering[0], sim.source_field) * z_phase
            Hxy_p = torch.matmul(sim.output_transform, Exy_p)
            Exy_m = torch.zeros_like(sim.source_field)
            Hxy_m = torch.zeros_like(sim.source_field)
        else:
            Exy_p = torch.matmul(sim.global_scattering[2], sim.source_field) * z_phase
            Hxy_p = torch.matmul(sim.output_transform, Exy_p)
            Exy_m = sim.source_field * torch.conj(z_phase)
            Hxy_m = torch.matmul(-sim.output_transform, Exy_m)

        Ex_mn = Exy_p[:sim.mode_count] + Exy_m[:sim.mode_count]
        Ey_mn = Exy_p[sim.mode_count:] + Exy_m[sim.mode_count:]
        Hz_mn = torch.matmul(Kx_norm, Ey_mn) / mu - torch.matmul(Ky_norm, Ex_mn) / mu
        Hx_mn = Hxy_p[:sim.mode_count] + Hxy_m[:sim.mode_count]
        Hy_mn = Hxy_p[sim.mode_count:] + Hxy_m[sim.mode_count:]
        Ez_mn = torch.matmul(Ky_norm, Hx_mn) / eps - torch.matmul(Kx_norm, Hy_mn) / eps

    else:
        if z_prop < 0 or z_prop > sim.layer_thicknesses[layer_num]:
            warnings.warn('z_prop is out of range. Return None.', UserWarning)
            return None

        kz_norm = sim.kz_eigenvalues[layer_num]
        E_vec = sim.e_mode_vectors[layer_num]
        H_vec = sim.h_mode_vectors[layer_num]

        C_mat = torch.matmul(sim.cascaded_couplings[0][layer_num], sim.source_field) if sim.source_direction == 'forward' else torch.matmul(sim.cascaded_couplings[1][layer_num], sim.source_field)
        Cp = torch.diag(C_mat[:2 * sim.mode_count, 0])
        Cm = torch.diag(C_mat[2 * sim.mode_count:, 0])

        eps_inv = torch.linalg.inv(sim.permittivity_conv[layer_num])
        mu_inv = torch.linalg.inv(sim.permeability_conv[layer_num])

        z_phase_p = torch.diag(torch.exp(1.j * sim.angular_freq * kz_norm * z_prop))
        z_phase_m = torch.diag(torch.exp(1.j * sim.angular_freq * kz_norm * (sim.layer_thicknesses[layer_num] - z_prop)))

        Exy_p = torch.matmul(E_vec, z_phase_p)
        Ex_p = Exy_p[:sim.mode_count, :]
        Ey_p = Exy_p[sim.mode_count:, :]
        Hz_p = torch.matmul(mu_inv, torch.matmul(Kx_norm, Ey_p)) - torch.matmul(mu_inv, torch.matmul(Ky_norm, Ex_p))
        Exy_m = torch.matmul(E_vec, z_phase_m)
        Ex_m = Exy_m[:sim.mode_count, :]
        Ey_m = Exy_m[sim.mode_count:, :]
        Hz_m = torch.matmul(mu_inv, torch.matmul(Kx_norm, Ey_m)) - torch.matmul(mu_inv, torch.matmul(Ky_norm, Ex_m))

        Hxy_p = torch.matmul(H_vec, z_phase_p)
        Hx_p = Hxy_p[:sim.mode_count, :]
        Hy_p = Hxy_p[sim.mode_count:, :]
        Ez_p = torch.matmul(eps_inv, torch.matmul(Ky_norm, Hx_p)) - torch.matmul(eps_inv, torch.matmul(Kx_norm, Hy_p))
        Hxy_m = torch.matmul(-H_vec, z_phase_m)
        Hx_m = Hxy_m[:sim.mode_count, :]
        Hy_m = Hxy_m[sim.mode_count:, :]
        Ez_m = torch.matmul(eps_inv, torch.matmul(Ky_norm, Hx_m)) - torch.matmul(eps_inv, torch.matmul(Kx_norm, Hy_m))

        Ex_mn = torch.sum(torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1)
        Ey_mn = torch.sum(torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1)
        Ez_mn = torch.sum(torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1)
        Hx_mn = torch.sum(torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1)
        Hy_mn = torch.sum(torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1)
        Hz_mn = torch.sum(torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1)

    xy_phase = torch.exp(1.j * sim.angular_freq * (sim.kx_vector * x_axis + sim.ky_vector * y_axis))
    Ex = torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2)
    Ey = torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2)
    Ez = torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2)
    Hx = torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2)
    Hy = torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2)
    Hz = torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2)

    return [Ex, Ey, Ez], [Hx, Hy, Hz]


def dispatch_field_slice(sim, plane, *, x_axis=None, y_axis=None, z_axis=None, layer_index=None, plane_position=0., z_offset=0.):
    if plane == 'xz':
        if x_axis is None or z_axis is None:
            raise ValueError("xz plane requires x_axis and z_axis.")
        return field_xz_slice(sim, x_axis, z_axis, plane_position)
    if plane == 'yz':
        if y_axis is None or z_axis is None:
            raise ValueError("yz plane requires y_axis and z_axis.")
        return field_yz_slice(sim, y_axis, z_axis, plane_position)
    if plane == 'xy':
        if layer_index is None or x_axis is None or y_axis is None:
            raise ValueError("xy plane requires layer_index, x_axis, and y_axis.")
        return field_xy_slice(sim, layer_index, x_axis, y_axis, z_prop=z_offset)
    raise ValueError("plane must be one of {'xz','yz','xy'}.")
