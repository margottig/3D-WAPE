import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
from .mach import amort_s1, amort_y_s1

def sourceSalomons_order2_1D(z_coords, hS, k0, Rp): 
    z = np.asarray(z_coords).ravel()       # If z_coords could be a 2D array (e.g., from a meshgrid) or other dimensions, .ravel() converts to 1D format.
    # direct wave
    k02z2_direct = k0**2 * (z - hS)**2
    U01 = np.sqrt(k0 * 1j + 1e-18j) * (1.3717 - 0.3701 * k02z2_direct) * np.exp(-k02z2_direct / 3.0)
    if Rp == 0.0 or Rp is None: 
        U0 = U01
    # reflected wave
    else: 
        k02z2_reflected = k0**2 * (z + hS)**2 
        U02 = np.sqrt(k0 * 1j + 1e-18j) * (1.3717 - 0.3701 * k02z2_reflected) * np.exp(-k02z2_reflected / 3.0)
        U0 = U01 + Rp * U02
    return U0.reshape(-1, 1)


def fast_turb_3D(pressure_level_field_3d, x_coords_interp_vec, y_coords_interp_vec, z_coords_interp_vec,
                 hS_source, yS_source, freq_source, gamT_turbulence): 
    xs = 0.0 
    Z_mesh, Y_mesh, X_mesh = np.meshgrid(z_coords_interp_vec, y_coords_interp_vec, x_coords_interp_vec, indexing='ij')
    d_distance_3d = np.sqrt((X_mesh - xs)**2 + (Y_mesh - yS_source)**2 + (Z_mesh - hS_source)**2 + 1e-12)

    log_gamT = -np.inf; log_freq_term = -np.inf
    if gamT_turbulence > 1e-9: log_gamT = np.log10(gamT_turbulence)
    if freq_source > 1e-9 : log_freq_term = np.log10(freq_source / 1000.0)
    
    log_d_term = np.full_like(d_distance_3d, -np.inf, dtype=float)
    if d_distance_3d.size > 0:
        mask_d_pos = d_distance_3d > 1e-9
        log_d_term[mask_d_pos] = np.log10(d_distance_3d[mask_d_pos] / 100.0)
        
    Turb_effect = 25 + 10*log_gamT + 3*log_freq_term + 10*log_d_term
    
    term1 = 10**(pressure_level_field_3d/10.0) 
    term2 = 10**(Turb_effect/10.0)             
    sum_terms = term1 + term2
    
    ppdBiTurb_corrected_3d = np.full_like(sum_terms, -np.inf, dtype=float) 
    if sum_terms.size > 0:
        mask_sum_pos = sum_terms > 1e-12 
        ppdBiTurb_corrected_3d[mask_sum_pos] = 10*np.log10(sum_terms[mask_sum_pos])
        
    return ppdBiTurb_corrected_3d

# --- Main PE Solver and Coefficient Calculation  ---
def _build_system_matrices_s1(nb_pts_dim, delta_dim, k0, delta_x,
                              gamma_x_prof_val, tau_bar1_prof_val, 
                              epsilon0_prof_val, epsilon1_prof_val,
                              Beta_val, is_ground_boundary):
    eps_denom = 1e-12 

    common_term_A = (2 - 1j*k0*delta_x*(2*gamma_x_prof_val**2 - tau_bar1_prof_val)) / \
                    (8*k0**2 * gamma_x_prof_val**2 * delta_dim**2 + eps_denom)
    
    a_A_coeff = common_term_A 
    
    b_A_coeff_base = (1 + epsilon1_prof_val/4 - 1j*k0*delta_x/2 * (gamma_x_prof_val**2 * epsilon1_prof_val/2 - (1 + epsilon1_prof_val/4) * tau_bar1_prof_val) -
                      common_term_A) 
                                     
    b_A_coeff = b_A_coeff_base - common_term_A 
    c_A_coeff = common_term_A 
    c_A_coeff_for_diag = c_A_coeff 

    common_term_B = (2 + 1j*k0*delta_x*(2*gamma_x_prof_val**2 - tau_bar1_prof_val)) / \
                    (8*k0**2 * gamma_x_prof_val**2 * delta_dim**2 + eps_denom)
    d_B_coeff = common_term_B 

    e_B_coeff_base = (1 + epsilon0_prof_val/4 + 1j*k0*delta_x/2 *
                      (gamma_x_prof_val**2 * epsilon0_prof_val/2 - (1 + epsilon0_prof_val/4) * tau_bar1_prof_val) -
                      common_term_B)
    e_B_coeff = e_B_coeff_base - common_term_B 
    f_B_coeff = common_term_B 
    f_B_coeff_for_diag = f_B_coeff

    if isinstance(gamma_x_prof_val, np.ndarray) and gamma_x_prof_val.ndim > 0 : 
        a_diag_A = a_A_coeff.copy()
        b_diag_A = b_A_coeff.copy()
        c_diag_A = c_A_coeff.copy()
        c_diag_A_boundary = c_A_coeff_for_diag.copy()

        d_diag_B = d_B_coeff.copy()
        e_diag_B = e_B_coeff.copy()
        f_diag_B = f_B_coeff.copy()
        f_diag_B_boundary = f_B_coeff_for_diag.copy()
    else: 
        a_diag_A = np.full(nb_pts_dim, a_A_coeff, dtype=complex)
        b_diag_A = np.full(nb_pts_dim, b_A_coeff, dtype=complex)
        c_diag_A = np.full(nb_pts_dim, c_A_coeff, dtype=complex)
        c_diag_A_boundary = np.full(nb_pts_dim, c_A_coeff_for_diag, dtype=complex) 

        d_diag_B = np.full(nb_pts_dim, d_B_coeff, dtype=complex)
        e_diag_B = np.full(nb_pts_dim, e_B_coeff, dtype=complex)
        f_diag_B = np.full(nb_pts_dim, f_B_coeff, dtype=complex)
        f_diag_B_boundary = np.full(nb_pts_dim, f_B_coeff_for_diag, dtype=complex)


    if is_ground_boundary and Beta_val is not None and nb_pts_dim > 0:
        beta_complex = complex(Beta_val)
        
        b_diag_A[0] += a_diag_A[0] * (2j * delta_dim * k0 * beta_complex) 
        if nb_pts_dim > 1: c_diag_A_boundary[0] = 2.0 * c_diag_A[0]      
        
        e_diag_B[0] += d_diag_B[0] * (2j * delta_dim * k0 * beta_complex)
        if nb_pts_dim > 1: f_diag_B_boundary[0] = 2.0 * f_diag_B[0]
    
    Amat = Bmat = scipy.sparse.csc_matrix((0,0), dtype=complex)
    if nb_pts_dim == 1:
        if len(b_diag_A)>0: Amat = scipy.sparse.csc_matrix([[b_diag_A.ravel()[0]]], dtype=complex) 
        if len(e_diag_B)>0: Bmat = scipy.sparse.csc_matrix([[e_diag_B.ravel()[0]]], dtype=complex) 
    elif nb_pts_dim > 1 : 
        Amat = scipy.sparse.diags(
            [a_diag_A[1:].ravel(), b_diag_A.ravel(), c_diag_A_boundary[:-1].ravel()], 
            [-1, 0, 1], 
            shape=(nb_pts_dim, nb_pts_dim), format="csc", dtype=complex
        )
        Bmat = scipy.sparse.diags(
            [d_diag_B[1:].ravel(), e_diag_B.ravel(), f_diag_B_boundary[:-1].ravel()], 
            [-1, 0, 1], 
            shape=(nb_pts_dim, nb_pts_dim), format="csc", dtype=complex
        )
    return Amat, Bmat


def main_calc_3D_ADI_s1(U0_2D_initial, gamma_x_profile, tau_bar1_profile, k0_wavenum, 
                        delta_x_step, delta_z_step, delta_y_step, 
                        nb_x_pts, nb_z_pts, nb_y_pts,
                        epsilon0_profile, epsilon1_profile, Beta_eff_admittance,
                        abs_layer_ratio_z, damping_coeff_z, 
                        abs_layer_ratio_y, damping_coeff_y): 
    Phi_potential_3D = np.zeros((nb_z_pts, nb_y_pts, nb_x_pts), dtype=complex)
    if nb_z_pts == 0 or nb_y_pts == 0 or nb_x_pts == 0: return Phi_potential_3D

    U_current_xy_slice = U0_2D_initial.astype(complex)
    if U_current_xy_slice.shape != (nb_z_pts, nb_y_pts):
        print(f"Error: U0_2D_initial shape {U0_2D_initial.shape} does not match (nb_z_pts, nb_y_pts) = ({nb_z_pts}, {nb_y_pts})")
        # Attempt to resize/pad U0 if makes sense, or raise error
        if nb_x_pts > 0:
            try:
                Phi_potential_3D[:, :, 0] = U_current_xy_slice # This might fail if shapes are incompatible
            except ValueError as e:
                print(f"ValueError during Phi_potential_3D[:, :, 0] assignment: {e}. Exiting.")
                return Phi_potential_3D # Or raise error
        else: # if nb_x_pts is 0, this is fine
            Phi_potential_3D[:, :, 0] = U_current_xy_slice

    else:
        Phi_potential_3D[:, :, 0] = U_current_xy_slice


    Amat_z, Bmat_z = _build_system_matrices_s1(nb_z_pts, delta_z_step, k0_wavenum, delta_x_step,
                                               gamma_x_profile, tau_bar1_profile,
                                               epsilon0_profile, epsilon1_profile,
                                               Beta_eff_admittance, is_ground_boundary=True)

    U_star_xy_slice = np.zeros_like(U_current_xy_slice, dtype=complex)

    for ix_main in range(1, nb_x_pts): 
        for iy_col in range(nb_y_pts): 
            U_z_vector_old_col = U_current_xy_slice[:, iy_col].reshape(-1,1) 
            if nb_z_pts > 0 and Amat_z.shape[0] > 0 and Bmat_z.shape[0] > 0: # Added Bmat_z check
                if Amat_z.shape[0] == U_z_vector_old_col.shape[0]: # Check dimension compatibility
                    U_z_interim_rhs = Bmat_z.dot(U_z_vector_old_col)
                    U_z_vector_star_col = scipy.sparse.linalg.spsolve(Amat_z, U_z_interim_rhs)
                    U_z_vector_star_col = U_z_vector_star_col.reshape(-1,1) 
                    U_z_vector_star_col = amort_s1(U_z_vector_star_col, abs_layer_ratio_z, damping_coeff_z, nb_z_pts)
                    U_star_xy_slice[:, iy_col] = U_z_vector_star_col.flatten()
                else: # Dimension mismatch
                    print(f"Warning Z-sweep: Dim mismatch Amat_z {Amat_z.shape} vs U_z_vec {U_z_vector_old_col.shape} at ix={ix_main}, iy={iy_col}")
                    U_star_xy_slice[:, iy_col] = U_z_vector_old_col.flatten() # Fallback
            elif nb_z_pts > 0 : 
                 U_star_xy_slice[:, iy_col] = U_z_vector_old_col.flatten()


        U_new_xy_slice = np.zeros_like(U_star_xy_slice, dtype=complex)
        for iz_row in range(nb_z_pts): 
            gamma_x_s   = gamma_x_profile[iz_row, 0] if gamma_x_profile.ndim > 1 and iz_row < gamma_x_profile.shape[0] else gamma_x_profile[iz_row] if iz_row < len(gamma_x_profile) else 1.0
            tau_bar1_s  = tau_bar1_profile[iz_row, 0] if tau_bar1_profile.ndim > 1 and iz_row < tau_bar1_profile.shape[0] else tau_bar1_profile[iz_row] if iz_row < len(tau_bar1_profile) else 0.0
            epsilon0_s  = epsilon0_profile[iz_row, 0] if epsilon0_profile.ndim > 1 and iz_row < epsilon0_profile.shape[0] else epsilon0_profile[iz_row] if iz_row < len(epsilon0_profile) else 0.0
            epsilon1_s  = epsilon1_profile[iz_row, 0] if epsilon1_profile.ndim > 1 and iz_row < epsilon1_profile.shape[0] else epsilon1_profile[iz_row] if iz_row < len(epsilon1_profile) else 0.0
            
            Amat_y_iz, Bmat_y_iz = _build_system_matrices_s1(
                nb_y_pts, delta_y_step, k0_wavenum, delta_x_step,
                gamma_x_s, tau_bar1_s, 
                epsilon0_s, epsilon1_s,
                Beta_val=None, is_ground_boundary=False 
            )
            
            U_y_vector_star_row = U_star_xy_slice[iz_row, :].reshape(-1,1) 
            if nb_y_pts > 0 and Amat_y_iz.shape[0] > 0 and Bmat_y_iz.shape[0] > 0: # Added Bmat_y_iz check
                if Amat_y_iz.shape[0] == U_y_vector_star_row.shape[0]: # Check dimension compatibility
                    U_y_interim_rhs = Bmat_y_iz.dot(U_y_vector_star_row)
                    U_y_vector_new_col = scipy.sparse.linalg.spsolve(Amat_y_iz, U_y_interim_rhs)
                    U_y_vector_new_col = U_y_vector_new_col.reshape(-1,1) 
                    U_y_vector_new_col = amort_y_s1(U_y_vector_new_col, abs_layer_ratio_y, damping_coeff_y, nb_y_pts)
                    U_new_xy_slice[iz_row, :] = U_y_vector_new_col.flatten()
                else: # Dimension mismatch
                    print(f"Warning Y-sweep: Dim mismatch Amat_y {Amat_y_iz.shape} vs U_y_vec {U_y_vector_star_row.shape} at ix={ix_main}, iz={iz_row}")
                    U_new_xy_slice[iz_row, :] = U_y_vector_star_row.flatten() # Fallback
            elif nb_y_pts > 0 :
                U_new_xy_slice[iz_row, :] = U_y_vector_star_row.flatten()


        Phi_potential_3D[:, :, ix_main] = U_new_xy_slice
        U_current_xy_slice = U_new_xy_slice 
        
    return Phi_potential_3D


def interpolation_3D(field_to_interp, stock_x_step, stock_y_step, stock_z_step, 
                     x_orig_coords, y_orig_coords, z_orig_coords):
    z_orig_1d = np.asarray(z_orig_coords).ravel(); 
    x_orig_1d = np.asarray(x_orig_coords).ravel()
    y_orig_1d = np.asarray(y_orig_coords).ravel()

    if len(x_orig_1d) == 0 or len(y_orig_1d) == 0 or len(z_orig_1d) == 0 or field_to_interp.size == 0:
        dummy_x = np.array([0.0]) if len(x_orig_1d)==0 else x_orig_1d
        dummy_y = np.array([0.0]) if len(y_orig_1d)==0 else y_orig_1d
        dummy_z = np.array([0.0]) if len(z_orig_1d)==0 else z_orig_1d
        
        xi_new = np.atleast_1d(np.arange(dummy_x[0], dummy_x[0] + stock_x_step, stock_x_step) if len(dummy_x)>0 and stock_x_step > 0 else np.array([dummy_x[0]]))
        yi_new = np.atleast_1d(np.arange(dummy_y[0], dummy_y[0] + stock_y_step, stock_y_step) if len(dummy_y)>0 and stock_y_step > 0 else np.array([dummy_y[0]]))
        zi_new = np.atleast_1d(np.arange(dummy_z[0], dummy_z[0] + stock_z_step, stock_z_step) if len(dummy_z)>0 and stock_z_step > 0 else np.array([dummy_z[0]]))

        nan_val = np.nan if not np.iscomplexobj(field_to_interp) else np.complex128(np.nan + 1j*np.nan)
        return np.full((len(zi_new), len(yi_new), len(xi_new)), nan_val), xi_new, yi_new, zi_new

    sort_x_idx = np.argsort(x_orig_1d); x_orig_1d_sorted = x_orig_1d[sort_x_idx]
    sort_y_idx = np.argsort(y_orig_1d); y_orig_1d_sorted = y_orig_1d[sort_y_idx]
    sort_z_idx = np.argsort(z_orig_1d); z_orig_1d_sorted = z_orig_1d[sort_z_idx]
    
    field_sorted = field_to_interp.copy() 
    if field_sorted.ndim == 3: 
        field_sorted = field_sorted[sort_z_idx, :, :]
        field_sorted = field_sorted[:, sort_y_idx, :]
        field_sorted = field_sorted[:, :, sort_x_idx]

    min_len_coord = 2 
    is_flat_x = len(x_orig_1d_sorted) < min_len_coord
    is_flat_y = len(y_orig_1d_sorted) < min_len_coord
    is_flat_z = len(z_orig_1d_sorted) < min_len_coord
    
    sx = max(stock_x_step, 1e-6); sy = max(stock_y_step, 1e-6); sz = max(stock_z_step, 1e-6)
    
    xi_new = np.array([x_orig_1d_sorted[0]]) if is_flat_x or len(x_orig_1d_sorted) == 0 else np.unique(np.arange(x_orig_1d_sorted[0], x_orig_1d_sorted[-1] + sx/2.0, sx))
    yi_new = np.array([y_orig_1d_sorted[0]]) if is_flat_y or len(y_orig_1d_sorted) == 0 else np.unique(np.arange(y_orig_1d_sorted[0], y_orig_1d_sorted[-1] + sy/2.0, sy))
    zi_new = np.array([z_orig_1d_sorted[0]]) if is_flat_z or len(z_orig_1d_sorted) == 0 else np.unique(np.arange(z_orig_1d_sorted[0], z_orig_1d_sorted[-1] + sz/2.0, sz))


    if not (is_flat_x or len(x_orig_1d_sorted)==0) and not np.isclose(xi_new[-1], x_orig_1d_sorted[-1]) and xi_new[-1] < x_orig_1d_sorted[-1]: xi_new = np.append(xi_new, x_orig_1d_sorted[-1])
    if not (is_flat_y or len(y_orig_1d_sorted)==0) and not np.isclose(yi_new[-1], y_orig_1d_sorted[-1]) and yi_new[-1] < y_orig_1d_sorted[-1]: yi_new = np.append(yi_new, y_orig_1d_sorted[-1])
    if not (is_flat_z or len(z_orig_1d_sorted)==0) and not np.isclose(zi_new[-1], z_orig_1d_sorted[-1]) and zi_new[-1] < z_orig_1d_sorted[-1]: zi_new = np.append(zi_new, z_orig_1d_sorted[-1])
    
    xi_new = np.atleast_1d(xi_new); yi_new = np.atleast_1d(yi_new); zi_new = np.atleast_1d(zi_new)
    
    expected_shape = (len(z_orig_1d_sorted), len(y_orig_1d_sorted), len(x_orig_1d_sorted))
    if field_sorted.shape != expected_shape:
        print(f"Interpolation Warning: Sorted field shape {field_sorted.shape} mismatch with sorted coord lengths. Expected: {expected_shape}. Returning NaNs.")
        nan_val = np.nan if not np.iscomplexobj(field_sorted) else np.complex128(np.nan + 1j*np.nan)
        return np.full((len(zi_new), len(yi_new), len(xi_new)), nan_val), xi_new, yi_new, zi_new

    points_for_interp = (z_orig_1d_sorted, y_orig_1d_sorted, x_orig_1d_sorted)
    ZI_eval, YI_eval, XI_eval = np.meshgrid(zi_new, yi_new, xi_new, indexing='ij')
    points_to_evaluate_final = np.array([ZI_eval.ravel(), YI_eval.ravel(), XI_eval.ravel()]).T
    interpolated_values_shape = (len(zi_new), len(yi_new), len(xi_new))

    fill_val_complex = np.complex128(np.nan + 1j*np.nan)

    if np.iscomplexobj(field_sorted):
        try:
            interp_real = scipy.interpolate.RegularGridInterpolator(points_for_interp, field_sorted.real, method='linear', bounds_error=False, fill_value=np.nan)
            interp_imag = scipy.interpolate.RegularGridInterpolator(points_for_interp, field_sorted.imag, method='linear', bounds_error=False, fill_value=np.nan)
            real_part_flat = interp_real(points_to_evaluate_final)
            imag_part_flat = interp_imag(points_to_evaluate_final)
            interpolated_values = (real_part_flat + 1j * imag_part_flat).reshape(interpolated_values_shape)
        except Exception as e: 
            print(f"Error during complex interpolation setup/evaluation: {e}")
            return np.full(interpolated_values_shape, fill_val_complex), xi_new, yi_new, zi_new
    else: 
        try:
            interpolator = scipy.interpolate.RegularGridInterpolator(points_for_interp, field_sorted, method='linear', bounds_error=False, fill_value=np.nan)
            interpolated_values_flat = interpolator(points_to_evaluate_final)
            interpolated_values = interpolated_values_flat.reshape(interpolated_values_shape)
        except Exception as e:
            print(f"Error during real interpolation setup/evaluation: {e}")
            return np.full(interpolated_values_shape, np.nan), xi_new, yi_new, zi_new
            
    return interpolated_values, xi_new, yi_new, zi_new

# defined as an inclusive variant of np.arange. Checks the step sign, handles a zero step and trims floating-point overflow past the end point.
def gen_range(a, d, b):
    if d == 0:
        if a == b: return np.array([a])
        return np.array([a]) if a <=b else np.array([])
    if (d > 0 and a > b + d*1e-9) or (d < 0 and a < b - d*1e-9): 
        return np.array([])
    arr = np.arange(a, b + d * 0.5, d) 
    if d > 0:
        arr = arr[arr <= b + 1e-9] 
    else: 
        arr = arr[arr >= b - 1e-9] 
    return arr


def Wgauss(kappa, sigmah, lc ): 
    W = ((sigmah**2)*(lc/(2*np.sqrt(np.pi))))*np.exp(-((kappa*lc)**2)/4)
    return W


def thickness(Zc, d_thick, kc): 
    if d_thick == 0:
        Zs = Zc
    else:
        val = -1j * kc * d_thick
        tanh_val = np.tanh(val)
        if np.any(np.abs(tanh_val) < 1e-12): 
            Zs = np.full_like(Zc, np.inf + 0j,dtype=complex) 
            non_zero_mask = np.abs(tanh_val) >= 1e-12
            if np.isscalar(Zc): 
                 if non_zero_mask.all() : Zs = Zc / tanh_val 
            else: 
                 Zs[non_zero_mask] = Zc[non_zero_mask] / tanh_val[non_zero_mask]
        else:
            Zs = Zc / tanh_val
    return Zs


def atmosISO(f_freq, hr_percent=None, T_celsius=None, pa_kpa=None): 
    if hr_percent is None: hr_percent = 50.0
    if T_celsius is None: T_celsius = 20.0
    if pa_kpa is None: pa_kpa = 101.325 
    
    pr = 101.325 
    To_ref = 293.15 
    To1 = 273.16 
    
    T_kelvin = T_celsius + 273.15
    
    C_val = -6.8346 * (To1 / T_kelvin)**1.261 + 4.6151
    h_molar_conc = hr_percent * (10**C_val) * (pr / pa_kpa) 
    
    fro = (pa_kpa / pr) * (24 + 4.04e4 * h_molar_conc * (0.02 + h_molar_conc) / (0.391 + h_molar_conc))
    frn = (pa_kpa / pr) * (T_kelvin / To_ref)**(-0.5) * \
          (9 + 280 * h_molar_conc * np.exp(-4.170 * ((T_kelvin / To_ref)**(-1/3) - 1)))
          
    f_freq_arr = np.asarray(f_freq)
    alpha_val = 8.686 * f_freq_arr**2 * (
        (1.84e-11 * (pr / pa_kpa) * (T_kelvin / To_ref)**(0.5)) + 
        (T_kelvin / To_ref)**(-2.5) * (
            0.01275 * (np.exp(-2239.1 / T_kelvin)) * (fro + (f_freq_arr**2 / (fro + 1e-12)))**(-1) + # Added epsilon to denominator
            0.1068 * (np.exp(-3352.0 / T_kelvin)) * (frn + (f_freq_arr**2 / (frn + 1e-12)))**(-1)  # Added epsilon to denominator
        )
    )
    return alpha_val


def SetAtmos3D(pressure_level_field_3d, x_coords_vec, y_coords_vec, z_coords_vec,
               hS_source, yS_source, freq_source, hr_humidity, T_temp_celsius, pa_kpa_val): 
    xs = 0.0 

    Z_mesh, Y_mesh, X_mesh = np.meshgrid(z_coords_vec, y_coords_vec, x_coords_vec, indexing='ij')
    
    Ri = np.sqrt((X_mesh - xs)**2 + (Y_mesh - yS_source)**2 + (Z_mesh - hS_source)**2 + 1e-12) 
                
    alpha = atmosISO(freq_source, hr_humidity*100, T_temp_celsius, pa_kpa_val) 
    ppdBiAtmos3D = pressure_level_field_3d - alpha * Ri
    return ppdBiAtmos3D


def roughness(lc, sigmah, k0, incidence): 
    kappa = k0 * np.sin(incidence); s1 = 1; s2 = -1
    if lc == 0 or sigmah == 0: beta_rough = 0.0 + 0.0j
    else:
        limit_alpha = np.sqrt(k0 + 1e-12); d_int_alpha = limit_alpha / 100.0 # Added epsilon to k0 for sqrt
        if np.isclose(d_int_alpha,0) and limit_alpha < 1e-9 : u_alpha = np.array([0.0]) 
        else: u_alpha = gen_range(0, d_int_alpha, limit_alpha)
        if len(u_alpha) == 0 and limit_alpha >=0 : u_alpha = np.array([0.0]) 
        if len(u_alpha) == 1 and limit_alpha > 1e-9: u_alpha = np.array([0.0, limit_alpha]) 

        sqrt_term_alpha_arg = -u_alpha**2 + 2 * k0
        sqrt_term_alpha_arg[sqrt_term_alpha_arg < 0] = 0 
        sqrt_term_alpha = np.sqrt(sqrt_term_alpha_arg)
        
        integrande_alpha1_num = (k0**2 + s1 * kappa * (k0 - u_alpha**2))**2 * Wgauss(kappa + s1 * (k0 - u_alpha**2), sigmah, lc)
        integrande_alpha2_num = (k0**2 + s2 * kappa * (k0 - u_alpha**2))**2 * Wgauss(kappa + s2 * (k0 - u_alpha**2), sigmah, lc)
        
        if np.isclose(k0, 0.0): 
            alpha1, alpha2 = 0.0, 0.0
        else:
            inv_denom_alpha = np.zeros_like(u_alpha, dtype=complex)
            mask_alpha_denom_safe = np.abs(k0 * sqrt_term_alpha) > 1e-12
            inv_denom_alpha[mask_alpha_denom_safe] = 1.0 / (k0 * sqrt_term_alpha[mask_alpha_denom_safe])
            
            integrande_alpha1 = np.real(inv_denom_alpha * integrande_alpha1_num)
            integrande_alpha2 = np.real(inv_denom_alpha * integrande_alpha2_num)
            
            alpha1 = scipy.integrate.trapz(integrande_alpha1, u_alpha) if len(u_alpha)>1 else 0.0
            alpha2 = scipy.integrate.trapz(integrande_alpha2, u_alpha) if len(u_alpha)>1 else 0.0
        alpha_val = alpha1 + alpha2

        limit_beta = 6.0 / (lc + 1e-12); d_int_beta = limit_beta / 100.0 # Added epsilon to lc
        if np.isclose(d_int_beta,0) and limit_beta < 1e-9: u_beta = np.array([0.0])
        else: u_beta = gen_range(0, d_int_beta, limit_beta)
        if len(u_beta) == 0 and limit_beta >=0 : u_beta = np.array([0.0])
        if len(u_beta) == 1 and limit_beta > 1e-9: u_beta = np.array([0.0, limit_beta])

        sqrt_term_beta_arg = k0**2 + u_beta**2 
        sqrt_term_beta = np.sqrt(sqrt_term_beta_arg) 
        
        integrande_beta1_num = (k0**2 + s1 * kappa * sqrt_term_beta)**2 * Wgauss(kappa + s1 * sqrt_term_beta, sigmah, lc)
        integrande_beta2_num = (k0**2 + s2 * kappa * sqrt_term_beta)**2 * Wgauss(kappa + s2 * sqrt_term_beta, sigmah, lc)

        if np.isclose(k0, 0.0) and np.all(np.isclose(u_beta,0.0)): 
            beta1, beta2 = 0.0, 0.0
        elif np.isclose(k0,0.0) and len(u_beta)>0 and np.all(np.abs(k0*sqrt_term_beta)<1e-12): 
             beta1, beta2 = 0.0,0.0
        else:
            inv_denom_beta = np.zeros_like(u_beta, dtype=complex)
            mask_beta_denom_safe = np.abs(k0 * sqrt_term_beta) > 1e-12
            inv_denom_beta[mask_beta_denom_safe] = 1.0 / (k0 * sqrt_term_beta[mask_beta_denom_safe])

            integrande_beta1 = np.real(inv_denom_beta * integrande_beta1_num)
            integrande_beta2 = np.real(inv_denom_beta * integrande_beta2_num)
            
            beta1 = -scipy.integrate.trapz(integrande_beta1, u_beta) if len(u_beta)>1 else 0.0
            beta2 = -scipy.integrate.trapz(integrande_beta2, u_beta) if len(u_beta)>1 else 0.0
        beta_val = beta1 + beta2
        beta_rough = alpha_val + 1j * beta_val
    return beta_rough


def Miki(freq_val, k0_val, sigma_resistivity, h_thickness): 
    freq_val = np.asarray(freq_val); k0_val = np.asarray(k0_val) 
    if sigma_resistivity >= 100000: 
        beta = np.zeros_like(freq_val, dtype=complex)
    else:
        term_sigma = freq_val / (sigma_resistivity + 1e-12) # Added epsilon
        Zc = 1 + 5.50 * (term_sigma)**(-0.632) + 1j * 8.43 * (term_sigma)**(-0.632)
        kc = k0_val * (1 + 7.81 * (term_sigma)**(-0.618) + 1j * 11.41 * (term_sigma)**(-0.618))
        
        Zs = thickness(Zc, h_thickness, kc) 
        
        beta = np.full_like(Zs, np.inf + 0j, dtype=complex) 
        non_zero_Zs_mask = np.abs(Zs) > 1e-14 
        
        if np.isscalar(Zs):
            if non_zero_Zs_mask: beta = 1.0 / Zs
        else:
            beta[non_zero_Zs_mask] = 1.0 / Zs[non_zero_Zs_mask]
            
    return beta