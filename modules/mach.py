import numpy as np

def amort_s1(U0_vec_in, haut_a_ratio, coeff_a_damping, current_nb_dim): 
    U11 = U0_vec_in.copy()
    dim_py = current_nb_dim 
    if dim_py == 0: return U11

    za_start_val_for_calc = haut_a_ratio * dim_py 
    py_idx_start_damping = int(np.ceil(za_start_val_for_calc)) -1 
    if za_start_val_for_calc <= 0: py_idx_start_damping = 0 
    if py_idx_start_damping < 0: py_idx_start_damping = 0 

    if py_idx_start_damping <= dim_py - 2: 
        idex_py = np.arange(py_idx_start_damping, dim_py - 1) 
        
        if len(idex_py) > 0:
            idex_matlab_equiv = idex_py + 1 
            
            numerator_val = idex_matlab_equiv - za_start_val_for_calc
            dim_matlab_equiv = dim_py 
            denominator_val = coeff_a_damping * (dim_matlab_equiv - idex_matlab_equiv + 1e-12) 
            
            safe_denom_mask = np.abs(denominator_val) > 1e-12
            exponent = np.zeros_like(numerator_val, dtype=float)
            if np.any(safe_denom_mask): # Ensure not all are zero before division
                 exponent[safe_denom_mask] = - (numerator_val[safe_denom_mask] / denominator_val[safe_denom_mask])**2
            
            U11[idex_py] = U0_vec_in[idex_py] * np.exp(exponent).reshape(-1, 1) 

    if dim_py > 0: U11[dim_py-1, 0] = 0.0 + 0.0j
    return U11

def amort_y_s1(U0_y_vec_in, haut_a_ratio_y, coeff_a_damping_y, current_nb_y): 
    if current_nb_y == 0: return U0_y_vec_in.copy()
    
    U_damped = U0_y_vec_in.copy()
    
    damping_profile_high_end = np.ones(current_nb_y, dtype=float)
    za_high = haut_a_ratio_y * current_nb_y 
    
    py_idx_start_damping_high = int(np.ceil(za_high)) -1
    if za_high <=0: py_idx_start_damping_high = 0
    if py_idx_start_damping_high < 0: py_idx_start_damping_high = 0

    if py_idx_start_damping_high <= current_nb_y - 2:
        idex_py_high = np.arange(py_idx_start_damping_high, current_nb_y - 1)
        if len(idex_py_high) > 0:
            idex_matlab_equiv_high = idex_py_high + 1
            numerator_high = idex_matlab_equiv_high - za_high
            denominator_high = coeff_a_damping_y * (current_nb_y - idex_matlab_equiv_high + 1e-12)
            
            safe_denom_mask_high = np.abs(denominator_high) > 1e-12
            exponent_high = np.zeros_like(numerator_high, dtype=float)
            if np.any(safe_denom_mask_high):
                exponent_high[safe_denom_mask_high] = - (numerator_high[safe_denom_mask_high] / denominator_high[safe_denom_mask_high])**2
            damping_profile_high_end[idex_py_high] = np.exp(exponent_high)
            
    if current_nb_y > 0: damping_profile_high_end[current_nb_y-1] = 0.0

    damping_profile_low_end = np.ones(current_nb_y, dtype=float)
    za_low_equivalent_start_point_from_min = (1.0 - haut_a_ratio_y) * current_nb_y
    
    py_idx_end_damping_low = int(np.floor(za_low_equivalent_start_point_from_min)) 
    if za_low_equivalent_start_point_from_min >= current_nb_y-1: py_idx_end_damping_low = current_nb_y -2 
    if py_idx_end_damping_low < 0 : py_idx_end_damping_low = -1 

    if py_idx_end_damping_low >= 1: 
        idex_py_low = np.arange(1, py_idx_end_damping_low + 1) 
        if len(idex_py_low) > 0:
            idex_matlab_equiv_low_mirrored = (current_nb_y - 1 - idex_py_low) + 1 
            
            numerator_low = idex_matlab_equiv_low_mirrored - za_high 
            denominator_low = coeff_a_damping_y * (current_nb_y - idex_matlab_equiv_low_mirrored + 1e-12)

            safe_denom_mask_low = np.abs(denominator_low) > 1e-12
            exponent_low = np.zeros_like(numerator_low, dtype=float)
            if np.any(safe_denom_mask_low):
                exponent_low[safe_denom_mask_low] = - (numerator_low[safe_denom_mask_low] / denominator_low[safe_denom_mask_low])**2
            damping_profile_low_end[idex_py_low] = np.exp(exponent_low)

    if current_nb_y > 0: damping_profile_low_end[0] = 0.0
    
    final_symmetric_profile = np.minimum(damping_profile_high_end, damping_profile_low_end)
    U_damped = U0_y_vec_in * final_symmetric_profile.reshape(-1,1)
    return U_damped


def Mach(v_ref_windspeed,z_ref_height,z_coords_vec,z0_roughness,d_displacement,shear_exp_profile,
         c0_sound_speed,Tlog_coeff,temp0_kelvin,Rg_gas_const,gam_heat_ratio,theta_angle_deg): 
    
    V_x_wind_profile=(v_ref_windspeed*(np.maximum(z_coords_vec,1e-6)/np.maximum(z_ref_height,1e-6))**shear_exp_profile)*np.cos(np.pi*theta_angle_deg/180.0)
    
    temp1_profile_kelvin=np.full_like(z_coords_vec,temp0_kelvin,dtype=float)
    current_z0=max(z0_roughness,1e-9) 

    mask_for_log_temp= (z_coords_vec > (d_displacement-current_z0+1e-9)) 
    mask_above_d = (z_coords_vec > d_displacement)
    actual_mask_log_calc = mask_for_log_temp & mask_above_d

    if np.any(actual_mask_log_calc):
        arg_log_val=(z_coords_vec[actual_mask_log_calc]-d_displacement)/current_z0 + 1.0 
        arg_log_val[arg_log_val<=1e-9]=1e-9 
        temp1_profile_kelvin[actual_mask_log_calc]=temp0_kelvin+Tlog_coeff*np.log(arg_log_val)
        
    cel_profile=np.sqrt(gam_heat_ratio*Rg_gas_const*temp1_profile_kelvin)
    
    if len(z_coords_vec)>0:
        indice_d_py=np.argmin(np.abs(z_coords_vec-d_displacement))
        cel_profile[0:indice_d_py+1]=c0_sound_speed 
        
    epsilon_profile=(c0_sound_speed**2/(cel_profile**2+1e-12))-1 
    epsilon_derivative = np.gradient(epsilon_profile, z_coords_vec)  #!New  
    Mx_mach_number=V_x_wind_profile/(cel_profile+1e-12) 
    
    return Mx_mach_number.reshape(-1,1), epsilon_profile.reshape(-1,1), epsilon_derivative.reshape(-1,1)  

