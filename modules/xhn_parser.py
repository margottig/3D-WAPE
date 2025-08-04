import numpy as np

TMFEx_default = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
                 100, 125, 160, 200, 250, 315, 400, 500, 630,
                 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                 5000, 6300, 8000, 10000]

dPhi_default, dTheta_default = 5, 5  # Default angular resolution for XHN

def _parse_data_block(lines, start_line_idx, num_phi_data_rows, num_theta_cols, current_data_array, freq_idx):
    for iPhi_data_row in range(num_phi_data_rows):
        line_no = start_line_idx + iPhi_data_row
        if line_no >= len(lines):
            break
        parts = lines[line_no].split(',')
        if len(parts) < num_theta_cols + 1:
            continue
        for iTheta_col in range(num_theta_cols):
            try:
                current_data_array[freq_idx, iPhi_data_row, iTheta_col] = float(parts[iTheta_col + 1])
            except ValueError:
                current_data_array[freq_idx, iPhi_data_row, iTheta_col] = np.nan
    if num_phi_data_rows > 0 and current_data_array.shape[1] > 0 and current_data_array.shape[0] > freq_idx: # Ensure last row (360 deg) is same as first (0 deg)
         current_data_array[freq_idx, -1, :] = current_data_array[freq_idx, 0, :]

def find_data_table_start(lines, search_start_idx, max_search_lines=25):
    for i in range(max_search_lines):
        current_line_idx = search_start_idx + i
        if current_line_idx >= len(lines): return -1
        line_content = lines[current_line_idx].strip()
        # Check if the line starts with a comma (or a quote not followed by degree symbol) and has many commas
        if (line_content.startswith(',') or (line_content.startswith('"') and '°"' not in line_content.split(',')[0])) and \
           line_content.count(',') > 5: # Heuristic: data tables have many columns
            # Check if the NEXT line starts with a quoted angle (like "0°")
            if current_line_idx + 1 < len(lines) and \
               lines[current_line_idx+1].strip().startswith('"') and \
               '°"' in lines[current_line_idx+1].split(',')[0]:
                return current_line_idx + 1 # Data starts on the line with the angle
    return -1

def import_xhn_complex(filename, TMFEx=TMFEx_default, dPhi=dPhi_default, dTheta=dTheta_default):
    try:
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: XHN File '{filename}' not found.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    nFrq_TMFEx = len(TMFEx)
    AnzPhi = int(360 / dPhi) + 1
    AnzTheta = int(180 / dTheta) + 1 # 0 to 180 degrees
    num_phi_data_rows = AnzPhi - 1 # Data rows for 0 to 355 (360 is copied from 0)
    num_theta_cols = AnzTheta

    magnitudes_db = np.full((nFrq_TMFEx, AnzPhi, AnzTheta), np.nan)
    phases_deg = np.full((nFrq_TMFEx, AnzPhi, AnzTheta), np.nan)
    processed_file_frequencies = []

    current_file_freq_idx_search = 0
    while current_file_freq_idx_search < len(lines):
        freq_line_idx = -1
        actual_frq_in_file = None
        # Find "Frequency" line
        for i in range(current_file_freq_idx_search, len(lines)):
            parts = lines[i].split(',')
            if len(parts) >= 2 and parts[0].strip('"') == "Frequency":
                try:
                    actual_frq_in_file = float(parts[1])
                    freq_line_idx = i
                    break
                except ValueError:
                    continue # Not a valid frequency value
        
        current_file_freq_idx_search = len(lines) if freq_line_idx == -1 else freq_line_idx + 1 # Move search start
        
        if freq_line_idx == -1: # No more "Frequency" lines found
            break
        
        # Find if this frequency is one we care about (in TMFEx)
        target_idx_in_TMFEx = -1
        for idx, tmf_frq in enumerate(TMFEx):
            if abs(actual_frq_in_file - tmf_frq) < 1e-6: # Using float comparison
                target_idx_in_TMFEx = idx
                break
        
        if target_idx_in_TMFEx == -1: # This frequency is not in our target list
            continue
        
        # Avoid reprocessing the same frequency if it appears multiple times in XHN
        if actual_frq_in_file in processed_file_frequencies:
            continue
        # Mark as processed only if we find data for it
        # processed_file_frequencies.append(actual_frq_in_file) # Moved down

        # Find start of magnitude data table
        mag_data_start_line = find_data_table_start(lines, freq_line_idx + 1)
        mag_data_found = False
        if mag_data_start_line != -1:
            _parse_data_block(lines, mag_data_start_line, num_phi_data_rows, num_theta_cols, magnitudes_db, target_idx_in_TMFEx)
            mag_data_found = True
        else:
            print(f"  Warning: Could not find magnitude data table for {actual_frq_in_file} Hz.")
            # if actual_frq_in_file in processed_file_frequencies: processed_file_frequencies.remove(actual_frq_in_file) # Rollback
            continue # Skip this frequency if no mag data

        # Find "PhaseData" line, then find start of phase data table
        search_phase_start_idx = mag_data_start_line + num_phi_data_rows if mag_data_start_line !=-1 else freq_line_idx + 10 # Heuristic start
        phase_anchor_line_idx = -1
        phase_data_found = False
        for i in range(search_phase_start_idx, len(lines)):
            if '"PhaseData"' in lines[i]: # Anchor for phase data section
                phase_anchor_line_idx = i
                break
        
        if phase_anchor_line_idx != -1:
            phase_data_start_line = find_data_table_start(lines, phase_anchor_line_idx + 1)
            if phase_data_start_line != -1:
                _parse_data_block(lines, phase_data_start_line, num_phi_data_rows, num_theta_cols, phases_deg, target_idx_in_TMFEx)
                phase_data_found = True
            else:
                print(f"  Warning: Found \"PhaseData\" but could not find phase data table for {actual_frq_in_file} Hz.")
                # if actual_frq_in_file in processed_file_frequencies: processed_file_frequencies.remove(actual_frq_in_file) # Rollback
        else: # No phase data found for this frequency
            print(f"  Warning: \"PhaseData\" keyword not found for {actual_frq_in_file} Hz.")
            # if actual_frq_in_file in processed_file_frequencies: processed_file_frequencies.remove(actual_frq_in_file) # Rollback
        
        if mag_data_found and phase_data_found:
            print(f"Successfully processed frequency {actual_frq_in_file} Hz (TMFEx index {target_idx_in_TMFEx}) found at line {freq_line_idx + 1}")
            processed_file_frequencies.append(actual_frq_in_file)
        else:
            print(f"  Incomplete data for frequency {actual_frq_in_file} Hz. Skipping.")


    # Filter out frequencies for which we didn't get full data
    valid_TMFEx_indices = []
    for idx, frq_tmf in enumerate(TMFEx):
        if frq_tmf in processed_file_frequencies: # Was successfully processed
            if not np.all(np.isnan(magnitudes_db[idx, :, :])) and \
               not np.all(np.isnan(phases_deg[idx, :, :])):
                valid_TMFEx_indices.append(idx)
            
    if not valid_TMFEx_indices:
        print("No valid frequency data parsed fully from XHN.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Select only the valid data
    final_freqs = np.array(TMFEx)[valid_TMFEx_indices]
    final_magnitudes_db = magnitudes_db[valid_TMFEx_indices, :, :]
    final_phases_deg = phases_deg[valid_TMFEx_indices, :, :]

    min_db_val = np.nanmin(final_magnitudes_db) if not np.all(np.isnan(final_magnitudes_db)) else -200.0
    min_db_val = min(min_db_val - 20.0, -100.0) 
    final_magnitudes_db = np.nan_to_num(final_magnitudes_db, nan=min_db_val) 
    final_phases_deg = np.nan_to_num(final_phases_deg, nan=0.0)      

    amplitude_linear = 10**(final_magnitudes_db / 20.0)
    phase_radians = np.deg2rad(final_phases_deg)
    complex_data = amplitude_linear * np.exp(1j * phase_radians)

    phi_angles_deg = np.arange(0, 360 + dPhi, dPhi) 
    theta_angles_deg = np.arange(0, 180 + dTheta, dTheta) 

    return final_freqs, phi_angles_deg, theta_angles_deg, complex_data, final_magnitudes_db
