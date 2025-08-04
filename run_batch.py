import os
from pathlib import Path
import re
import nbformat
import pandas as pd
import papermill as pm

# # Parameter combinations to explore #! SUBS
# PARAM_GRID = [
#     {'freq': 25, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':31.4},
#     {'freq': 31.5, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':43.5},
#     {'freq': 40, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':54.0},
#     {'freq': 50, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':57.9},
#     {'freq': 63, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':59.6},
#     {'freq': 80, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':61.2},
#     {'freq': 100, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE': 60.2},
#     {'freq': 125, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':56.8},
#     {'freq': 160, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE': 45.4},
#     {'freq': 200, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':42.7},
#     {'freq': 250, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE': 35.0},
# ]

# Parameter combinations to explore #! FULL System
PARAM_GRID = [
    {'freq': 25, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':31.7},
    {'freq': 31.5, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':43.6},
    {'freq': 40, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':54.1},
    {'freq': 50, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':57.1},
    {'freq': 63, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':54.0},
    {'freq': 80, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':48.6},
    {'freq': 100, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE': 52.5},
    {'freq': 125, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':64.9},
    {'freq': 160, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE': 66.4},
    {'freq': 200, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE':55.9},
    {'freq': 250, 'hv_veg': 0.0, 'CALIBRATION_SPL_AT_DISTANCE': 60.5},
]

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS = []

def parse_results(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    text = []
    for cell in nb.cells:
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                text.append(out.get('text', ''))
    full_text = ''.join(text)
    result = {}
    m = re.search(r'Total elapsed time: ([0-9.]+) seconds', full_text)
    if m:
        result['elapsed_seconds'] = float(m.group(1))
    m = re.search(r'Effective SPL at 1m for DeltaL calc: ([0-9.]+) dB', full_text)
    if m:
        result['spl_1m'] = float(m.group(1))
    return result

def main():
    for i, params in enumerate(PARAM_GRID, 1):
        out_nb = OUTPUT_DIR / f"run_{i}.ipynb"
        print(f"Running notebook for parameters {params}")
        pm.execute_notebook('t6_actual.ipynb', out_nb, parameters=params, log_output=True)
        run_res = params.copy()
        run_res.update(parse_results(out_nb))
        RESULTS.append(run_res)
    df = pd.DataFrame(RESULTS)
    df.to_csv(OUTPUT_DIR / 'batch_results.csv', index=False)
    print(f"Saved results to {OUTPUT_DIR / 'batch_results.csv'}")

if __name__ == '__main__':
    main()