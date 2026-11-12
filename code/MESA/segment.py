from pathlib import Path
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import zarr
from scipy.stats import zscore
import scipy.signal as signal
import pickle
from tqdm import tqdm
from datetime import datetime,timedelta
import argparse

SKIP_SUBJECTS = [2934,2416]
def segment_process(meta_file : Path, raw_file: Path, output_file: Path, skip_subjects = SKIP_SUBJECTS):
    output_dir = Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_l = []
    meta = pd.read_csv(meta_file)
    for i in range(len(meta)):
        if meta['overall5'][i] == 7.0 and meta['mesaid'][i] not in skip_subjects:
            idx = meta['mesaid'][i]
            print('Subject ID:', idx)
            with open(f'{raw_file}/s_{idx}.pkl', 'rb') as f:
                data = pickle.load(f)
            fs = 256
            step_size = fs*30      # เลื่อนทีละ 30 วินาที (overlap 30 วินาที)
            window_size = fs*60    # หน้าต่าง 60 วินาที
            label_size = fs*30     # ใช้ event ใน 30 วินาทีแรกเป็น label
            ppg_w = []
            label_w = []
            a = 0
            h = 0
            tst = 0
            for i in range(0, len(data) - window_size + 1, step_size):
                window_ppg = data.iloc[i:i + window_size]
                # ppg_w.append(window_ppg['ppg'].values)
                label_sleep = window_ppg['label_sleep']
                counts = label_sleep.value_counts()
                label_event = window_ppg['label_event'].iloc[:label_size]
                if label_event.isin(['Obstructive Apnea']).any():
                    label_w.append('Apnea')
                    ppg_w.append(list(window_ppg['PPG']))

                elif label_event.isin(['Hypopnea']).any():
                    label_w.append('Hypopnea')
                    ppg_w.append(list(window_ppg['PPG']))
                else:
                    label_w.append('Normal')
                    ppg_w.append(list(window_ppg['PPG']))
            ppg_w = np.array(ppg_w)
            label_w = np.array(label_w)
            df = pd.DataFrame({
                'ppg': list(ppg_w),
                'label': label_w
            })
            with open(f'{output_file}/s_{idx}.pkl', 'wb') as f:
                pickle.dump(df, f)
            id_l.append(idx)
    print(f'Finish serment process: {len(id_l)}')
    return id_l

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Step 01: Annotation of event")
    p.add_argument("--meta", required=True, help='metadata')
    p.add_argument("--raw", required=True, help="coordinate ppg with event")
    p.add_argument("--output", required=True, help="event annotation located")
    p.add_argument("--skip", type=int, nargs="*", default=SKIP_SUBJECTS)
    args = p.parse_args()
    
    segment_process(args.meta, args.raw, args.output)


# python segment.py \
#     --meta /home/Data/MESA/datasets/mesa-sleep-dataset-0.5.0.csv \
#     --raw /home/work/Apnea/ApSense/benchmark/preprocess/MESA/ppgcoordinate \
#     --output   /home/work/Apnea/ApSense/benchmark/preprocess/MESA/segments/        
        