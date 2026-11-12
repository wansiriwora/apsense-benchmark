from pathlib import Path
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import zscore
import scipy.signal as signal
import pickle
from tqdm import tqdm
from datetime import datetime,timedelta
import argparse

skip = [700045, 700217, 700173, 700144, 700065, 700129, 700151, 700267, 700053, 700174]
def segment_process(meta_file : Path, raw_file: Path, output_file: Path, skip_subjects = skip):
    output_dir = Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_l = []
    meta = pd.read_csv(f'{meta_file}')
    for i in range(len(meta)):
        if meta['studyqual'][i] >= 5 and meta['nsrrid'][i] not in skip:
            print('Processing ID:', meta['nsrrid'][i])
            idx = meta['nsrrid'][i]
            with open(f'{raw_file}/s_{idx}.pkl', 'rb') as f:
                data = pickle.load(f)
            minu = 1
            fs = 200
            step_size = fs*30
            window_size = minu*fs*60
            label_size = fs * 30
            ppg_w = []
            label_w = []
            for i in range(0,len(data)- int(window_size) +1, int(step_size)):
                window_ppg = data[i:i + window_size]
                label_sleep = window_ppg['label_sleep']
                counts = label_sleep.value_counts()
                label_event = window_ppg['label_event'].iloc[:label_size]
                if label_event.isin(['Apnea']).any():
                    label_w.append(1)
                    ppg_w.append(list(window_ppg['ppg']))

                elif label_event.isin(['Hypopnea']).any():
                    label_w.append(1)
                    ppg_w.append(list(window_ppg['ppg']))
                    
                else:
                    label_w.append(0)
                    ppg_w.append(list(window_ppg['ppg']))
                    
            ppg_w = np.array(ppg_w)
            label_w = np.array(label_w)
            df = pd.DataFrame({
                'ppg': list(ppg_w),
                'label': label_w
            })
            with open(f'{output_file}/s_{idx}.pkl', 'wb') as f:
                pickle.dump(df, f)
        
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Step 01: Annotation of event")
    p.add_argument("--meta", required=True, help='metadata')
    p.add_argument("--raw", required=True, help="coordinate ppg with event")
    p.add_argument("--output", required=True, help="event annotation located")
    p.add_argument("--skip", type=int, nargs="*", default=skip)
    args = p.parse_args()
    
    segment_process(args.meta, args.raw, args.output)                        

# python segment.py \
#     --meta /home/Data/heartbeat/datasets/heartbeat-baseline-dataset-0.4.0.csv \
#     --raw /home/work/Apnea/ApSense/benchmark/preprocess/HB/ppgcoordinate \
#     --output   /home/work/Apnea/ApSense/benchmark/preprocess/HB/segments/