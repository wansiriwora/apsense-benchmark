
import numpy as np
import pickle
import pandas as pd
import zarr
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt, find_peaks
import gc


def read_stage(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        # Check that there is only one 'Start time' and that it is 0
        patterns_start = re.findall(
            r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', 
            content)
        assert len(patterns_start) == 1
        patterns_time = re.findall(
        r'<ClockTime>(\d{2}\.\d{2}\.\d{2} \d{2}\.\d{2}\.\d{2})</ClockTime>',
        content
        )
        date_format = "%d.%m.%y %H.%M.%S"
        start_time = datetime.strptime(patterns_time[0], date_format)
        patterns_wake = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>Wake.0</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        apnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_wake:
            lines = pattern.splitlines()
            stageline = lines[1]
            apnea = str(stageline[-21:-17])
            if apnea == 'Wake':
                apnea_l.append('Wake')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            duration.append(epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
        df_w = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': apnea_l
        })
        patterns_n1 = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>Stage 1 sleep.1</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        apnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_n1:
            lines = pattern.splitlines()
            stageline = lines[1]
            apnea = str(stageline[-30:-23])
            if apnea == 'Stage 1':
                apnea_l.append('N1')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            duration.append(epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
        df_n1 = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': apnea_l
        })
        patterns_n2 = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>Stage 2 sleep.2</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        apnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_n2:
            lines = pattern.splitlines()
            stageline = lines[1]
            apnea = str(stageline[-30:-23])
            if apnea == 'Stage 2':
                apnea_l.append('N2')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            duration.append(epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
        df_n2 = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': apnea_l
        })
        patterns_n3 = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>Stage 3 sleep.3</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        apnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_n3:
            lines = pattern.splitlines()
            stageline = lines[1]
            apnea = str(stageline[-30:-23])
            if apnea == 'Stage 3':
                apnea_l.append('N3')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            duration.append(epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
        df_n3 = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': apnea_l
        })
        patterns_n4 = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>Stage 4 sleep.4</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        apnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_n4:
            lines = pattern.splitlines()
            stageline = lines[1]
            apnea = str(stageline[-30:-23])
            if apnea == 'Stage 4':
                apnea_l.append('N4')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            duration.append(epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
        df_n4 = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': apnea_l
        })
        patterns_r = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>REM sleep.5</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        apnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_r:
            lines = pattern.splitlines()
            stageline = lines[1]
            apnea = str(stageline[-26:-23])
            if apnea == 'REM':
                apnea_l.append('R')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            duration.append(epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
        df_r = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': apnea_l
        })
    except Exception as e:
        print(f"Error: {e}")
    sleep = pd.concat([df_n1, df_n2, df_n3, df_n4, df_r]).reset_index(drop=True)
    sleep = sleep.sort_values(by='Start').reset_index(drop=True)
    return df_w,sleep

# MESA
import os
import pickle
import glob

import re

import pyedflib

import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def read_annot_event(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        # Check that there is only one 'Start time' and that it is 0
        patterns_start = re.findall(
            r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', 
            content)
        assert len(patterns_start) == 1
        patterns_time = re.findall(
        r'<ClockTime>(\d{2}\.\d{2}\.\d{2} \d{2}\.\d{2}\.\d{2})</ClockTime>',
        content
        )
        date_format = "%d.%m.%y %H.%M.%S"
        start_time = datetime.strptime(patterns_time[0], date_format)
        patterns_apnea = re.findall(
            r'<EventType>Respiratory.Respiratory</EventType>\n' +
            r'<EventConcept>Obstructive apnea.Obstructive Apnea</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        apnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_apnea:
            lines = pattern.splitlines()
            stageline = lines[1]
            apnea = str(stageline[-20:-15])
            if apnea == 'Apnea':
                apnea_l.append('Apnea')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            duration.append(epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
        df_a = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': apnea_l
        })
        patterns_hypopnea = re.findall(
            r'<EventType>Respiratory.Respiratory</EventType>\n' +
            r'<EventConcept>Hypopnea.Hypopnea</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n' +
            r'<Duration>[0-9\.]+</Duration>',
            content)
        # print(patterns_stages[-1])
        hypopnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_hypopnea:
            lines = pattern.splitlines()
            stageline = lines[1]
            # print(stageline)
            hypopnea = str(stageline[-23:-15])
            # print(hypopnea)
            if hypopnea == 'Hypopnea':
                hypopnea_l.append('Hypopnea')
            
            startline = lines[2]
            start = float(startline[7:-8])
            durationline = lines[3]
            epochs_duration = float(durationline[10:-11])
            epoch_start = start_time + timedelta(seconds=start)
            epoch_end = epoch_start + timedelta(seconds=epochs_duration)
            starts.append(epoch_start)
            ends.append(epoch_end)
            duration.append(epochs_duration)
        df_h = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': duration,
            'Event': hypopnea_l
        })
        df = pd.concat([df_a, df_h]).reset_index(drop=True)
    except Exception as e:
        print(f"Error: {e}")
        df = pd.DataFrame()
    return df, df_a, df_h
import pandas as pd
def merge_event(df):
    df = df.sort_values(by='Start').reset_index(drop=True)
    merged_events = []

    i = 0
    while i < len(df):
        row = df.iloc[i]
        
        if row['Event'] == 'Apnea':
            start = row['Start']
            end = row['End']

            # Check if next row is Hypopnea and overlaps
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                if next_row['Event'] == 'Hypopnea' and next_row['Start'] <= end:
                    # Extend end to the end of Hypopnea
                    end = max(end, next_row['End'])
                    i += 1  # Skip this Hypopnea
            merged_events.append({
                'Start': start,
                'End': end,
                'Event': 'Obstructive Apnea'
            })

        elif row['Event'] == 'Hypopnea':
            # Check if already merged with Obstructive Apnea
            if i == 0 or df.iloc[i - 1]['End'] < row['Start']:
                merged_events.append({
                    'Start': row['Start'],
                    'End': row['End'],
                    'Event': 'Hypopnea'
                })
        i += 1

    return pd.DataFrame(merged_events)

from pathlib import Path
import argparse

SKIP_SUBJECTS = [1738,6476]


def event_process(meta_file: Path, annotate: Path, output_file: Path, skip_subjects = SKIP_SUBJECTS):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    our_event = pd.DataFrame()
    id_l = []
    meta = pd.read_csv(meta_file)
    for i in range(len(meta)):
        if meta['overall5'][i] == 7.0 and meta['mesaid'][i] not in skip_subjects:
            idx = meta['mesaid'][i]
            print('Processing ID:', idx)
            file_event = f'{annotate}/mesa-sleep-{str(idx).zfill(4)}-nsrr.xml'
            total, apnea, hypop = read_annot_event(file_event)
            wake, sleep = read_stage(file_event)
            # Filter wake class out
            drop_idx_apnea = []
            for i in range(len(apnea)):
                for j in range(len(wake)):
                    if apnea['End'][i] >= wake['Start'][j] and apnea['Start'][i] <= wake['End'][j]:
                        # print(f"Apnea {i} overlaps with Wake Stage {wake['Event'][j]} from {wake['Start'][j]} to {wake['End'][j]}")
                        drop_idx_apnea.append(i)
                        break 
                    elif apnea['Duration'][i] < 10:
                        drop_idx_apnea.append(i)
                        break

            apnea = apnea.drop(index=drop_idx_apnea).reset_index(drop=True)

            drop_idx_hypop = []
            for i in range(len(hypop)):
                for j in range(len(wake)):
                    if hypop['End'][i] >= wake['Start'][j] and hypop['Start'][i] <= wake['End'][j]:
                        drop_idx_hypop.append(i)
                        break
                    elif hypop['Duration'][i] < 10:
                        drop_idx_hypop.append(i)
                        break

            hypop = hypop.drop(index=drop_idx_hypop).reset_index(drop=True)

            df = pd.concat([apnea, hypop]).reset_index(drop=True).sort_values(by='Start').reset_index(drop=True)  
            i = 1
            while i < len(df):
                if (df.iloc[i]['Start'] - df.iloc[i-1]['End']).total_seconds() < 0:
                    df.at[i-1, 'End'] = df.iloc[i]['End']
                    df.drop(index=df.index[i], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    # อย่าเพิ่ม i ในกรณีลบแถว เพราะ index เปลี่ยน
                else:
                    i += 1
            df = merge_event(df)
            with open(f'{output_file}/s_{idx}.pkl', 'wb') as f:
                pickle.dump(df, f)
            id_l.append(idx)
    print(f'Finish event annotation: {len(id_l)}')
    return id_l

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Step 01: Annotation of event")
    p.add_argument("--metadata", required=True, help='metadata')
    p.add_argument("--annotate", required=True, help="xml file annotated")
    p.add_argument("--output", required=True, help="event annotation located")
    p.add_argument("--skip", type=int, nargs="*", default=SKIP_SUBJECTS)
    args = p.parse_args()
    
    event_process(args.metadata, args.annotate, args.output)
    
# python event.py \
#     --metadata /home/Data/MESA/datasets/mesa-sleep-dataset-0.5.0.csv \
#     --annotate /home/Data/MESA/polysomnography/annotations-events-nsrr \
#     --output   /home/work/Apnea/ApSense/benchmark/preprocess/MESA/events/