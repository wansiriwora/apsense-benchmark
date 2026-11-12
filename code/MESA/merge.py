import os
import pickle
import glob
import re
import pyedflib
import numpy as np
from datetime import datetime, timedelta

import pyedflib
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import argparse
from pathlib import Path

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

def read_signal(filename):
    f = pyedflib.EdfReader(filename)
    start_time = f.getStartdatetime()
    start_time = pd.Timestamp(start_time)

    signal_labels = f.getSignalLabels()
    channel = 'Pleth'
    assert channel in signal_labels, f"Channel {channel} not found in signal labels."
    idx_chan = [i for i, x in enumerate(signal_labels) if x == channel][0]
    sigbuf = np.zeros((f.getNSamples()[idx_chan]))
    sigbuf[:] = f.readSignal(idx_chan)
    sampling_rate = len(sigbuf) / f.file_duration

    t = []

    for i in range(len(sigbuf)):
        t.append(start_time + pd.Timedelta(seconds=i/sampling_rate))

    df = pd.DataFrame({
        'time': t,
        'PPG': sigbuf
    })

    f.close()
    return df

SKIP_SUBJECTS = [1738,6476]

def merge_process(meta_file: Path, raw_file: Path, anno_file: Path, hyp_file: Path, output_file: Path, skip_subjects = SKIP_SUBJECTS):
    output_dir = Path(output_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(meta_file)
    id_l = []
    for i in range(len(meta)):
        if meta['overall5'][i] == 7.0:
            if meta['mesaid'][i] not in skip_subjects:
                id_o = meta['mesaid'][i]
                print('Subject ID:', id_o)
                id_str = f"{id_o:04}"
                ppg_file = f'{raw_file}/mesa-sleep-{id_str}.edf'
                # ppg = read_ppg(ppg_file, channel='Pleth', window_size=120, overlap=0)
                ppg = read_signal(ppg_file)
                with open(f'{anno_file}/s_{id_o}.pkl', 'rb') as f:
                    anno = pickle.load(f)
                file_event = f'{hyp_file}/mesa-sleep-{str(id_o).zfill(4)}-nsrr.xml'
                wake, sleep = read_stage(file_event)
                
                label_sleep = ['S'] * len(ppg)
                for i in range(len(wake)):
                    wake_ppg = ppg[(ppg['time'] <= wake['End'][i]) & (ppg['time'] >= wake['Start'][i])]
                    for j in range(len(wake_ppg)):
                        label_sleep[wake_ppg.index[j]] = 'W'
                ppg['label_sleep'] = label_sleep

                label_event = ['Normal'] * len(ppg)
                for k in range(len(anno)):
                    if anno['Event'][k] == 'Hypopnea':
                        hypopnea_ppg = ppg[(ppg['time'] <= anno['End'][k]) & (ppg['time'] >= anno['Start'][k])]
                        for j in range(len(hypopnea_ppg)):
                            label_event[hypopnea_ppg.index[j]] = 'Hypopnea'
                    elif anno['Event'][k] == 'Obstructive Apnea':
                        apnea_ppg = ppg[(ppg['time'] <= anno['End'][k]) & (ppg['time'] >= anno['Start'][k])]
                        for j in range(len(apnea_ppg)):
                            label_event[apnea_ppg.index[j]] = 'Obstructive Apnea'
                ppg['label_event'] = label_event
                with open(f'{output_file}/s_{id_o}.pkl', 'wb') as f:
                    pickle.dump(ppg, f)     
                id_l.append(id_o) 
    print(f'Finish merging event with raw ppg: {len(id_l)}')     
    return id_l

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Step 01: Annotation of event")
    p.add_argument("--meta", required=True, help='metadata')
    p.add_argument("--raw", required=True, help="raw signal .edf file")
    p.add_argument("--anno", required=True, help="event annotation located")
    p.add_argument("--hyp", required=True, help="hypnogram")
    p.add_argument("--output", required=True, help="ppg merge event annotation")
    p.add_argument("--skip", type=int, nargs="*", default=SKIP_SUBJECTS)
    args = p.parse_args()
    
    merge_process(args.meta, args.raw, args.anno, args.hyp, args.output)
    
# python merge.py \
#     --meta /home/Data/MESA/datasets/mesa-sleep-dataset-0.5.0.csv \
#     --raw /home/Data/MESA/polysomnography/edfs \
#     --anno /home/work/Apnea/ApSense/benchmark/preprocess/MESA/events \
#     --hyp /home/Data/MESA/polysomnography/annotations-events-nsrr \
#     --output   /home/work/Apnea/ApSense/benchmark/preprocess/MESA/ppgcoordinate/