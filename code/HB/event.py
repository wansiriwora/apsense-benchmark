# HeartBEAT
import os
import pickle
import glob
import gc
import re

import pyedflib

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
def our_annot_hypop(filename):
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
        date_format = "%H.%M.%S"
        start_time = datetime.strptime(patterns_time[0][9:], date_format)
        patterns_hypopnea = re.findall(
            r'<EventType>Respiratory.Respiratory</EventType>\n' +
            r'<EventConcept>Hypopnea.HYPOPNEA</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n' +
            r'<Duration>[0-9\.]+</Duration>\n'+
            r'<SignalLocation>Thermistor</SignalLocation>\n'+
            r'</ScoredEvent>\n'+
            r'<ScoredEvent>\n'+
            r'<EventType>Respiratory.Respiratory</EventType>\n'+
            r'<EventConcept>SpO2 desaturation.DESAT</EventConcept>\n'+
            r'<Start>[0-9\.]+</Start>\n'+
            r'<Duration>[0-9\.]+</Duration>\n'+
            r'<SignalLocation>SpO2</SignalLocation>\n'+
            r'<SpO2Nadir>[0-9\.]+</SpO2Nadir>\n'+
            r'<SpO2Baseline>[0-9\.]+</SpO2Baseline>\n'+
            r'</ScoredEvent>'
            ,
            content)
        hypopnea_l = []
        starts = []
        ends = []
        duration = []
        for pattern in patterns_hypopnea:
            lines = pattern.splitlines()
            stageline = lines[1]
            hypopnea = str(stageline[23:31])
            spo2line = lines[8]
            spo2 = str(spo2line[14:18])
            spo2line = lines[8]
            drop = lines[12]
            dropnum = str(drop[11:13])
            number = re.search(r'\d+', dropnum)
            dropnum = int(number.group())
            baseline = lines[13]
            basenum = str(baseline[14:16])
            number = re.search(r'\d+', basenum)
            basenum = int(number.group())
            if hypopnea == 'HYPOPNEA' and spo2 == 'SpO2' and basenum - dropnum >= 3:
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
    except Exception as e:
        print(f"Error: {e}")
        df_h = pd.DataFrame()
    return df_h

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
        date_format = "%H.%M.%S"
        start_time = datetime.strptime(patterns_time[0][9:], date_format)
        patterns_apnea = re.findall(
            r'<EventType>Respiratory.Respiratory</EventType>\n' +
            r'<EventConcept>Mixed apnea.APNEA-MIXED</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n' +
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
            apnea = str(stageline[-26:-21])
            if apnea == 'APNEA':
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
            r'<EventConcept>Hypopnea.HYPOPNEA</EventConcept>\n' +
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
            hypopnea = str(stageline[23:31])
            # print(hypopnea)
            if hypopnea == 'HYPOPNEA':
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

import numpy as np
import pickle
import pandas as pd
import zarr
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt, find_peaks

def lowpass_filter(signal, fs, cutoff=1.0, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)

def apnea_detection(df, sleep_event, fs):
    pass_apnea = []
    start_our = []
    end_our = []
    for e in range(len(sleep_event)):
        if sleep_event['Duration'][e] >= 10:
            start_a = sleep_event['Start'][e].tz_localize('UTC')
            end_a = sleep_event['End'][e].tz_localize('UTC')
            start_a =start_a.replace(year=df['time'].iloc[-1].year)
            end_a =end_a.replace(year=df['time'].iloc[-1].year)
            focus_end = end_a + timedelta(seconds = 10)
            focus_e = start_a - timedelta(minutes = 2)
            thres = 1.75
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC')
            else:
                df['time'] = df['time'].dt.tz_convert('UTC')
            
            signal = df[(df['time'] >= focus_e) & (df['time'] <= focus_end)].reset_index(drop=True)
            norm_signal = (signal['Therm'] - np.mean(signal['Therm'])) / np.std(signal['Therm'])
            filtered_signal = lowpass_filter(norm_signal, fs, cutoff=0.5, order=4)
            peaks, _ = find_peaks(filtered_signal, prominence = 0.00005, distance=fs*thres)
            
            offsets = []
            for i in range(len(peaks)-1):
                start = peaks[i]
                end = peaks[i + 1]
                segment = filtered_signal[start:end + 1]
                
                if len(segment) > 0:
                    local_min_idx = np.argmin(segment)
                    offsets.append(start + local_min_idx)
                            
            amplitude = []
            for i in range(len(offsets)):
                amplitude.append(np.abs(filtered_signal[peaks[i]] - filtered_signal[offsets[i]]))
            
            if len(offsets) > 0 :
                # Find the average of baseline value
                baseline = signal[(signal['time'] >= focus_e) & (signal['time'] <= start_a)].reset_index(drop=True)
                if len(baseline) == 1:
                    print('No baseline found')
                    continue
                norm_signal = (baseline['Therm'] - np.mean(baseline['Therm'])) / np.std(baseline['Therm'])
                filtered_baseline = lowpass_filter(norm_signal, fs, cutoff=0.5, order=4)
                baseline_peaks, _ = find_peaks(filtered_baseline, prominence = 0.00005, distance=fs*thres)
                baseline_offsets = []
                for i in range(len(baseline_peaks)-1):
                    start = baseline_peaks[i]
                    end = baseline_peaks[i + 1]
                    segment = filtered_baseline[start:end + 1]
                    
                    if len(segment) > 0:
                        local_min_idx = np.argmin(segment)
                        baseline_offsets.append(start + local_min_idx)
                baseline_amplitude = []
                for i in range(len(baseline_offsets)):
                    baseline_amplitude.append(np.abs(filtered_baseline[baseline_peaks[i]] - filtered_baseline[baseline_offsets[i]]))
                baseline_mean = np.mean(np.sort(baseline_amplitude)[-3:])
            
                amplitude_event = amplitude[len(baseline_amplitude): len(amplitude)]
                if len(amplitude_event) == 0:
                    print('No amplitude event found')
                    continue
                lowest_point = np.argmin(amplitude_event)
                lowest_amplitude = amplitude_event[lowest_point]
                lowest_point = (lowest_point + 1) + len(baseline_amplitude)
                if lowest_point == len(amplitude):
                    amplitude_event = amplitude[len(baseline_amplitude): len(amplitude)-1]
                    if len(amplitude_event) == 0:
                        amplitude_event = amplitude[len(baseline_amplitude): len(amplitude)]
                    lowest_point = np.argmin(amplitude_event)
                    lowest_amplitude = amplitude_event[lowest_point]
                    lowest_point = (lowest_point + 1) + len(baseline_amplitude)
                    if (baseline_mean - lowest_amplitude)/baseline_mean * 100 >= 90:
                        for i in range(len(baseline_amplitude),len(amplitude)):
                            drop_value = baseline_mean - amplitude[i]
                            if drop_value > 0:
                                start_time_our = signal['time'].iloc[offsets[i]]
                                break
                        
                        threshold = lowest_amplitude + 0.02 * lowest_amplitude    
                        for j in range(lowest_point,len(amplitude)):
                            if amplitude[j] >= threshold:
                                end_time_our = signal['time'].iloc[peaks[lowest_point+1]]
                                break
                            else:
                                end_time_our = signal['time'].iloc[peaks[-1]]
                                break
                        
                        if (end_time_our - start_time_our).total_seconds() >= 10:
                            start_our.append(start_time_our)
                            end_our.append(end_time_our)
                            pass_apnea.append('Obstructive Apnea')
                            
                else:
                    if (baseline_mean - lowest_amplitude)/baseline_mean * 100 >= 90:
                        
                        for i in range(len(baseline_amplitude),len(amplitude)):
                            drop_value = baseline_mean - amplitude[i]
                            if drop_value > 0:
                                start_time_our = signal['time'].iloc[offsets[i]]
                                break
                        
                        threshold = lowest_amplitude + 0.02 * lowest_amplitude    
                        for j in range(lowest_point,len(amplitude)):
                            if amplitude[j] >= threshold:
                                end_time_our = signal['time'].iloc[peaks[lowest_point+1]]
                                break
                            else:
                                end_time_our = signal['time'].iloc[peaks[-1]]
                                break
                        
                        if (end_time_our - start_time_our).total_seconds() >= 10:
                            start_our.append(start_time_our)
                            end_our.append(end_time_our)
                            pass_apnea.append('Obstructive Apnea')
                            
                

            else:
                print('No offsets found')
    apnea_our = pd.DataFrame({
        'Start': start_our,
        'End': end_our,
        'Event': pass_apnea
    })
    return apnea_our
import pandas as pd
def merge_event(df):
    df = df.sort_values(by='Start').reset_index(drop=True)
    merged_events = []

    i = 0
    while i < len(df):
        row = df.iloc[i]
        
        if row['Event'] == 'Obstructive Apnea':
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
import pandas as pd
import re
import mne
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

SKIP_SUBJECTS = [700045, 700217, 700173, 700144, 700065, 700129, 700151, 700267, 700053, 700174]
def event_process(meta_file: Path, annotate: Path, edf_file: Path, output_file: Path, skip_subjects: SKIP_SUBJECTS):
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    meta = pd.read_csv(f'{meta_file}')            
    skip = SKIP_SUBJECTS
    our_event = pd.DataFrame()
    for i in range(len(meta)):
        if meta['studyqual'][i] >= 5 and meta['nsrrid'][i] not in skip:
            idx = meta['nsrrid'][i]
            print(f'Processing {idx}')
            
            # Load xml file
            file_event = f'{annotate}/heartbeat-baseline-{str(idx).zfill(4)}-nsrr.xml'
            
            # ดึงเฉพาะ apnea, hypopnea event จาก xml โดยไม่สนใจ stage
            total, apnea, hypop = read_annot_event(file_event)
            
            # ดึงเฉพาะ hypopnea event จาก xml ที่มี SpO2 drop มากกว่า 3% หรือมีี arousal ที่อยู่ตามหลัง Hypopnea event โดยไม่สนใจ stage
            hypop_a = our_annot_hypop(file_event)
            
            # ดึงเวลาที่เริ่มanalysis period คิดว่าเป็นจุดแรกที่เริ่มนาน (start sleep time)
            edf = f'{annotate}/heartbeat-baseline-{str(idx).zfill(4)}-nsrr.xml'
            with open(edf, 'r') as f:
                content = f.read()
            patterns_start = re.findall(
                r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', 
                content)
            assert len(patterns_start) == 1
            patterns_time = re.findall(
                r'<ClockTime>(\d{2}\.\d{2}\.\d{2} \d{2}\.\d{2}\.\d{2})</ClockTime>',
                content
            )
            date_format = "%H.%M.%S"
            start_time = datetime.strptime(patterns_time[0][9:], date_format)
            pattern_analysis = re.findall(
                r'<EventType>Stages.Stages</EventType>\n' +
                r'<EventConcept>Beginning of analysis period.ANALYSIS-START</EventConcept>\n' +
                r'<Start>[0-9\.]+</Start>\n' +
                r'<Duration>[0-9\.]+</Duration>',
                content
            )
            for pattern in pattern_analysis:
                lines = pattern.splitlines()
                start = lines[2]
                start_dt = float(start[7:-8])
            
            # idx 700279 มีปัญหาเรื่องเวลาที่เริ่มนอน 
            if idx == 700279 or idx == 700143:
                new_time = start_time
            else:
                new_time = start_time + timedelta(seconds=start_dt)
            # start_time = new_time.replace(year=df['time'][0].year)
            start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # เอาเวลาที่เริ่มนอนมาเป็นเวลาที่เริ่ม sleep แล้วรวมกับ total sleep time เป็น wake time
            end_time = new_time + timedelta(seconds=int(meta['sleeptotal'][meta['nsrrid'] == idx].values[0]))
            # end_time = end_time.replace(year=df['time'].iloc[-1].year)
            end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            
            apnea_e = []
            start_a = []
            end_a = []
            duration_a = []
            for i in range(len(apnea)):
                if (apnea['Start'][i] >= start_time) and (apnea['End'][i] <= end_time) and apnea['Duration'][i] >= 10:
                    apnea_e.append(apnea['Event'][i])
                    start_a.append(apnea['Start'][i])
                    end_a.append(apnea['End'][i])
                    duration_a.append(apnea['Duration'][i])
            
            apnea_anno = pd.DataFrame({
                'Event': apnea_e,
                'Start': start_a,
                'End': end_a,
                'Duration': duration_a
            })
            
            hypop_e = []
            start_h = []
            end_h = []
            duration_h = []
            for i in range(len(hypop)):
                if (hypop['Start'][i] >= start_time) and (hypop['End'][i] <= end_time) and hypop['Duration'][i] >= 10:
                    hypop_e.append(hypop['Event'][i])
                    start_h.append(hypop['Start'][i])
                    end_h.append(hypop['End'][i])
                    duration_h.append(hypop['Duration'][i])
            
            hypop_anno = pd.DataFrame({
                'Event': hypop_e,
                'Start': start_h,
                'End': end_h,
                'Duration': duration_h
            })

            our_event = pd.concat([apnea_anno,hypop_anno])
            our_event = our_event.sort_values(by='Start').reset_index(drop=True)
            i = 1
            while i < len(our_event):
                if (our_event.iloc[i]['Start'] - our_event.iloc[i-1]['End']).total_seconds() < 0:
                    our_event.at[i-1, 'End'] = our_event.iloc[i]['End']
                    our_event.drop(index=our_event.index[i], inplace=True)
                    our_event.reset_index(drop=True, inplace=True)
                    # อย่าเพิ่ม i ในกรณีลบแถว เพราะ index เปลี่ยน
                else:
                    i += 1
            our_event = merge_event(our_event)
            with open(f'{output_file}/s_{idx}.pkl', 'wb') as f:
                pickle.dump(our_event, f)
            del apnea, total, apnea_anno, hypop, hypop_anno
            gc.collect()
            
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Step 01: Annotation of event")
    p.add_argument("--metadata", required=True, help='metadata')
    p.add_argument("--annotate", required=True, help="xml file annotated")
    p.add_argument("--edf", required=True, help="edf file location")
    p.add_argument("--output", required=True, help="event annotation located")
    p.add_argument("--skip", type=int, nargs="*", default=SKIP_SUBJECTS)
    args = p.parse_args()
    
    event_process(args.metadata, args.annotate, args.edf, args.output, args.skip)
    
# python event.py \
#     --metadata /home/Data/heartbeat/datasets/heartbeat-baseline-dataset-0.4.0.csv \
#     --annotate /home/Data/heartbeat/polysomnography/annotations-events-nsrr/baseline \
#     --edf /home/Data/heartbeat/polysomnography/edfs/baseline \
#     --output   /home/work/Apnea/ApSense/benchmark/preprocess/HB/events/