import pandas as pd
import mne
import pickle
import re
from datetime import datetime, timedelta
import pandas as pd
import re
import mne
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
skip = [700045, 700217, 700173, 700144, 700065, 700129, 700151, 700267, 700053, 700174]
        
def merge_event(meta_file, edf_file, anno_file, hyp_file, output_file, skip):
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    meta = pd.read_csv(f'{meta_file}')
    for i in range(len(meta)):
        if meta['studyqual'][i] >= 5 and meta['nsrrid'][i] not in skip:
            id_o = meta['nsrrid'][i]
            print(f'Processing {id_o}')
            if id_o not in skip:
                filename = f'{edf_file}/heartbeat-baseline-{id_o}.edf'
                raw = mne.io.read_raw_edf(filename, preload=True)
                print(f'Done reading EDF')

                start_time = raw.info['meas_date']
                try:
                    ppg = raw.get_data(picks=['RD-Pleth'])
                except ValueError:
                    ppg = raw.get_data(picks=['RD-Pleth'])
                print(f'Therm shape: {ppg.shape}')

                start_time = pd.Timestamp(start_time)
                t = []
                for j in range(len(ppg[0])):
                    t.append(start_time + pd.Timedelta(seconds=j/raw.info['sfreq']))
                ppg = pd.DataFrame({
                    'time': t,
                    'ppg': ppg[0]
                })

                edf = f'{hyp_file}/heartbeat-baseline-{id_o}-nsrr.xml'
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
                if id_o == 700279 or id_o == 700143:
                    new_time = start_time
                else:
                    new_time = start_time + timedelta(seconds=start_dt)
                start_time = new_time.replace(year=ppg['time'][0].year)
                start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_time = new_time + timedelta(seconds=int(meta['sleeptotal'][meta['nsrrid'] == id_o].values[0]))
                end_time = end_time.replace(year=ppg['time'].iloc[-1].year)
                end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
                # start_time = pd.to_datetime(start_time).tz_localize('UTC')
                # end_time = pd.to_datetime(end_time).tz_localize('UTC')
                ppg['label_sleep'] = ['W'] * len(ppg)
                ppg.loc[(ppg['time'] >= start_time) & (ppg['time'] <= end_time), 'label_sleep'] = 'S'
                        
                with open(f'{anno_file}/s_{id_o}.pkl', 'rb') as f:
                    anno = pickle.load(f)
                
                ppg['time'] = ppg['time'].dt.tz_localize(None)
                rec_start = ppg['time'].iloc[0]            # tz-naive by now (cell 0 ran tz_localize(None))
                rec_day = rec_start.normalize()            # midnight of the recording date
                rec_tod = rec_start - rec_day              # time of day the recording started

                for col in ['Start', 'End']:
                    tod = anno[col] - anno[col].dt.normalize()                     # time of day as a Timedelta
                    anno[col] = rec_day + tod + pd.to_timedelta((tod < rec_tod).astype(int), unit='D')

                ppg['label_event'] = ['Normal'] * len(ppg)
                # ppg['time'] = ppg['time'].dt.tz_localize(None)
                for k in range(len(anno)):
                    if anno['Event'][k] == 'Hypopnea':
                        ppg.loc[(ppg['time'] >= anno['Start'][k]) & (ppg['time'] <= anno['End'][k]), 'label_event'] = 'Hypopnea'
                    elif anno['Event'][k] == 'Obstructive Apnea':
                        ppg.loc[(ppg['time'] >= anno['Start'][k]) & (ppg['time'] <= anno['End'][k]), 'label_event'] = 'Apnea'
                with open(f'{output_file}/s_{id_o}.pkl', 'wb') as f:
                    pickle.dump(ppg, f) 

                    
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Step 01: Annotation of event")
    p.add_argument("--meta", required=True, help='metadata')
    p.add_argument("--raw", required=True, help="raw signal .edf file")
    p.add_argument("--anno", required=True, help="event annotation located")
    p.add_argument("--hyp", required=True, help="hypnogram")
    p.add_argument("--output", required=True, help="ppg merge event annotation")
    p.add_argument("--skip", type=int, nargs="*", default=skip)
    args = p.parse_args()
    
    merge_event(args.meta, args.raw, args.anno, args.hyp, args.output, args.skip)

# python merge.py \
#     --meta /home/Data/heartbeat/datasets/heartbeat-baseline-dataset-0.4.0.csv \
#     --raw /home/Data/heartbeat/polysomnography/edfs/baseline \
#     --anno /home/work/Apnea/ApSense/benchmark/preprocess/HB/events \
#     --hyp /home/Data/heartbeat/polysomnography/annotations-events-nsrr/baseline \
#     --output /home/work/Apnea/ApSense/benchmark/preprocess/HB/ppgcoordinate/