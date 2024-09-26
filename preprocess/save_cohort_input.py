from datetime import datetime
import pickle
from tqdm import tqdm
import os
from utils import *
import numpy as np

def convert_to_date(obj):
    if isinstance(obj, datetime):
        return obj.date()
    else:
        return obj
        

def preprocess_cohort(args, saved_drug, saved_cohort_rx, saved_cohort_dx, user_outcome):
    cohorts_size = dict()
    n_patients = 0
    for drug, patients in tqdm(saved_drug.items()): 
        file_x = '{}/{}.pkl'.format(args.output_dir, drug)
        outcomes = []
        triples = []
        for patient, all_dates in patients.items():
            index_date = convert_to_date(all_dates[0])
            date_rx, date_dx = None, None
            set_rxs = saved_cohort_rx.get(drug)
            if set_rxs:
                date_rx = set_rxs.get(patient)
            
            set_dxs = saved_cohort_dx.get(drug)
            if set_dxs:
                date_dx = set_dxs.get(patient)

            rx_codes, dx_codes = [], []
            rx_dates, dx_dates = [], []
            if date_rx:
                for date, rx_code in sorted(date_rx.items(), key= lambda x:x[0]):
                    date = convert_to_date(date)
                    rx_codes.append(rx_code)
                    rx_dates.append((index_date-date).days)
            if date_dx:
                for date, dx_code in sorted(date_dx.items(), key= lambda x:x[0]):
                    date = convert_to_date(date)
                    dx_codes.append(list(dx_code))
                    dx_dates.append((index_date-date).days)

            if len(rx_codes) != 0 or len(dx_codes) != 0:
                outcome_date = user_outcome.get(patient, None)
                if outcome_date:
                    outcome = 1
                else:
                    outcome = 0
                outcomes.append(outcome)
                
                triples.append((patient, dx_codes, dx_dates, rx_codes, rx_dates, outcome))
        
               
        print('-'*100, flush=True)
        print(f'drug: {drug}', flush=True)
        print(f'> number of patients: {len(triples)}', flush=True)
        if len(triples) >= args.min_patients:
            my_dump(triples, file_x) 
            cohorts_size['{}.pkl'.format(drug)] = [len(triples)]
            unique, counts = np.unique(outcomes, return_counts=True)
            for value, count in zip(unique, counts):
                cohorts_size['{}.pkl'.format(drug)].append((value, count))
                print(f'>> {value}: {count}', flush=True)
        n_patients += len(triples)
        
    print(f'\nTotal number of drugs: {len(saved_drug)}', flush=True)
    print(f'Total number of patients: {n_patients}', flush=True)
    my_dump(cohorts_size, os.path.join(args.output_dir, 'cohorts_size.pkl'))
        
