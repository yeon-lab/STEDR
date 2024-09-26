import argparse
import pickle
import os
import pandas as pd
from sas7bdat import SAS7BDAT
from drug import preprocess_drug
from cohort import exclude
from save_cohort_rx_dx import save_user_rx, save_user_dx
from cohort_dx_outcome import save_user_dx_outcome 
from save_cohort_input import preprocess_cohort
from utils import *

def get_patient_list(min_patients, saved_drugs):
    patients_list = set()
    for drug, patients in saved_drugs.items():
        if len(patients) >= min_patients:
            for patient in patients:
                patients_list.add(patient)
    return patients_list
    
def save_user_demo(args, patient_list):
    saved_cohort_demo = {}
    file = os.path.join(args.input_dir, 'cohort_demo.sas7bdat')
    with SAS7BDAT(file, skip_header=True) as demo:
        for row in demo:
            id, db, sex = row[0], row[1], row[2]
            if id in patient_list:
                saved_cohort_demo[id] = (db, sex)
    my_dump(saved_cohort_demo, os.path.join(args.pkl_dir, 'saved_cohort_demo.pkl'))
    return saved_cohort_demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--input_dir', default='../data/market', help='input data directory')
    parser.add_argument('--pkl_dir', default='../pickles', help='output data directory')
    parser.add_argument('--output_dir', default='../saved_cohort')
    parser.add_argument('--min_patients', default=500, type=int)
    parser.add_argument('--interval', default=180, type=int,help='minimum time interval for every two prescriptions')
    parser.add_argument('--followup', default=730, type=int, help='number of days of followup period')
    parser.add_argument('--baseline', default=365, type=int, help='number of days of baseline period')
    parser.add_argument('--src_dir', default='../resource', help='resource directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_dir):
        os.makedirs(args.pkl_dir)    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

        
    ndc_mapping = pickle.load(open(os.path.join(args.src_dir, 'ndc_to_ingredient.pkl'), 'rb'))
    
    
    icd9_to_ccs = pickle.load(open(os.path.join(args.src_dir, 'icd9_to_ccs.pkl'), 'rb'))
    icd10_to_ccs = pickle.load(open(os.path.join(args.src_dir, 'icd10_to_ccs.pkl'), 'rb'))
    icd_mapping = {'9': icd9_to_ccs, '0': icd10_to_ccs}
    
    icd9_cohort = [ 
        '33183', '2949'
    ]
    icd10_cohort = [
        'G3184', 'F09'
    ]
    cohort_dict = {'9': icd9_cohort, '0': icd10_cohort}
    
    icd9_outcomes = [ 
            '3310','29410', '29411', '29420','29421', '290'
    ]
    icd10_outcomes = [
            'G30', 'F01', 'F02', 'F03'
    ]
    outcome_dict = {'9': icd9_outcomes, '0': icd10_outcomes}
    
    print('='*100, flush=True)
    print(f'interval: {args.interval}', flush=True)
    print(f'followup: {args.followup}', flush=True)
    print(f'baseline: {args.baseline}', flush=True)
    print(f'icd9_outcomes: {icd9_outcomes}', flush=True)
    print(f'icd10_outcomes: {icd10_outcomes}', flush=True)
    print('='*100, flush=True)
    
    drug_taken_by_patient = preprocess_drug(args, ndc_mapping)
        
    cohort_list, user_outcome, user_dx = save_user_dx_outcome(args, icd_mapping, outcome_dict, cohort_dict)
        

    saved_drug, saved_patient = exclude(args, drug_taken_by_patient, cohort_list, user_outcome)
    #patient_list = get_patient_list(args.min_patients, saved_drug)
        
    saved_cohort_rx = save_user_rx(args, saved_drug, saved_patient)
    saved_cohort_dx = save_user_dx(args, saved_drug, user_dx)
    
    print('Preprocess cohort input...', flush=True)      
    preprocess_cohort(args, saved_drug, saved_cohort_rx, saved_cohort_dx, user_outcome)
