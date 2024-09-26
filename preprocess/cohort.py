from collections import defaultdict
from sas7bdat import SAS7BDAT
from datetime import datetime
import os
from tqdm import tqdm
from utils import *

def get_patient_enroll_date(args):
    patient_start_date, patient_end_date = {}, {}            
    file = os.path.join(args.input_dir, 'cohort_ins.sas7bdat') 
    with SAS7BDAT(file, skip_header=True) as reader:
        print(f'load {file}...', flush=True)
        for row in reader:
            enrolid, start, end = row[0], row[1], row[2] # 'ENROLID', 'DTSTART', 'DTEND'
            patient_start_date[enrolid] = start
            patient_end_date[enrolid] = end
    return patient_start_date, patient_end_date
    
def get_patient_demo(args):
    patient_demo = {}          
    file = os.path.join(args.input_dir, 'cohort_demo.sas7bdat') 
    with SAS7BDAT(file, skip_header=True) as reader:
        print(f'load {file}...', flush=True)
        for row in reader:
            enrolid, DOBYR = row[0], row[1]
            patient_demo[enrolid] = int(DOBYR)
    return patient_demo
    
    
def criteria_1_is_valid(index_date, DX_date, DOBYR):
    return index_date > DX_date and int(DX_date.year) - DOBYR >= 50

def criteria_2_is_valid(index_date, start_date, baseline):
    return (index_date - start_date).days >= baseline
    
def criteria_3_is_valid(index_date, outcome_date):
    return outcome_date > index_date

def exclude(args, drug_taken_by_patient, cohort_list, user_outcome):
    patient_start_date, patient_end_date = get_patient_enroll_date(args)
    patient_demo = get_patient_demo(args)
    print('Exclude patient...', flush=True)
    saved_drug = defaultdict(dict)
    saved_patient = defaultdict(dict)
    for drug, patients in tqdm(drug_taken_by_patient.items(), total=len(drug_taken_by_patient)):
        for patient, taken_times in patients.items():
            dates = [convert_to_date(date) for (date, days) in taken_times if date and days]
            dates_days = {convert_to_date(date): int(days) for (date, days) in taken_times if date and days}
            dates = sorted(dates)
            index_date = dates[0]
            DOBYR = patient_demo.get(patient, int(datetime.max.year))
            DX_date = convert_to_date(cohort_list.get(patient, datetime.max))
            outcome_date = convert_to_date(user_outcome.get(patient, datetime.max))
            start_date = convert_to_date(patient_start_date.get(patient, datetime.max))
            #end_date = convert_to_date(patient_end_date.get(patient, datetime.min))
            if criteria_1_is_valid(index_date, DX_date, DOBYR) and criteria_2_is_valid(
            index_date, start_date, args.baseline) and criteria_3_is_valid(
            index_date, outcome_date):
                saved_drug[drug][patient] = dates
                saved_patient[patient][drug] = dates
                
    my_dump(saved_drug, os.path.join(args.pkl_dir, 'saved_drug.pkl'))
    my_dump(saved_patient, os.path.join(args.pkl_dir, 'saved_patient.pkl'))
    print('Exclude patient completed!', flush=True)  
    print('-'*100, flush=True)
    print(f' > Total of drugs: {len(saved_drug)}', flush=True)
    n_patients = 0
    for drug, patients in saved_drug.items():
        print(f' > {drug} drug) # of patients: {len(patients)}', flush=True)
        n_patients += len(patients)
    print(f'Total number of patients: {n_patients}', flush=True)
    print('='*100, flush=True)
    
    return saved_drug, saved_patient
    
    
def convert_to_date(obj):
    if isinstance(obj, datetime):
        return obj.date()
    else:
        return obj
