from sas7bdat import SAS7BDAT
from collections import defaultdict
import os
from utils import *

def preprocess_drug(args, ndc_mapping):
    print('Preprocessing drug...', flush=True)
    drug_taken_by_patient = defaultdict(dict)
    files = os.listdir(args.input_dir)
    for file in files:
        if 'drug' in file:
            input_file = os.path.join(args.input_dir, file)
            with SAS7BDAT(input_file, skip_header=False) as reader:
                print(f'> Load {file}...', flush=True)
                for i, row in enumerate(reader):
                    if i == 0:
                        id_idx = row.index('ENROLID')
                        ndc_idx = row.index('NDCNUM')
                        svc_idx = row.index('SVCDATE')
                        #age_idx = row.index('AGE')
                        supday_idx = row.index('DAYSUPP')
                        continue
                    enroll_id, ndc, svc_date, sup_day = row[id_idx],row[ndc_idx],row[svc_idx],row[supday_idx]
                    if svc_date and sup_day:
                        concept_ID = ndc_mapping.get(ndc)
                        if concept_ID:
                            if concept_ID not in drug_taken_by_patient:
                                drug_taken_by_patient[concept_ID][enroll_id] = set([(svc_date, sup_day)])
                            else:
                                if enroll_id not in drug_taken_by_patient.get(concept_ID):
                                    drug_taken_by_patient[concept_ID][enroll_id] = set([(svc_date, sup_day)])
                                else:
                                    drug_taken_by_patient[concept_ID][enroll_id].add((svc_date, sup_day))
                                    
    my_dump(drug_taken_by_patient, os.path.join(args.pkl_dir, 'drug_taken_by_patient.pkl'))
    print('Preprocessing drug completed!', flush=True)  
    print('-'*100, flush=True)
    print(f' > Total of drugs: {len(drug_taken_by_patient)}', flush=True)
    n_patients = 0
    for drug, patients in drug_taken_by_patient.items():
        print(f' > {drug} drug) # of patients: {len(patients)}', flush=True)
        n_patients += len(patients)
    print(f' > Total number of patients: {n_patients}', flush=True)
    print('='*100, flush=True)
    
    return drug_taken_by_patient

