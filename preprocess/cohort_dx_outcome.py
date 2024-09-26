from collections import defaultdict
from datetime import  datetime
from sas7bdat import SAS7BDAT
from tqdm import tqdm
import os
import pandas as pd
from utils import *
import numpy as np

def get_ccs_code_for_icd(icd_codes, icd_to_ccs):
    ccs_codes = []
    for icd_code in icd_codes:
        for i in range(len(icd_code), -1, -1):
            ccs = icd_to_ccs.get(icd_code[:i])
            if ccs:
                ccs_codes.append(ccs)
                break
    return ccs_codes
    
def is_valid_range(icd_codes, code_range):
    for icd_code in icd_codes:
        for code in code_range:
            if icd_code.startswith(code):
                return True
    return False



def get_user_dx_outcome(args, icd_mapping, outcome_dict, cohort_dict):
    user_dx = defaultdict(dict)
    cohort_list, user_outcome = defaultdict(list), defaultdict(list)
    print('='*100)
    print('\nPreprocessing dx start...', flush=True)
    files = os.listdir(args.input_dir)
    for file in files:
        if 'inpat' in file or 'outpat' in file:
            input_file = os.path.join(args.input_dir, file)
            print(f'load {input_file}...', flush=True)
            with SAS7BDAT(input_file, skip_header=False) as reader:
                for i, row in enumerate(reader):
                    if i == 0:
                        id_idx = row.index('ENROLID')
                        svc_idx = [i for i, name in enumerate(row) if 'ADMDATE' in name or 'SVCDATE' in name][0]
                        dx_idx = [i for i, name in enumerate(row) if 'DX' in name and 'DXVER' not in name \
                                  and 'PDX' not in name and 'POADX' not in name]
                        try:
                            dxver_idx = row.index('DXVER')
                        except:
                            dxver_idx = False
                            DXVER = '9'
                        continue
                    enrolid = row[id_idx]
                    date = row[svc_idx]
                    dxs = np.array(row)[dx_idx]
                    dxs = dxs[dxs!='']
                    if dxver_idx:
                        DXVER = row[dxver_idx]
                    
                    if date and dxs.size > 0:
                        if is_valid_range(dxs, cohort_dict[DXVER]):
                            cohort_list[enrolid].append(date)
                        if is_valid_range(dxs, outcome_dict[DXVER]):
                            user_outcome[enrolid].append(date)
    
                        mapped_dxs = get_ccs_code_for_icd(dxs, icd_mapping[DXVER])
                        if len(mapped_dxs) > 0:
                            if enrolid not in user_dx:
                                user_dx[enrolid][date] = mapped_dxs
                            else:
                                if date not in user_dx[enrolid]:
                                    user_dx[enrolid][date] = mapped_dxs
                                else:
                                    user_dx[enrolid][date].extend(mapped_dxs)
    
    for user, dates in cohort_list.items():
        cohort_list[user] = min(dates)   
    for user, dates in user_outcome.items():
        user_outcome[user] = min(dates)   
    
    my_dump(cohort_list, os.path.join(args.output_dir, 'cohort_list.pkl'))
    my_dump(user_outcome, os.path.join(args.output_dir, 'user_outcome.pkl'))
    my_dump(user_dx, os.path.join(args.output_dir, 'user_dx.pkl'))
    
    
    print(f'length of cohort_list: {len(cohort_list)}', flush=True)
    print(f'length of user_outcome: {len(user_outcome)}', flush=True)
    print(f'length of user_dx: {len(user_dx)}', flush=True)
                
    return cohort_list, user_outcome, user_dx


def save_user_dx_outcome(args, icd_mapping, outcome_dict, cohort_dict):
    cohort_list, user_outcome, user_dx = get_user_dx_outcome(args, icd_mapping, outcome_dict, cohort_dict)
    my_dump(cohort_list, os.path.join(args.pkl_dir, 'cohort_list.pkl'))
    my_dump(user_outcome, os.path.join(args.pkl_dir, 'user_outcome.pkl'))
    my_dump(user_dx, os.path.join(args.pkl_dir, 'user_dx.pkl'))
    return cohort_list, user_outcome, user_dx
  
    
