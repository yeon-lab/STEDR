import os
from utils import *
from tqdm import tqdm


def drug_is_taken_in_baseline(index_date, dates):
    saved_dates = []
    for date in dates:
        if (index_date - date).days > 0:
            saved_dates.append(date)
    if len(saved_dates)>0:
        return saved_dates
    return False
                
                
def save_user_rx(args, saved_drug, saved_patient):
    print('Save user rx...', flush=True)
    saved_cohort_rx = AutoVivification()
    for drug, patients in saved_drug.items():
        for patient, take_dates in patients.items():
            index_date = take_dates[0]
            prescription_dates = saved_patient.get(patient)
            for prescription, dates_days in prescription_dates.items():
                saved_dates = drug_is_taken_in_baseline(index_date, dates_days)
                if saved_dates:
                    for date in saved_dates:
                        if drug not in saved_cohort_rx:
                            saved_cohort_rx[drug][patient][date] = [prescription]
                        else:
                            if patient not in saved_cohort_rx[drug]:
                                saved_cohort_rx[drug][patient][date] = [prescription]
                            else:
                                if date not in saved_cohort_rx[drug][patient]:
                                    saved_cohort_rx[drug][patient][date] = [prescription]
                                else:
                                    saved_cohort_rx[drug][patient][date].append(prescription)
    
    
    print(f'length of saved_cohort_rx: {len(saved_cohort_rx)}', flush=True)
    my_dump(saved_cohort_rx, os.path.join(args.pkl_dir, 'saved_cohort_rx.pkl'))
    
    return saved_cohort_rx
    
    
def save_user_dx(args, saved_drug, user_dx):
    print('Save user dx...', flush=True)
    saved_cohort_dx = AutoVivification()
    for drug, patients in tqdm(saved_drug.items()):
        for patient, taken_times in patients.items():
            index_date = taken_times[0]
            date_codes = user_dx.get(patient)
            if date_codes:
                for date, codes in date_codes.items():
                    if date < index_date:
                        if drug not in saved_cohort_dx:
                            saved_cohort_dx[drug][patient][date] = set(codes)
                        else:
                            if patient not in saved_cohort_dx[drug]:
                                saved_cohort_dx[drug][patient][date] = set(codes)
                            else:
                                if date not in saved_cohort_dx[drug][patient]:
                                    saved_cohort_dx[drug][patient][date] = set(codes)
                                else:
                                    saved_cohort_dx[drug][patient][date].union(codes)

    print(f'length of saved_cohort_dx: {len(saved_cohort_dx)}', flush=True)
    my_dump(saved_cohort_dx, os.path.join(args.pkl_dir, 'saved_cohort_dx.pkl'))
    
    return saved_cohort_dx

