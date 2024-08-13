The data analyzed in this paper is MarketScan MDCR from 2012 to 2017. Access to the MarketScan data is provided by the Ohio State University. The dataset is available from 

***Please note that the dataset provided here is entirely composed of randomly generated example data. As a result, any experimental findings or results derived from it are not considered meaningful or significant.***


***

#### Data description
1. demo.csv contains the demographics information of all the patients.
    - ENROLID: 511,274 unique patients’ id
    - DOBYR: birth year
    - SEX: gender: 1- male; 2- female
    - MSA: Metropolitan Statistical Area – city mappings
    - REGION: Geographic Region of employee residence – region mappings


2. There are tables from year 2012 to year 2017
     [SPECIAL Note: Schema changed on and after year 2015. There is a new field DXVER! Before 2015, all codes are ICD9. After 2015, codes are mixture of ICD-9 and ICD-10]
     - ENROLID: Patient id
     - DX1-DX4: diagnosis codes
     - DXVER: “9” = ICD-9-CM and “0” = ICD-10-CM
     - PROC1: a procedure code
     - PROCTYP: *: ICD-9-CM; 0: ICD-10-CM; 1: CPT; 3: UB92 Revenue Code 6: NABSP; 7: HCPC; 8: CDT (ADA)
     - SVCDATE: Service date for this outpatient visit
