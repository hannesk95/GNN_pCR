import time
import pandas as pd
from tcia_utils import nbia
from tqdm import tqdm

if __name__ == "__main__":

    df = pd.read_excel("/home/johannes/Data/SSD_2.0TB/GNN_pCR/data/breast_cancer/ISPY2-Cohort1-inclu-ACRIN6698-full-manifes-nbia-digest.xlsx")
    df = df[df["Patient ID"].str.contains("ISPY2")]
    df = df[df["Series Description"].str.contains("DCE") | df["Modality"].str.contains("SEG")]
    df = df.sort_values(by=["Patient ID", "Study Description"])

    patients_with_3_timepoints = []
    patients_with_4_timepoints = []

    for patient in tqdm(df["Patient ID"].unique()):
        t0_dce = 0
        t0_seg = 0
        
        t1_dce = 0
        t1_seg = 0
        
        t2_dce = 0
        t2_seg = 0
        
        t3_dce = 0
        t3_seg = 0

        patient_df = df[df["Patient ID"] == patient]

        for index, row in patient_df.iterrows():
            if "T0" in row["Study Description"]:
                if "DCE" in row["Series Description"]:
                    t0_dce += 1
                if row["Modality"] == "SEG":
                    t0_seg += 1
                    
            if "T1" in row["Study Description"]:
                if "DCE" in row["Series Description"]:
                    t1_dce += 1
                if row["Modality"] == "SEG":
                    t1_seg += 1
                    
            if "T2" in row["Study Description"]:
                if "DCE" in row["Series Description"]:
                    t2_dce += 1
                if row["Modality"] == "SEG":
                    t2_seg += 1
                    
            if "T3" in row["Study Description"]:
                if "DCE" in row["Series Description"]:
                    t3_dce += 1
                if row["Modality"] == "SEG":
                    t3_seg += 1

        if t0_dce + t0_seg + t1_dce + t1_seg + t2_dce + t2_seg == 6:
            patients_with_3_timepoints.append(patient)
        
        if t0_dce + t0_seg + t1_dce + t1_seg + t2_dce + t2_seg + t3_dce + t3_seg == 8:
            patients_with_4_timepoints.append(patient)    

    print("Patients with 3 timepoints:", len(patients_with_3_timepoints))
    print("Patients with 4 timepoints:", len(patients_with_4_timepoints))

    df_final = df[df["Patient ID"].isin(patients_with_3_timepoints)]

    for attempt in range(5):
        try:
            nbia.downloadSeries(df_final["Series Instance UID"].tolist(), input_type="list", path="/media/johannes/HDD_8.0TB/ISPY2_new/")        
            break

        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            time.sleep(30)
    