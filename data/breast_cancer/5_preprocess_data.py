
import SimpleITK as sitk
import os
import torchio as tio
import torch
from tqdm import tqdm
from glob import glob
import shutil
import subprocess
import numpy as np
import time

if __name__ == "__main__":

    patient_ids_3 = torch.load("./final_patient_ids_3_timepoints.pt")
    print(len(patient_ids_3))

    patient_ids_4 = torch.load("./final_patient_ids_4_timepoints.pt")
    print(len(patient_ids_4))

    patient_ids = patient_ids_3 + patient_ids_4
    print(len(patient_ids))
    patient_ids = np.unique(patient_ids)
    patient_ids = list(patient_ids)
    print(len(patient_ids))

    failed_patients = []

    for patient_id in tqdm(patient_ids):

        try: 
            # print(patient_id)

            target_folder = f"/home/johannes/Data/SSD_2.0TB/GNN_pCR/data/breast_cancer/data_processed/{patient_id}/"

            if os.path.exists(target_folder):
                print(f"Patient {patient_id} already processed, skipping.")
                # time.sleep(5)
                continue

            files_dce = sorted(glob(f"/media/johannes/HDD_8.0TB/ISPY2_new/{patient_id}/*DCE*.nii.gz"))
            files_ftv = sorted(glob(f"/media/johannes/HDD_8.0TB/ISPY2_new/{patient_id}/*Analysis*.nii.gz"))

            try:
                assert len(files_dce) == len(files_ftv), f"Number of DCE {len(files_dce)} and FTV {len(files_ftv)} files do not match for patient {patient_id}."
            except:
                print(f"Number of DCE {len(files_dce)} and FTV {len(files_ftv)} files do not match for patient {patient_id}.")
                continue

            for file in files_dce+files_ftv:
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                shutil.copy(file, target_folder)
            
            files_dce = sorted(glob(f"{target_folder}/*DCE.nii.gz"))

            for file in files_dce:
                img = sitk.ReadImage(file)  # 4D image

                size = list(img.GetSize())      # [x, y, z, t]
                index = [0, 0, 0, 0]

                # Extract timepoint 0
                size_tp = size.copy()
                size_tp[3] = 0  # remove time dimension

                index[3] = 0
                img0 = sitk.Extract(img, size_tp, index)

                index[3] = 1
                img1 = sitk.Extract(img, size_tp, index)

                index[3] = 4
                img5 = sitk.Extract(img, size_tp, index)

                sitk.WriteImage(img0, file.replace(".nii.gz", "_timepoint0.nii.gz"))
                sitk.WriteImage(img1, file.replace(".nii.gz", "_timepoint1.nii.gz"))
                sitk.WriteImage(img5, file.replace(".nii.gz", "_timepoint5.nii.gz"))
            
            # files_dce_timepoints = sorted(glob(f"{target_folder}/*DCE*_timepoint*.nii.gz"))

            # fixed_image = files_dce_timepoints[0]  # timepoint0 as fixed image
            
            # for moving_image in files_dce_timepoints:

            #     cmd = [
            #     "./reg_aladin",
            #     "-ref", fixed_image,
            #     "-flo", moving_image,
            #     "-res", moving_image.replace(".nii.gz", "_reg.nii.gz"),
            #     "-platf", "1", 
            #     "-rigOnly"
            #     ]


            #     result = subprocess.run(
            #         cmd,
            #         stdout=subprocess.PIPE,
            #         stderr=subprocess.PIPE,
            #         text=True,
            #     )
        
        except:
            print(f"Error processing patient {patient_id}, skipping.")
            failed_patients.append(patient_id)
            continue
        
for patient_id in failed_patients:
    os.remove(f"/home/johannes/Data/SSD_2.0TB/GNN_pCR/data/breast_cancer/data_processed/{patient_id}/")