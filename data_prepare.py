import pandas as pd
import numpy as np

excels = [
    [
        "filtered_data\C_Saba_01-07-22Sheet1.xlsx",
        "filtered_data\C_Saba_01-07-22Sheet2.xlsx",
        "filtered_data\C_Saba_01-07-22Sheet3.xlsx"
    ],
    [
        "filtered_data\C_Saba_27-06-22Sheet1.xlsx",
        "filtered_data\C_Saba_27-06-22Sheet2.xlsx",
        "filtered_data\C_Saba_27-06-22Sheet3.xlsx",
    ],
    [
        "filtered_data\C_Saba_28-06-22Sheet1.xlsx",
        "filtered_data\C_Saba_28-06-22Sheet2.xlsx",
        "filtered_data\C_Saba_28-06-22Sheet3.xlsx",
    ],
    [
        "filtered_data\C_Saba_29-06-22Sheet1.xlsx",
        "filtered_data\C_Saba_29-06-22Sheet2.xlsx",
        "filtered_data\C_Saba_29-06-22Sheet3.xlsx",
    ],
    [
        "filtered_data\C_Saba_30-06-22Sheet1.xlsx",
        "filtered_data\C_Saba_30-06-22Sheet2.xlsx",
        "filtered_data\C_Saba_30-06-22Sheet3.xlsx",
    ]
]

green = "filtered_data\ground.xlsx"

green = pd.read_excel(green).to_numpy()

green = green[:, 1]
green = np.expand_dims(green, 1)
saba_total_forces = np.empty((0, 1))
saba_total_pos = np.empty((0, 2))

for file in excels:

    file_pos = pd.read_excel(file[0])
    file_forces = pd.read_excel(file[1])

    file_pos = file_pos.to_numpy()[:60001, 1:]
    file_forces = file_forces.to_numpy()[:60001, 1:]

    for exp in range(file_forces.shape[1]):
        deney_pos = file_pos[:, exp]
        deney_force = file_forces[:, exp]
        
        deney_force = np.expand_dims(deney_force, 1)
        deney_pos = np.expand_dims(deney_pos, 1)
        
        saba_total_forces = np.concatenate((saba_total_forces, deney_force))
        temp = np.concatenate((deney_pos, green), axis = 1)
        saba_total_pos = np.concatenate((saba_total_pos, temp), axis = 0)
total = np.concatenate((saba_total_forces, saba_total_pos), axis = 1)
np.save("data.npy", total, allow_pickle=True)

#force, saba pos, green pos