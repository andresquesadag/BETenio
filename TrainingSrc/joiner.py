import random
import pandas as pd
import sys
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from sklearn.model_selection import train_test_split

def checkFiles(fileList):
    result = True
    for i in fileList:
        if not os.path.isfile(i):
            print(i + " doesnt exist")
            result = False
    return result

def split_csv(filepath, train_ratio = 0.8,):
    df = pd.read_csv(filepath)
    random_state = random.randint(0, 4294967295)
    train_df, test_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state,
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def create_csv(dataframes, output):

    final_data_frame = pd.DataFrame()
    
    for df in dataframes:
        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Check if 'modelo' column exists
        if 'modelo' not in df_copy.columns:
            print(f"Warning: 'modelo' column not found in DataFrame. Skipping...")
            continue
        
        # Check if any value in 'modelo' column contains "Real"
        has_real = df_copy['modelo'].astype(str).str.contains('Real', case=False, na=False).any()
        
        if has_real:
            # If any row has "Real" in modelo, set all rows to "Humano"
            df_copy['authorship'] = 'Humano'
        else:
            # If no row has "Real" in modelo, set all rows to "IA"
            df_copy['authorship'] = 'IA'
        
        # Append to final DataFrame
        final_data_frame = pd.concat([final_data_frame, df_copy], ignore_index=True)
    
    final_data_frame = final_data_frame.sample(frac=1)
    final_data_frame.to_csv(output, index=False)
    print(f"Successfully created CSV file: {output}")
    print(f"Total rows: {len(final_data_frame)}")
    print(f"Authorship distribution:")
    print(final_data_frame['authorship'].value_counts())


files = ["Reales_filtered.csv", "Claude_filtered.csv", "Deepseek_filtered.csv", "GPT_filtered.csv"]
Tk().withdraw()
folderPath = askdirectory()
for i in range(len(files)):
    files[i] = folderPath + "/" + files[i]

if not checkFiles(fileList= files):
    print("Files dont exist")
else:
    print("All files found")
    random.seed(a=None)
    train_data = []
    test_data = []
    for i in files:
        current_train, current_test = split_csv(i)
        train_data.append(current_train)
        test_data.append(current_test)
    print("Train dataset created:")
    create_csv(train_data, folderPath + "/train_dataset.csv")
    print("Test dataset created:")
    create_csv(test_data, folderPath + "/test_dataset.csv")


