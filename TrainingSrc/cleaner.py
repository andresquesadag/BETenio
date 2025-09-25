import os
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
filename = askopenfilename()
origin = os.path.basename(filename).replace(".csv", "")
print(filename)
print(origin)

csv = pd.read_csv(filename)
csvFinal = csv[["cuerpo","titular"]].copy()
csvFinal["modelo"] = origin

output = filename.replace(origin, origin + "_filtered")
csvFinal.to_csv(output, index=False)
