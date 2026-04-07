import pandas as pd
import matplotlib.pyplot as plt
import glob

# Szukamy wszystkich plików Bledy_*.txt lub .csv
files = glob.glob("Bledy_*.txt")

for file in files:
    data = pd.read_csv(file)
    unit_num = file.split('_')[1].split('.')[0]
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['Epoch'], data['Error Value'], label=f'Unit {unit_num}')
    plt.title(f'Wykres błędu - Cyfra {unit_num}')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'wykres_{unit_num}.png')
    plt.show()