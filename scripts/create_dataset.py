import csv
from datetime import datetime, timedelta

# Rutas de los archivos CSV
files = [
    r"../data/ayacucho_raw.csv",
    r"../data/caicara_raw.csv",
    r"../data/ciudad_bolivar_raw.csv",
    r"../data/palua_raw.csv",
]

# Leer los datos de cada archivo
data = []
for file in files:
    with open(file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Saltar cabecera
        station_data = [list(map(lambda x: x.strip(), row)) for row in reader]
        data.append(station_data)

# Construir el dataset: cada fila tendrá fecha, ayacucho, caicara, ciudad_bolivar, palua
dataset = []
start_date = datetime(1974, 1, 1)
current_date = start_date
num_years = 2025 - 1974 + 1
for year in range(num_years):
    year_actual = 1974 + year
    station_rows = [station for station in data]
    num_days = len(station_rows[0])
    day = 0
    while day < num_days:
        date_str = current_date.strftime("%Y/%m/%d")
        values = [station[day][year] for station in data]
        # Si todos los valores están vacíos, no agregues la fila, pero no avances la fecha
        if not any(values):
            day += 1
            continue
        row = [date_str] + values
        dataset.append(row)
        current_date += timedelta(days=1)
        day += 1

# Guardar el dataset en un nuevo archivo CSV con cabeceras y rutas relativas
headers = ["fecha", "ayacucho", "caicara", "ciudad_bolivar", "palua"]
with open("../data/dataset_combinado.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for row in dataset:
        writer.writerow(row)

print("Dataset combinado guardado en data/dataset_combinado.csv")
