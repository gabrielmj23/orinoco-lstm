import csv

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

# Construir el dataset: cada fila tendrá solo un valor por estación (ayacucho, caicara, ciudad_bolivar, palua)
dataset = []
num_days = min(len(station) for station in data)
num_years = 2025 - 1974 + 1
for year in range(num_years):
    for day in range(num_days):
        row = [station[day][year] for station in data]
        dataset.append(row)

# Mostrar las primeras 5 filas del dataset
for row in dataset[:5]:
    print(row)

# Guardar el dataset en un nuevo archivo CSV con cabeceras y rutas relativas
headers = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]
with open("../data/dataset_combinado.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for row in dataset:
        writer.writerow(row)

print("Dataset combinado guardado en data/dataset_combinado.csv")
