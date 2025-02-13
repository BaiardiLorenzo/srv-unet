import pandas as pd
import matplotlib.pyplot as plt

# Lettura delle tabelle da file CSV
quality_data_540 = pd.read_csv("./results/540_models_fix.csv")
quality_data_720 = pd.read_csv("./results/720_models.csv")
bitrate_data = pd.read_csv("./results/bitrate_values.csv")

# Filtra i dati per la risoluzione 540 e 720
bitrate_540 = bitrate_data[bitrate_data["resolution"] == 540]
bitrate_720 = bitrate_data[bitrate_data["resolution"] == 720]

# Calcolo della VMAF-NEG corretta
quality_data_540["distortion"] = 100 - quality_data_540["vmaf-neg"]
quality_data_720["distortion"] = 100 - quality_data_720["vmaf-neg"]

quality_data_540["distortion-vmaf"] = 100 - quality_data_540["vmaf"]
quality_data_720["distortion-vmaf"] = 100 - quality_data_720["vmaf"]

# Add 540p and 720p in model name
quality_data_540["models"] = quality_data_540["models"] + " (540p)"
quality_data_720["models"] = quality_data_720["models"] + " (720p)"

# Associa i valori di bitrate ai CRF corrispondenti
bitrate_map_540 = dict(zip(bitrate_540["crf"], bitrate_540["avg_bitrate"]))
bitrate_map_720 = dict(zip(bitrate_720["crf"], bitrate_720["avg_bitrate"]))

quality_data_540["bitrate"] = quality_data_540["CRF"].map(bitrate_map_540)
quality_data_720["bitrate"] = quality_data_720["CRF"].map(bitrate_map_720)

# Raggruppa i dati per modello per 540p
grouped_data_540 = quality_data_540.groupby("models").apply(
    lambda x: {
        "crf_values": x["CRF"].tolist(),
        "distortion": x["distortion"].tolist(),
        "distortion-vmaf": x["distortion-vmaf"].tolist(),
        "bitrate": x["bitrate"].tolist(),
        "lpips": x["lpips"].tolist(),
        "vmaf": x["vmaf"].tolist(),
    }
).to_dict()

# Raggruppa i dati per modello per 720p
grouped_data_720 = quality_data_720.groupby("models").apply(
    lambda x: {
        "crf_values": x["CRF"].tolist(),
        "distortion": x["distortion"].tolist(),
        "distortion-vmaf": x["distortion-vmaf"].tolist(),
        "bitrate": x["bitrate"].tolist(),
        "lpips": x["lpips"].tolist(),
        "vmaf": x["vmaf"].tolist(),
    }
).to_dict()

# Funzione per annotare i punti con i valori di CRF
def annotate_points(ax, data):
    for i, crf in enumerate(data["crf_values"]):
        ax.annotate("CRF "+str(crf), (data["distortion"][i], data["bitrate"][i]), textcoords="offset points", xytext=(0, 10), ha='center')

# Creazione del grafico per 540p
fig, ax = plt.subplots(figsize=(12, 8))
for model, data in grouped_data_540.items():
    ax.scatter(data["distortion"], data["bitrate"], label=model)
    ax.plot(data["distortion"], data["bitrate"])
    # annotate_points(ax, data)

# Personalizzazioni del grafico
ax.set_title("Bitrate vs Distortion (540p)", fontsize=16)
ax.set_xlabel("Distortion (100 - VMAF-NEG)", fontsize=14)
ax.set_ylabel("Bitrate (kbps)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(title="Models", fontsize=12)
plt.savefig("bitrate_vs_distortion_540p.png")
plt.close()

# Creazione del grafico per 720p
fig, ax = plt.subplots(figsize=(12, 8))
for model, data in grouped_data_720.items():
    ax.scatter(data["distortion"], data["bitrate"], label=model)
    ax.plot(data["distortion"], data["bitrate"])
    # annotate_points(ax, data)

# Personalizzazioni del grafico
ax.set_title("Bitrate vs Distortion (720p)", fontsize=16)
ax.set_xlabel("Distortion (100 - VMAF-NEG)", fontsize=14)
ax.set_ylabel("Bitrate (kbps)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(title="Models", fontsize=12)
plt.savefig("bitrate_vs_distortion_720p.png")
plt.close()

# Creazione del grafico per LPIPS e VMAF per 540p
fig, ax = plt.subplots(figsize=(12, 8))
for model, data in grouped_data_540.items():
    ax.scatter(data["lpips"], data["bitrate"], label=model)
    ax.plot(data["lpips"], data["bitrate"])
    # annotate_points(ax, data)

# Personalizzazioni del grafico
ax.set_title("Bitrate vs LPIPS (540p)", fontsize=16)
ax.set_xlabel("LPIPS", fontsize=14)
ax.set_ylabel("Bitrate (kbps)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(title="Models", fontsize=12)
plt.savefig("bitrate_vs_lpips_540p.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
for model, data in grouped_data_540.items():
    ax.scatter(data["distortion-vmaf"], data["bitrate"], label=model)
    ax.plot(data["distortion-vmaf"], data["bitrate"])
    # annotate_points(ax, data)

# Personalizzazioni del grafico
ax.set_title("Bitrate vs Distortion (540p)", fontsize=16)
ax.set_xlabel("Distortion (100 - VMAF)", fontsize=14)
ax.set_ylabel("Bitrate (kbps)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(title="Models", fontsize=12)
plt.savefig("bitrate_vs_vmaf_540p.png")
plt.close()

# Creazione del grafico per LPIPS e VMAF per 720p
fig, ax = plt.subplots(figsize=(12, 8))
for model, data in grouped_data_720.items():
    ax.scatter(data["lpips"], data["bitrate"], label=model)
    ax.plot(data["lpips"], data["bitrate"])
    # annotate_points(ax, data)

# Personalizzazioni del grafico
ax.set_title("Bitrate vs LPIPS (720p)", fontsize=16)
ax.set_xlabel("LPIPS", fontsize=14)
ax.set_ylabel("Bitrate (kbps)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(title="Models", fontsize=12)
plt.savefig("bitrate_vs_lpips_720p.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
for model, data in grouped_data_720.items():
    ax.scatter(data["distortion-vmaf"], data["bitrate"], label=model)
    ax.plot(data["distortion-vmaf"], data["bitrate"])
    # annotate_points(ax, data)

# Personalizzazioni del grafico
ax.set_title("Bitrate vs Distortion (720p)", fontsize=16)
ax.set_xlabel("Distortion (100 - VMAF)", fontsize=14)
ax.set_ylabel("Bitrate (kbps)", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(title="Models", fontsize=12)
plt.savefig("bitrate_vs_vmaf_720p.png")
plt.close()