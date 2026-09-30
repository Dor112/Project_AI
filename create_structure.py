from pathlib import Path

dirs = [
    "NeuroLab/app/ui",
    "NeuroLab/app/ml",
    "NeuroLab/app/visualization",
    "NeuroLab/assets/data",
    "NeuroLab/assets/models",
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"Created: {d}")

print("\nСтруктура создана!")