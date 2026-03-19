import json

with open("fluid_simulation.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "from src.config import SimConfig" in source:
            # Update imports
            new_source = source.replace("from src.config import SimConfig", "from src.config import SimConfig, generate_random_config")
            cell["source"] = [s + "\n" for s in new_source.split("\n")]
            # remove trailing empty strings created by split
            if cell["source"][-1] == "\n":
                cell["source"].pop()
        
        elif "config = SimConfig(" in source:
            # We can delete this cell's content or replace it with comment
            cell["source"] = ["# Questa cella usava una config statica fissa. Ora usiamo generate_random_config() nel loop sotto.\n"]
            
        elif "for i in range(1,3):" in source:
            # Replace the loop to use generate_random_config
            new_code = """# ─────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────
torch.manual_seed(42) # Fissa il seed iniziale per riproducibilità tra i run interi del notebook

for i in range(1,3):
    # Genera config dinamica (nuovo seed per gli ostacoli, nuova velocità, ecc.)
    config = generate_random_config()
    
    # Sovrascriviamo eventuali setting grafici o di timestep se usavamo valori ad-hoc nel notebook:
    # config.dt = 0.05
    # config.colormap = "plasma"
    
    save_name = f"sim{i}.npy"
    print(f"\\n---> LANCIANDO SIMULAZIONE {i} <---")
    
    # Eseguiamo passandogli il config univoco appena generato
    run_simulation(config, tensor_name=save_name, num_steps=50)
"""
            cell["source"] = [s + "\n" for s in new_code.split("\n")]
            if cell["source"][-1] == "\n":
                cell["source"].pop()

with open("fluid_simulation.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook modified successfully.")
