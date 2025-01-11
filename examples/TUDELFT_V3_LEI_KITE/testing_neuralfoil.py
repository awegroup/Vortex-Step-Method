import pandas as pd
import matplotlib.pyplot as plt
import neuralfoil as nf
import numpy as np

# ---------------------------
# 1. Read the CSV Data
# ---------------------------
csv_file_path = "corrected_polar_8.csv"
df = pd.read_csv(csv_file_path, header=0)

# ---------------------------
# 2. Obtain NeuralFoil Data
# ---------------------------

# Correct the dat file to desired format
input_file = "/home/jellepoland/ownCloud/phd/code/Vortex-Step-Method/examples/TUDELFT_V3_LEI_KITE/y1.dat"
output_file = "/home/jellepoland/ownCloud/phd/code/Vortex-Step-Method/examples/TUDELFT_V3_LEI_KITE/y1_corrected.dat"
df_airfoil = pd.read_csv(input_file, sep="\t", header=0, index_col=False)
x_values = df_airfoil["       x [m]"].values
y_values = df_airfoil["       y [m]"].values
df_airfoil_corrected = pd.DataFrame({"x": x_values, "y": y_values})
df_airfoil_corrected.to_csv(output_file, sep=" ", header=False, index=False)

# Check the content of y1.dat
# print("Content of y1.dat:")
# with open("y1.dat", "r") as f:
#     print(f.read())

# # Check the content of y1_corrected.dat
# print("Content of y1_corrected.dat:")
# with open("y1_corrected.dat", "r") as f:
#     print(f.read())

dat_file_path = output_file
Re = 5.6e5
model_size = "xxlarge"
alpha_values_deg = np.rad2deg(np.copy(df["alpha"].values))
neuralfoil_alphas = alpha_values_deg


aero = nf.get_aero_from_dat_file(
    filename=dat_file_path, alpha=neuralfoil_alphas, Re=Re, model_size=model_size
)
df_neuralfoil = pd.DataFrame(
    {
        "alpha": neuralfoil_alphas,
        "cl": aero["CL"],
        "cd": aero["CD"],
        "cm": aero["CM"],
    }
)

# ---------------------------
# 3. Combine Corrected and NeuralFoil Data
# ---------------------------
df_combined = df.copy()
df_neuralfoil_expanded = df_neuralfoil.copy()
df_neuralfoil_expanded["cl_breukels"] = np.nan
df_neuralfoil_expanded["cd_breukels"] = np.nan
df_neuralfoil_expanded["cm_breukels"] = np.nan

df_combined = pd.concat([df_combined, df_neuralfoil_expanded], ignore_index=True)

# ---------------------------
# 4. Plotting the Data
# ---------------------------
# plt.style.use("seaborn-darkgrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: CL vs Alpha
df["alpha"] = np.rad2deg(df["alpha"])
ax = axes[0]
ax.plot(df["alpha"], df["cl"], "o-", label="Corrected CL", color="blue")
ax.plot(df["alpha"], df["cl_breukels"], "s--", label="Breukels CL", color="green")
ax.plot(
    df_neuralfoil["alpha"],
    df_neuralfoil["cl"],
    "d-.",
    label="NeuralFoil CL",
    color="red",
)
ax.set_xlabel("Angle of Attack (°)")
ax.set_ylabel("Lift Coefficient (CL)")
ax.set_title("CL vs. Alpha")
ax.legend()
ax.grid(True)

# Subplot 2: CD vs Alpha
ax = axes[1]
ax.plot(df["alpha"], df["cd"], "o-", label="Corrected CD", color="blue")
ax.plot(df["alpha"], df["cd_breukels"], "s--", label="Breukels CD", color="green")
ax.plot(
    df_neuralfoil["alpha"],
    df_neuralfoil["cd"],
    "d-.",
    label="NeuralFoil CD",
    color="red",
)
ax.set_xlabel("Angle of Attack (°)")
ax.set_ylabel("Drag Coefficient (CD)")
ax.set_title("CD vs. Alpha")
ax.legend()
ax.grid(True)

# Subplot 3: CM vs Alpha
ax = axes[2]
ax.plot(df["alpha"], df["cm"], "o-", label="Corrected CM", color="blue")
ax.plot(df["alpha"], df["cm_breukels"], "s--", label="Breukels CM", color="green")
ax.plot(
    df_neuralfoil["alpha"],
    df_neuralfoil["cm"],
    "d-.",
    label="NeuralFoil CM",
    color="red",
)
ax.set_xlabel("Angle of Attack (°)")
ax.set_ylabel("Pitching Moment Coefficient (CM)")
ax.set_title("CM vs. Alpha")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
