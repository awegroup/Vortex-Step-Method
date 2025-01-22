import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from VSM.plot_styling import set_plot_style


def run_neuralfoil(
    PROJECT_DIR,
):
    import neuralfoil as nf

    Re = 5.6e5
    model_size = "xxlarge"
    dat_file_path = Path(
        PROJECT_DIR,
        "examples",
        "TUDELFT_V3_LEI_KITE",
        "polar_engineering",
        "profiles",
        "y1_corrected.dat",
    )
    alpha_values_deg = np.linspace(-40, 40, 80)
    neuralfoil_alphas = alpha_values_deg

    aero = nf.get_aero_from_dat_file(
        filename=dat_file_path,
        alpha=neuralfoil_alphas,
        Re=Re,
        model_size=model_size,
    )
    df_neuralfoil = pd.DataFrame(
        {
            "alpha": neuralfoil_alphas,
            "cl": aero["CL"],
            "cd": aero["CD"],
            "cm": aero["CM"],
        }
    )
    return df_neuralfoil


def main(n_i, PROJECT_DIR):

    set_plot_style()
    # ---------------------------
    # 1. Read the CSV Data
    # ---------------------------
    csv_file_path = Path(
        PROJECT_DIR,
        "examples",
        "TUDELFT_V3_LEI_KITE",
        "polar_engineering",
        "csv_files",
        f"corrected_polar_{n_i}.csv",
    )
    df = pd.read_csv(csv_file_path, header=0)

    # ---------------------------
    # 2. Obtain NeuralFoil Data
    # ---------------------------

    # Correct the dat file to desired format
    profiles_folder = Path(
        PROJECT_DIR,
        "examples",
        "TUDELFT_V3_LEI_KITE",
        "polar_engineering",
        "profiles",
    )
    input_file = Path(profiles_folder, "y1.dat")
    output_file = Path(profiles_folder, "y1_corrected.dat")
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

    df_neuralfoil = run_neuralfoil(PROJECT_DIR)
    # ---------------------------
    # 3. Combine Corrected and NeuralFoil Data
    # ---------------------------
    df_combined = df.copy()
    df_neuralfoil_expanded = df_neuralfoil.copy()
    df_neuralfoil_expanded["cl_breukels"] = np.nan
    df_neuralfoil_expanded["cd_breukels"] = np.nan
    df_neuralfoil_expanded["cm_breukels"] = np.nan

    df_combined = pd.concat([df_combined, df_neuralfoil_expanded], ignore_index=True)

    ### Adding flat-plate aerodynamics to it
    cl_list = []
    cd_list = []
    cm_list = []
    for alpha in df["alpha"].values:
        # alpha = np.deg2rad(alpha)
        if alpha < -10:
            cl = 2 * ((np.sin(alpha)) ** 2) * np.cos(alpha)
            cd = 2 * ((np.sin(alpha)) ** 3)
            cm = 0.5 * ((np.sin(alpha)) ** 2)
        elif alpha > -10 and alpha < 15:
            cl = 2 * np.sin(alpha)
            cd = 2 * np.sin(alpha) ** 2
            cm = -(np.pi / 2) * alpha
        else:
            cl = 2 * ((np.sin(alpha)) ** 2) * np.cos(alpha)
            cd = 2 * ((np.sin(alpha)) ** 3)
            cm = 0.5 * ((np.sin(alpha)) ** 2)

        # print(f"alpha: {alpha}, cl: {cl}, cd: {cd}, cm: {cm}")
        cl_list.append(cl)
        cd_list.append(cd)
        cm_list.append(cm)

    df_flat_plate = pd.DataFrame(
        {
            "alpha": np.rad2deg(df["alpha"].values),
            "cl": cl_list,
            "cd": cd_list,
            "cm": cm_list,
        }
    )

    ### Saving the dfs to the profiles folder
    df["alpha"] = np.rad2deg(df["alpha"])
    df_corrected = df.copy()
    df_corrected.to_csv(Path(profiles_folder, f"df_corrected_{n_i}.csv"), index=False)
    df_neuralfoil.to_csv(Path(profiles_folder, f"df_neuralfoil_{n_i}.csv"), index=False)
    df_flat_plate.to_csv(Path(profiles_folder, f"df_flat_plate_{n_i}.csv"), index=False)
    df_breukels.to_csv(Path(profiles_folder, f"df_breukels_{n_i}.csv"), index=False)

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
    ax.plot(
        df_flat_plate["alpha"],
        df_flat_plate["cl"],
        "x--",
        label="Flat Plate CL",
        color="black",
    )
    ax.set_xlabel(r"$\alpha$ [°]")
    ax.set_ylabel("Lift Coefficient (CL)")
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
    ax.plot(
        df_flat_plate["alpha"],
        df_flat_plate["cd"],
        "x--",
        label="Flat Plate CD",
        color="black",
    )
    ax.set_xlabel(r"$\alpha$ [°]")
    ax.set_ylabel("Drag Coefficient (CD)")
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
    ax.plot(
        df_flat_plate["alpha"],
        df_flat_plate["cm"],
        "x--",
        label="Flat Plate CM",
        color="black",
    )
    ax.set_xlabel(r"$\alpha$ [°]")
    ax.set_ylabel("Pitching Moment Coefficient (CM)")
    ax.set_ylim(-0.5, 0.5)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(Path(profiles_folder, f"polar_neural_foil_comparison_{n_i}.pdf"))
    # plt.show()
    plt.close()


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    panel_i = 18
    main(
        n_i=panel_i,
        PROJECT_DIR=PROJECT_DIR,
    )
