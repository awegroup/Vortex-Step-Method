import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from collections import OrderedDict
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Read the wing geometry CSV file.
wing_geometry_path = (
    Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "wing_geometry.csv"
)
wing_geometry = pd.read_csv(wing_geometry_path)

# Define the directory where polar CSV files are stored.
airfoil_dir = (
    Path(PROJECT_DIR)
    / "examples"
    / "TUDELFT_V3_KITE"
    / "polar_engineering"
    / "csv_files"
)

# Define the directory where YAML files will be saved.
save_dir = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / "airfoils"
save_dir.mkdir(parents=True, exist_ok=True)

kite_name = "TUDELFT_V3_KITE"


# Custom YAML Dumper that forces explicit type tags.
class ExplicitTagDumper(yaml.Dumper):
    def represent_str(self, data):
        return self.represent_scalar("tag:yaml.org,2002:str", data)

    def represent_int(self, data):
        return self.represent_scalar("tag:yaml.org,2002:int", str(data))

    def represent_float(self, data):
        return self.represent_scalar("tag:yaml.org,2002:float", str(data))

    def represent_bool(self, data):
        return self.represent_scalar(
            "tag:yaml.org,2002:bool", "true" if data else "false"
        )


# Override list representation to force inline (flow) style.
def represent_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


ExplicitTagDumper.add_representer(str, ExplicitTagDumper.represent_str)
ExplicitTagDumper.add_representer(int, ExplicitTagDumper.represent_int)
ExplicitTagDumper.add_representer(float, ExplicitTagDumper.represent_float)
ExplicitTagDumper.add_representer(bool, ExplicitTagDumper.represent_bool)
ExplicitTagDumper.add_representer(list, represent_list)


# Function to properly format YAML output with explicit tags
def manual_yaml_generation(data_dict):
    # Initialize the YAML string with proper indentation
    yaml_lines = []

    # Add key-value pairs with explicit type tags
    yaml_lines.append(f"kite_name: !!str \"{data_dict['kite_name']}\"")
    yaml_lines.append(f"rib_number: !!int {data_dict['rib_number']}")
    yaml_lines.append(
        f"is_strut: !!bool {'true' if data_dict['is_strut'] else 'false'}"
    )
    yaml_lines.append(f"d_tube: !!float {data_dict['d_tube']}")
    yaml_lines.append(f"x_camber: !!float {data_dict['x_camber']}")
    yaml_lines.append(f"y_camber: !!float {data_dict['y_camber']}")
    yaml_lines.append(f"delta_te_angle: !!float {data_dict['delta_te_angle']}")

    # Add airfoil_2D section with empty sequences
    yaml_lines.append("airfoil_2D:")
    yaml_lines.append("  x_2D: !!seq []")
    yaml_lines.append("  y_2D: !!seq []")

    # Add 3D vectors with explicit tags for each float
    le = data_dict["leading_edge_3D"]
    te = data_dict["trailing_edge_3D"]
    nv = data_dict["normal_vector"]

    yaml_lines.append(
        f"leading_edge_3D: !!seq [!!float {le[0]}, !!float {le[1]}, !!float {le[2]}]"
    )
    yaml_lines.append(
        f"trailing_edge_3D: !!seq [!!float {te[0]}, !!float {te[1]}, !!float {te[2]}]"
    )
    yaml_lines.append(
        f"normal_vector: !!seq [!!float {nv[0]}, !!float {nv[1]}, !!float {nv[2]}]"
    )

    # Add polar data with explicit tags
    yaml_lines.append("polar_data:")

    # Format each list with explicit tags
    for key in ["alpha", "cl", "cd", "cm"]:
        values = data_dict["polar_data"][key]
        if not values:  # If empty
            yaml_lines.append(f"  {key}: !!seq []")
        else:
            # Format each element with the proper spacing and alignment
            values_str = ", ".join([str(val) for val in values])
            yaml_lines.append(f"  {key}: !!seq [{values_str}]")

    return "\n".join(yaml_lines)


# Loop over the polar CSV files
for i in range(1, 3):
    print(f"Processing rib number: {i}")
    airfoil_path = airfoil_dir / f"corrected_polar_{i}.csv"
    save_yaml_path = save_dir / f"{i}.yaml"

    # Read the polar CSV.
    polar_df = pd.read_csv(airfoil_path)
    alpha = polar_df["alpha"].to_numpy()
    cl = polar_df["cl_new"].to_numpy()
    cd = polar_df["cd_new"].to_numpy()
    cm = polar_df["cm_new"].to_numpy()

    rib_number = i

    # Use index i+1 for wing geometry (adjust if necessary)
    idx = i + 1
    le_x = wing_geometry["LE_x"].to_numpy()[idx]
    le_y = wing_geometry["LE_y"].to_numpy()[idx]
    le_z = wing_geometry["LE_z"].to_numpy()[idx]
    te_x = wing_geometry["TE_x"].to_numpy()[idx]
    te_y = wing_geometry["TE_y"].to_numpy()[idx]
    te_z = wing_geometry["TE_z"].to_numpy()[idx]
    d_tube = wing_geometry["d_tube"].to_numpy()[idx]
    camber = wing_geometry["camber"].to_numpy()[idx]
    # Default (base) values for these entries.
    delta_te_angle = 0.0
    y_camber = 0.0

    LE = np.array([le_x, le_y, le_z])
    TE = np.array([te_x, te_y, te_z])

    # Create a dictionary with all the data
    data_to_save = {
        "kite_name": kite_name,
        "rib_number": rib_number,
        "is_strut": False,
        "d_tube": float(d_tube),
        "x_camber": float(camber),
        "y_camber": float(y_camber),
        "delta_te_angle": float(delta_te_angle),
        "airfoil_2D": {"x_2D": [], "y_2D": []},
        "leading_edge_3D": LE.tolist(),
        "trailing_edge_3D": TE.tolist(),
        "normal_vector": [0.0, 1.0, 0.0],
        "polar_data": {
            "alpha": alpha.tolist(),
            "cl": cl.tolist(),
            "cd": cd.tolist(),
            "cm": cm.tolist(),
        },
    }

    # Generate the YAML manually to ensure exact formatting
    formatted_yaml = manual_yaml_generation(data_to_save)

    # Write the formatted YAML to file
    with open(save_yaml_path, "w") as f:
        f.write(formatted_yaml)

    print(f"Saved YAML file to: {save_yaml_path}")
