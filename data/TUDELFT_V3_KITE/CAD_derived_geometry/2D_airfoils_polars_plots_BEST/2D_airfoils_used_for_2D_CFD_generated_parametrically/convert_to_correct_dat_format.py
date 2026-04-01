import os
import glob


def convert_dat_file(input_file, output_file):
    """
    Convert a .dat file to the correct format:
    - Remove first two lines
    - Split each line by space
    - Remove third item
    - Save only first two items separated by comma
    - Shift indices to start from trailing edge (1,0) without reordering
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove first two lines
    data_lines = lines[2:]

    # Process each line and extract x,y coordinates
    points = []
    for line in data_lines:
        line = line.strip()
        if line:  # Skip empty lines
            parts = line.split()
            if len(parts) >= 2:  # Ensure we have at least 2 items
                x = float(parts[0])
                y = float(parts[1])
                points.append((x, y))

    if not points:
        print(f"Warning: No valid points found in {input_file}")
        return

    # Find the index of the point closest to trailing edge (1,0)
    trailing_edge_idx = min(
        range(len(points)), key=lambda i: (points[i][0] - 1.0) ** 2 + points[i][1] ** 2
    )

    # Shift the indices to start from trailing edge, maintaining original order
    # This is a circular shift - no reordering, just changing the starting point
    shifted_points = points[trailing_edge_idx:] + points[:trailing_edge_idx]

    # Convert to strings and write
    converted_lines = []
    for x, y in shifted_points:
        converted_line = f"{x:.18e},{y:.18e}\n"
        converted_lines.append(converted_line)

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(converted_lines)


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input directory (uncorrected_format)
    input_dir = os.path.join(script_dir, "uncorrected_format")

    # Output directory (main directory)
    output_dir = script_dir

    # Find all .dat files in the uncorrected_format directory
    dat_files = glob.glob(os.path.join(input_dir, "*.dat"))

    print(f"Found {len(dat_files)} .dat files to convert")

    for dat_file in dat_files:
        # Get the filename without the path
        filename = os.path.basename(dat_file)

        # Create output file path in the main directory
        output_file = os.path.join(output_dir, filename)

        # Convert the file
        convert_dat_file(dat_file, output_file)

        print(f"Converted uncorrected_format/{filename} -> {filename}")

    print("Conversion complete!")


if __name__ == "__main__":
    main()
