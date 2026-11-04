"""Run SurfplanAdapter to produce the geometry files the fancy plot needs.

The interactive "fancy" plot needs physical tube diameters and airfoil profiles
that are not present in a plain VSM aero-geometry yaml. They live in a raw
Surfplan export (a ``.txt`` plus a ``profiles/`` directory). This module drives
SurfplanAdapter to convert that export into VSM-readable yaml files, caching the
result so the conversion only runs when the export changes.
"""

from pathlib import Path
from typing import Optional


def _find_surfplan_txt(surfplan_dir: Path) -> Path:
    """Return the single ``.txt`` export in ``surfplan_dir`` (error if not one)."""
    txt_files = sorted(surfplan_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No Surfplan '.txt' export found in {surfplan_dir}. Expected a raw "
            "Surfplan export (a '<name>.txt' plus a 'profiles/' directory)."
        )
    if len(txt_files) > 1:
        raise ValueError(
            f"Multiple '.txt' files found in {surfplan_dir}: "
            f"{[f.name for f in txt_files]}. Keep only the Surfplan export."
        )
    return txt_files[0]


def ensure_surfplan_processed(
    surfplan_dir: Path, output_dir: Optional[Path] = None
) -> Path:
    """Convert a raw Surfplan export into VSM geometry yamls, with caching.

    Args:
        surfplan_dir (Path): Directory containing the Surfplan ``.txt`` export and
            a ``profiles/`` subdirectory of airfoil ``.dat`` files.
        output_dir (Optional[Path]): Where to write the generated files. Defaults
            to ``surfplan_dir / "vsm_processed"``.

    Returns:
        Path: The directory holding the generated ``aero_geometry.yaml``,
            ``struc_geometry_all_in_surfplan.yaml`` and ``profiles/``.

    Raises:
        ImportError: If SurfplanAdapter is not installed.
        FileNotFoundError: If the export ``.txt`` or ``profiles/`` is missing.
    """
    surfplan_dir = Path(surfplan_dir)
    if output_dir is None:
        output_dir = surfplan_dir / "vsm_processed"
    output_dir = Path(output_dir)

    txt_path = _find_surfplan_txt(surfplan_dir)
    profile_load_dir = surfplan_dir / "profiles"
    if not profile_load_dir.is_dir():
        raise FileNotFoundError(
            f"No 'profiles/' directory in {surfplan_dir}; the fancy plot needs the "
            "airfoil '.dat' files from the Surfplan export."
        )

    aero_yaml = output_dir / "aero_geometry.yaml"
    struc_yaml = output_dir / "struc_geometry_all_in_surfplan.yaml"
    if (
        aero_yaml.exists()
        and struc_yaml.exists()
        and aero_yaml.stat().st_mtime >= txt_path.stat().st_mtime
        and struc_yaml.stat().st_mtime >= txt_path.stat().st_mtime
    ):
        return output_dir

    try:
        from SurfplanAdapter.process_wing import main_process_wing
        from SurfplanAdapter.process_bridle_lines import main_process_bridle_lines
        from SurfplanAdapter.generate_yaml import main_generate_yaml
    except ImportError as error:
        raise ImportError(
            "The fancy plot requires SurfplanAdapter to convert the raw Surfplan "
            "export. Install it with 'pip install SurfplanAdapter'."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    ribs_data = main_process_wing.main(
        surfplan_txt_file_path=txt_path,
        profile_load_dir=profile_load_dir,
        profile_save_dir=output_dir / "profiles",
        is_make_plots=False,
    )
    bridle_lines = main_process_bridle_lines.main(txt_path)
    main_generate_yaml.main(
        ribs_data=ribs_data,
        bridle_lines=bridle_lines,
        yaml_file_path=output_dir / "config_kite.yaml",
    )
    return output_dir
