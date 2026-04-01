The wing geometry is provided in 2 different formats :

  The first one consists of a list of foils that is lying under FoilsVTK/ folder, each foil being contained in it own .vtk file. 
  The points are sorted starting from TE, walking around the airfoil and coming back to TE. It can be suited for XFoil polar data generation and direct use in VSM.

  The second one consists of a meshed 3D wing in the projected_structure.vtk file. It is a triangular mesh of around 330k points.
  It can be suited for direct use in a 3DPM solver or high fidelity methods.

For an even faster use in the VSM, XFoil polar datas have also been pre-generated for each foil. It can been found inside allPolarsVSM.dat along with LE and TE coordinates. 
Note that the Re used for each foil in the polar generation corresponds to a far-field velocity of 10 m/s in dry air.

Finally, the experimental_data.ods file is a spreadsheet containing H.Belloc's experimental mesurements of lift and drag. This file is provided for comparison purposes.
