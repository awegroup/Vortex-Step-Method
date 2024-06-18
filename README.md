# Vortex_step_method
Implementation of the Vortex Step Method for a static wing shape

# functions_VSM_LLT.py

Contains all the necessary functions to solve the aerodynamics of the wing with both the Vortex Step Method [1,2] and the classic Lifting Line Theory.

# main.py

Script to solve for one angle of attack
INPUTS:
- Coordinates of the wing (look at the rectangular wing example). 
The coordinates are defined so that the first point is the LE position of the first section, the second point the TE of the first section and so on. 
- Convergence criteria
- Velocity magnitude and direction
- Number of filaments per horseshoe (default should be 5)
- Model : 'VSM' or 'LLT'
- Airfoil polars

# References

[1] Cayon, O. Fast aeroelastic model of a leading-edge inflatable kite for the design phase of airborne wind energy systems. Master’s
thesis, Delft University of Technology, 2022. http://resolver.tudelft.nl/uuid:aede2a25-4776-473a-8a75-fb6b17b1a690

[2] Maximilian Ranneberg: "Direct Wing Design and Inverse Airfoil Identification with the Nonlinear Weissinger Method". arXiv:1501.04983 [physics.flu-dyn], June 2015. https://arxiv.org/abs/1501.04983

[3] Rick Damiani, Fabian F. Wendt, Jason M. Jonkman, Jerome Sicard: "A Vortex Step Method for Nonlinear Airfoil Polar Data as Implemented in KiteAeroDyn". AIAA Scitech 2019 Forum, San Diego, California, 7-11 January 2019. https://doi.org/10.2514/6.2019-0804

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Copyright

Copyright (c) 2022 Oriol Cayon
