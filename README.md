# Vortex Step Method
Implementation of the Vortex Step Method for a static wing shape.

## Installation Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/ocayon/Vortex-Step-Method
    ```

2. Navigate to the repository folder:
    ```bash
    cd Vortex-Step-Method
    ```
    
3. Create a virtual environment:
   
   Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
5. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows
    ```bash
    .\venv\Scripts\activate
    ```

6. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```

7. To deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Contributing Guide
We welcome contributions to this project! Whether you're reporting a bug, suggesting a feature, or writing code, here’s how you can contribute:

1. Create an issue on GitHub
2. Create a branch from this issue
   ```bash
   git checkout -b issue_number-new-feature
   ```
3. --- Implement your new feature---
4. Verify nothing broke using pytest
```
  pytest
```
5. Commit your changes with a descriptive message
```
  git commit -m "#<number> <message>"
```
6. Push your changes to the github repo:
   git push origin branch-name
   
7. Create a pull-request, with `base:develop`, to merge this feature branch
8. Once the pull request has been accepted, close the issue


### Code Style and Guidelines
- Follow PEP 8 for Python code style.
- Ensure that your code is well-documented.
- Write unit tests for new features and bug fixes.

## Citation
If you use this project in your research, please consider citing it. Citation details can be found in the `CITATION.cff` file included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Copyright
Copyright (c) 2022 Oriol Cayon
Copyright (c) 2024 Oriol Cayon, Jelle Poland, TU Delft
