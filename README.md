# AI Specialization — Module 1: Math Foundations with NumPy

This repository is the first module in an AI specialization track.
It focuses on foundational linear algebra and NumPy skills through hands-on notebooks.

## Module Goals

By the end of this module, you should be able to:
- Work confidently with NumPy arrays.
- Represent linear systems in matrix form.
- Perform core vector and matrix operations.
- Understand matrix multiplication and linear transformations.
- Visualize simple linear systems in 2D.

## Repository Structure

- `math/` — Jupyter notebooks and images for module exercises.
- `utils.py` — plotting helper used in linear-system labs.
- `requirements.txt` — Python dependencies.

## Notebook Sequence (Suggested)

1. `math/C1_W1_Lab_1_introduction_to_numpy_arrays.ipynb`
2. `math/C1_W1_Lab_2_linear_systems_as_matrices.ipynb`
3. `math/C1W2_UGL_solving_linear_systems_3_variables.ipynb`
4. `math/C1W3_UGL_1_vector_operations.ipynb`
5. `math/C1W3_UGL_2_matrix_multiplication.ipynb`
6. `math/C1W3_UGL_3_linear_transformations.ipynb`

## Setup

### Prerequisites
- Python 3.12+ (recommended)
- `venv`

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Labs

From the project root:

```bash
source .venv/bin/activate
jupyter notebook
```

Then open notebooks in the `math/` folder in order.

## Utility Function

`utils.py` includes `plot_lines(...)`, which plots 2D linear equations in augmented-matrix form and can optionally mark the intersection for 2-equation systems.

## AI Specialization Roadmap

This module is **Module 1** (Math Foundations). Future modules can build on this base (for example: probability, optimization, machine learning fundamentals, and deep learning).

## Notes

- This repository is learning-focused and notebook-centric.
- Some dependencies in `requirements.txt` are broader than strictly required for these notebooks; this is normal for exploratory environments.
