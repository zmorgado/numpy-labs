import numpy as np
import matplotlib.pyplot as plt


def plot_lines(A_system, x1_range=(-10, 10), num_points=400, show_solution=True, ax=None):
    """Plot 2D lines for a linear system represented as an augmented matrix.

    Parameters
    ----------
    A_system : array-like, shape (m, 3)
        Each row represents an equation of the form a*x1 + b*x2 = c.
    x1_range : tuple(float, float)
        Range of x1 values to plot.
    num_points : int
        Number of points in the x1 grid.
    show_solution : bool
        If True and there are exactly 2 equations, attempt to plot their intersection.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axes. Otherwise creates a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """

    system = np.asarray(A_system, dtype=float)
    if system.ndim != 2 or system.shape[1] != 3:
        raise ValueError(
            "A_system must be a 2D array with shape (m, 3) representing [a, b, c] rows"
        )

    created_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
        created_ax = True

    x1 = np.linspace(float(x1_range[0]), float(x1_range[1]), int(num_points))
    eps = 1e-12

    for idx, (a, b, c) in enumerate(system):
        label = f"Eq {idx + 1}: {a:.2g}·x1 + {b:.2g}·x2 = {c:.2g}"

        if abs(b) > eps:
            x2 = (c - a * x1) / b
            ax.plot(x1, x2, label=label)
        else:
            if abs(a) <= eps:
                # Degenerate equation: 0*x1 + 0*x2 = c
                # Nothing meaningful to plot.
                continue
            x1_const = c / a
            ax.axvline(x=x1_const, label=label)

    if show_solution and system.shape[0] == 2:
        A = system[:, :2]
        b_vec = system[:, 2]
        try:
            sol = np.linalg.solve(A, b_vec)
            ax.plot(sol[0], sol[1], "ro", label=f"Solution ({sol[0]:.3g}, {sol[1]:.3g})")
        except np.linalg.LinAlgError:
            # Singular / no unique solution (parallel or coincident lines)
            pass

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(float(x1_range[0]), float(x1_range[1]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    if created_ax:
        plt.show()

    return ax
