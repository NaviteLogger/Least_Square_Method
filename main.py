import numpy as np
import matplotlib.pyplot as plt

# --- Numerical Functions ---


class NumericalError(Exception):
    """Custom exception for numerical errors in this module."""
    pass


def is_symmetric(A: np.ndarray, tol=1e-8) -> bool:
    """Checks if a matrix A is symmetric."""
    return np.allclose(A, A.T, atol=tol)


def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Performs Cholesky decomposition (L L^T) for a symmetric positive-definite matrix A.

    Args:
        A (np.array): The input matrix.

    Returns:
        np.array: The lower triangular matrix L.

    Raises:
        NumericalError: If the matrix is not square, not symmetric, or not positive-definite.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise NumericalError(
            "Macierz musi być kwadratowa do rozkładu Cholesky'ego.")
    if not is_symmetric(A):
        raise NumericalError(
            "Macierz nie jest symetryczna, nie można zastosować rozkładu Cholesky'ego.")

    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            # L[i,k] * L[j,k] for k in 0..j-1
            sum_k = np.dot(L[i, :j], L[j, :j])

            if i == j:
                diag_val = A[i, i] - sum_k
                # Check for positive definiteness
                if diag_val < 1e-10:  # Use a small tolerance instead of checking > 0
                    raise NumericalError(
                        f"Macierz nie jest dodatnio określona (element diagonalny L[{i},{i}]^2 = {diag_val:.2e} <= 0).")
                L[i, j] = np.sqrt(diag_val)
            else:
                # Check for division by zero (should not happen if positive-definite check is robust)
                if L[j, j] < 1e-10:
                    raise NumericalError(
                        f"Dzielenie przez bliską zeru wartość L[{j},{j}] podczas rozkładu Cholesky'ego.")
                L[i, j] = (A[i, j] - sum_k) / L[j, j]
    return L


def solve_forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves the lower triangular system Lz = b using forward substitution.

    Args:
        L (np.array): Lower triangular matrix.
        b (np.array): Right-hand side vector.

    Returns:
        np.array: Solution vector z.

    Raises:
        NumericalError: If division by zero occurs.
    """
    n = L.shape[0]
    if n != len(b):
        raise ValueError("Niezgodne wymiary macierzy L i wektora b.")
    z = np.zeros(n, dtype=float)
    for i in range(n):
        # Check for division by zero
        if abs(L[i, i]) < 1e-10:
            raise NumericalError(
                f"Dzielenie przez bliską zeru wartość L[{i},{i}] podczas forward substitution.")
        sum_j = np.dot(L[i, :i], z[:i])  # L[i,j] * z[j] for j in 0..i-1
        z[i] = (b[i] - sum_j) / L[i, i]
    return z


def solve_backward_substitution(U: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Solves the upper triangular system Ua = z using backward substitution.

    Args:
        U (np.array): Upper triangular matrix (e.g., L.T).
        z (np.array): Right-hand side vector.

    Returns:
        np.array: Solution vector a.

    Raises:
        NumericalError: If division by zero occurs.
    """
    n = U.shape[0]
    if n != len(z):
        raise ValueError("Niezgodne wymiary macierzy U i wektora z.")
    a = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        # Check for division by zero
        if abs(U[i, i]) < 1e-10:
            raise NumericalError(
                f"Dzielenie przez bliską zeru wartość U[{i},{i}] podczas backward substitution.")
        sum_j = np.dot(U[i, i+1:], a[i+1:])  # U[i,j] * a[j] for j in i+1..n-1
        a[i] = (z[i] - sum_j) / U[i, i]
    return a


def plot_approximation(x_data: np.ndarray, y_data: np.ndarray, coefficients: np.ndarray, title="Aproksymacja średniokwadratowa") -> None:
    """
    Plots the data points and the polynomial approximation.

    Args:
        x_data (np.array): X-coordinates of data points.
        y_data (np.array): Y-coordinates of data points.
        coefficients (np.array): Polynomial coefficients [a0, a1, ..., an].
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Dane punkty')

    x_poly = np.linspace(min(x_data), max(x_data), 100)
    y_poly = np.polyval(coefficients[::-1], x_poly)

    degree = len(coefficients) - 1
    plt.plot(x_poly, y_poly, label=f'Wielomian aproksymujący (st. {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def Kacprzak_Marek_MNK(x: np.ndarray, y: np.ndarray, n: int, plot=True) -> np.ndarray | None:
    """
    Performs least squares polynomial approximation using Cholesky decomposition.

    Args:
        x (list or np.array): X-coordinates of data points.
        y (list or np.array): Y-coordinates of data points.
        n (int): Degree of the approximating polynomial.
        plot (bool): Whether to plot the result.

    Returns:
        np.array: Polynomial coefficients a = [a0, a1, ..., an], or None if an error occurs.
    """
    try:
        # Input validation
        if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
            raise TypeError(
                "Dane wejściowe x i y muszą być listami lub tablicami numpy.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Dane wejściowe x i y muszą być jednowymiarowe.")
        if len(x) != len(y):
            raise ValueError("Wektory x i y muszą mieć tę samą długość.")
        if not isinstance(n, int) or n < 0:
            raise ValueError(
                "Stopień wielomianu n musi być nieujemną liczbą całkowitą.")
        if len(x) < n + 1:
            raise ValueError(
                f"Liczba punktów ({len(x)}) musi być co najmniej n+1 ({n+1}) dla wielomianu stopnia {n}.")

        A = np.vander(x, N=n + 1, increasing=True)

        AtA = A.T @ A
        Aty = A.T @ y

        L = cholesky_decomposition(AtA)

        z = solve_forward_substitution(L, Aty)
        a = solve_backward_substitution(L.T, z)

        if plot:
            plot_approximation(
                x, y, a, title=f'Aproksymacja MNK (st. {n}) - Kacprzak_Marek')

        return a

    except (ValueError, TypeError) as e:
        raise e
    except NumericalError as e:
        print(f"Błąd obliczeń numerycznych w Kacprzak_Marek_MNK: {e}")
        return
    except Exception as e:
        print(f"Nieoczekiwany błąd podczas obliczeń w Kacprzak_Marek_MNK: {e}")
        return None


if __name__ == '__main__':
    print("\nPrzykładowe użycie funkcji Kacprzak_Marek_MNK:")

    x_dane: np.ndarray = [0, 1, -1, 2, -2, 3, -3]  # type: ignore
    y_dane: np.ndarray = [0, 1, 1, 4, 4, -9, -9]  # type: ignore
    stopien: int = 5  # Stopień wielomianu

    print(
        f"\nWywołanie MNK dla danych: x={x_dane}, y={y_dane}, stopień={stopien}")
    wspolczynniki = Kacprzak_Marek_MNK(
        x_dane, y_dane, stopien, plot=True)

    if wspolczynniki is not None:
        print(
            f"\nObliczone współczynniki (a0, a1, ..., an): {np.round(wspolczynniki, 5)}")
    else:
        print("\nNie udało się obliczyć współczynników z powodu błędu.")
