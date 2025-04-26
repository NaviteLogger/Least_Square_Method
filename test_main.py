import unittest
import numpy as np
from main import (
    is_symmetric,
    cholesky_decomposition,
    solve_forward_substitution,
    solve_backward_substitution,
    Kacprzak_Marek_MNK,
    NumericalError
)


class TestApproximationFunctions(unittest.TestCase):

    def test_is_symmetric(self) -> None:
        A_sym = np.array([[1, 2], [2, 3]])
        A_non_sym = np.array([[1, 2], [1, 3]])
        self.assertTrue(is_symmetric(A_sym))
        self.assertFalse(is_symmetric(A_non_sym))

    def test_cholesky_decomposition(self) -> None:
        # Test case 1: Standard symmetric positive-definite matrix
        A1 = np.array([[4, 2, 0], [2, 5, 2], [0, 2, 5]], dtype=float)
        L1_expected = np.array([[2, 0, 0], [1, 2, 0], [0, 1, 2]], dtype=float)
        L1_computed = cholesky_decomposition(A1)
        np.testing.assert_allclose(
            L1_computed, L1_expected, atol=1e-8, err_msg="Test Cholesky 1 nie powiódł się")

        # Test case 2: Non-symmetric matrix
        A2 = np.array([[1, 2], [3, 4]])
        with self.assertRaisesRegex(NumericalError, "Macierz nie jest symetryczna"):
            cholesky_decomposition(A2)

        # Test case 3: Not positive-definite matrix
        A3 = np.array([[1, 2], [2, 1]])
        with self.assertRaisesRegex(NumericalError, "Macierz nie jest dodatnio określona"):
            cholesky_decomposition(A3)

        # Test case 4: Non-square matrix
        A4 = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(NumericalError, "Macierz musi być kwadratowa"):
            cholesky_decomposition(A4)

    def test_forward_substitution(self) -> None:
        L = np.array([[2, 0, 0], [1, 2, 0], [0, 1, 2]], dtype=float)
        b = np.array([2, 5, 8], dtype=float)
        z_expected = np.array([1, 2, 3], dtype=float)
        z_computed = solve_forward_substitution(L, b)
        np.testing.assert_allclose(
            z_computed, z_expected, atol=1e-8, err_msg="Test forward substitution nie powiódł się")

        # Test division by zero
        L_zero = np.array([[0, 0], [1, 1]])
        b_zero = np.array([1, 2])
        with self.assertRaisesRegex(NumericalError, "Dzielenie przez bliską zeru wartość"):
            solve_forward_substitution(L_zero, b_zero)

    def test_backward_substitution(self) -> None:
        # L.T from cholesky test case 1
        U = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=float)
        # from forward substitution test case
        z = np.array([1, 2, 3], dtype=float)
        # Solution to A1*a = b where A1 is from Cholesky test 1 and b = L*z = [2, 5, 8]
        # Verified with np.linalg.solve(A1, b)
        a_expected = np.array([0.375, 0.25, 1.5], dtype=float)
        a_computed = solve_backward_substitution(U, z)
        np.testing.assert_allclose(
            a_computed, a_expected, atol=1e-8, err_msg="Test backward substitution nie powiódł się")

        # Test division by zero
        U_zero = np.array([[1, 1], [0, 0]])
        z_zero = np.array([1, 2])
        with self.assertRaisesRegex(NumericalError, "Dzielenie przez bliską zeru wartość"):
            solve_backward_substitution(U_zero, z_zero)

    def test_Kacprzak_Marek_MNK(self):
        # Test case 1: Simple linear fit (n=1)
        x1 = np.array([0, 1, 2, 3])
        y1 = np.array([1, 3, 5, 7])  # Perfect line y = 2x + 1
        a1_expected = np.array([1, 2])  # [a0, a1]
        # Turn off plotting for tests
        a1_computed = Kacprzak_Marek_MNK(x1, y1, 1, plot=False)
        self.assertIsNotNone(
            a1_computed, "MNK zwrócił None dla danych liniowych")
        np.testing.assert_allclose(
            a1_computed, a1_expected, atol=1e-8, err_msg="Test MNK dla prostej nie powiódł się")

        # Test case 2: Quadratic fit (n=2)
        x2 = np.array([-1, 0, 1, 2])
        y2 = np.array([2, 1, 2, 5])  # Perfect parabola y = x^2 + 1
        a2_expected = np.array([1, 0, 1])  # [a0, a1, a2]
        a2_computed = Kacprzak_Marek_MNK(x2, y2, 2, plot=False)
        self.assertIsNotNone(
            a2_computed, "MNK zwrócił None dla danych kwadratowych")
        np.testing.assert_allclose(
            a2_computed, a2_expected, atol=1e-8, err_msg="Test MNK dla paraboli nie powiódł się")

        # Test case 3: Noisy linear data (n=1)
        x3 = np.array([0, 1, 2, 3, 4, 5])
        # Noisy data around y = x + 1
        y3 = np.array([1.1, 1.9, 3.2, 4.1, 4.9, 6.2])
        # Expected values calculated using numpy's polyfit for comparison
        # numpy gives [a1, a0], so reverse
        a3_expected_numpy = np.polyfit(x3, y3, 1)[::-1]
        a3_computed = Kacprzak_Marek_MNK(x3, y3, 1, plot=False)
        self.assertIsNotNone(
            a3_computed, "MNK zwrócił None dla danych zaszumionych")
        np.testing.assert_allclose(a3_computed, a3_expected_numpy, atol=1e-6,
                                   err_msg="Test MNK dla danych zaszumionych nie powiódł się")

        # Test case 4: Insufficient points for degree
        x4 = np.array([0, 1])
        y4 = np.array([0, 1])
        with self.assertRaisesRegex(ValueError, "Liczba punktów .* musi być co najmniej"):
            # n=2 requires at least 3 points
            Kacprzak_Marek_MNK(x4, y4, 2, plot=False)

        # Test case 5: Mismatched input lengths
        x5 = np.array([0, 1, 2])
        y5 = np.array([0, 1])
        with self.assertRaisesRegex(ValueError, "Wektory x i y muszą mieć tę samą długość"):
            Kacprzak_Marek_MNK(x5, y5, 1, plot=False)

        # Test case 6: Invalid degree
        x6 = np.array([0, 1, 2])
        y6 = np.array([0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Stopień wielomianu n musi być nieujemną liczbą całkowitą"):
            Kacprzak_Marek_MNK(x6, y6, -1, plot=False)


# If this script is run directly, execute the unit tests.
if __name__ == '__main__':
    unittest.main()
