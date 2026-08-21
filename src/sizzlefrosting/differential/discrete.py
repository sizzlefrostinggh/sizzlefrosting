"""
Discrete module containing diff function
"""
import array

def diff(t: array.array, x: array.array) -> array.array:
        """
        Compute a discrete derivative of a timeseries data.

        The discrete derivative v(t) of the timeseries is given by:

                v(t) = (x(t_k) - x(t_k-1)) / (t_k - t_k-1)

        Formula: (x[k] - x[k-1]) / (t[k] - t[k-1])

        Args:
                t (array): Time value
                x (array): Signal value

        Returns:
                array: Computed descrete derivative

        Raises:
                ValueError: If array lengths are not equal

        Example:
                >>> diff([0, 1, 2], [10, 12, 18])
                array('f', [2.0, 6.0])
        """
        if len(t) != len(x):
                raise ValueError("Arrays are not equal")

        v = array.array('f', [])

        for k in range(1, len(t)):
                derivative = (x[k] - x[k-1]) / (t[k] - t[k-1])
                v.append(derivative)

        return v
