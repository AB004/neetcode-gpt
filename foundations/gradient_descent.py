class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x = init
        # for i in range(0,iterations):
        #     x = x - learning_rate*(2*x)
        while iterations>0:
            x = x - learning_rate*(2*x)
            iterations = iterations - 1
        return round(x,5)

