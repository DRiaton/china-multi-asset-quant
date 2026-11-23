
import cvxpy as cp
import numpy as np


def solve(problem: cp.Problem, variable: cp.Variable, *, allow_infeasible=False):
    """Return solver-certified weights, surfacing genuine formulation failures."""
    problem.solve(solver=cp.CLARABEL)
    if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE} and allow_infeasible:
        return None
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or variable.value is None:
        raise RuntimeError(f"Portfolio solver failed: {problem.status}")
    weights = np.maximum(np.asarray(variable.value).ravel(), 0)
    weights[weights < 1e-9] = 0
    return weights / weights.sum()
