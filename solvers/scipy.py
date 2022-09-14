from benchopt import BaseSolver, safe_import_context


with safe_import_context() as ctx:
    from scipy.sparse.linalg import cgs


class Solver(BaseSolver):
    name = "scipy"

    requirements = ["scipy"]
    install_cmd = "conda"
    parameters = {"solver": ["cgs"]}

    def skip(self, X, y, fit_intercept):
        if fit_intercept:
            return True, "scipy does not support fit_intercept"
        return False, None

    def set_objective(self, X, y, fit_intercept=False):
        self.X, self.y = X, y
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        X, y = self.X, self.y
        if self.solver == "cgs":
            algo = cgs
        else:
            raise ValueError(f"Unknown solver {self.solver}")

        x, _ = algo(X.T @ X, X.T @ y, maxiter=n_iter)
        self.w = x

    def get_result(self):
        return self.w
