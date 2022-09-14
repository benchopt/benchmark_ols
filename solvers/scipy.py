from benchopt import BaseSolver, safe_import_context


with safe_import_context() as ctx:
    from scipy.sparse.linalg import cgs, gmres, tfqmr


class Solver(BaseSolver):
    name = "scipy"

    requirements = ["scipy>=1.8"]
    install_cmd = "conda"
    parameters = {"solver": ["cgs", "gmres", "tfqmr"]}

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
        elif self.solver == "gmres":
            algo = gmres
        elif self.solver == "tfqmr":
            algo = tfqmr
        else:
            raise ValueError(f"Unknown solver {self.solver}")

        x, _ = algo(X.T @ X, X.T @ y, maxiter=n_iter, tol=1e-12)
        self.w = x

    def get_result(self):
        return self.w
