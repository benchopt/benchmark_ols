from benchopt import BaseObjective, safe_import_context

with safe_import_context() as ctx:
    import numpy as np


class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "Ordinary Least Squares"

    parameters = {
        'fit_intercept': [False],
    }

    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        diff = self.y - self.X.dot(beta)
        return .5 * diff.dot(diff)

    def get_objective(self):
        return dict(X=self.X, y=self.y, fit_intercept=self.fit_intercept)

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])
