from benchopt import BaseSolver, safe_import_context


with safe_import_context() as ctx:
    import numpy as np
    from modopt.opt.algorithms import ForwardBackward
    from modopt.opt.gradient import GradBasic
    from modopt.opt.proximity import IdentityProx


class Solver(BaseSolver):
    name = 'modopt-gd'

    install_cmd = 'conda'
    requirements = [
        'pip:modopt'
    ]

    stopping_strategy = 'callback'

    def set_objective(self, X, y, fit_intercept):
        self.X, self.y = X, y

    def run(self, callback):
        x = np.zeros(self.X.shape[1])
        L = np.linalg.norm(self.X, ord=2) ** 2
        self.fb = ForwardBackward(
            x,
            grad=GradBasic(
                input_data=self.y,
                op=lambda w: self.X @ w,
                trans_op=lambda w: self.X.T @ w,
                input_data_writeable=True,
            ),
            prox=IdentityProx(),
            beta_param=1./L,
            auto_iterate=False,
            cost=None,
            progress=False,
        )

        self.fb.iterate(1)
        while callback(self.fb.x_final):
            self.fb.iterate(max_iter=10)

    def get_result(self):
        return self.fb.x_final
