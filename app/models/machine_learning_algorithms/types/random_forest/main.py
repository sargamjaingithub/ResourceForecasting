from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

from app.models.machine_learning_algorithms.machine_learning_algorithm_interface import \
    MachineLearningAlgorithmInterface


class RandomForest(MachineLearningAlgorithmInterface):

    def __init__(self, lag_size: int, prediction_size: int, number_features: int, device: str, **args):
        super().__init__(lag_size, prediction_size, number_features, device)
        self.with_search = 'with_search' in args
        self.prediction_size = prediction_size

        if self.with_search:
            with_search_config = args['with_search']
            self.algorithm_args = with_search_config['default_params']
            search_grid_params = with_search_config['grid_params']
            search_n_iter = with_search_config['n_iter']
            search_n_jobs = with_search_config['n_jobs'] if 'n_jobs' in with_search_config else 1
            search_pre_dispatch = with_search_config['pre_dispatch'] if 'pre_dispatch' in with_search_config else 2
        else:
            self.algorithm_args = args

        if self.prediction_size == 1:  # regression (without discretization)
            self.algorithm = RandomForestRegressor(**self.algorithm_args)
        else:  # classification (with discretization)
            self.algorithm = RandomForestClassifier(**self.algorithm_args)

        if self.with_search:
            self.algorithm = BayesSearchCV(
                self.algorithm,
                search_grid_params,
                optimizer_kwargs={'model_queue_size': 1},
                n_iter=search_n_iter,
                n_jobs=search_n_jobs,
                pre_dispatch=search_pre_dispatch,
                verbose=5
            )

        self.pipeline = make_pipeline(StandardScaler(), self.algorithm)

    def load(self, algorithm_args, x, y):
        if self.prediction_size == 1:  # regression (without discretization)
            self.algorithm = RandomForestRegressor(**algorithm_args)
        else:  # classification (with discretization)
            self.algorithm = RandomForestClassifier(**algorithm_args)
        self.pipeline = make_pipeline(
                StandardScaler(),
                self.algorithm
            )
        self.pipeline.fit(x, y)

    def train(self, x, y):
        self.pipeline.fit(x, y)
        if self.with_search:
            args = {**self.algorithm.best_params_, **self.algorithm_args}
            print("best params: %s" % str(args))
            return args
        else:
            return self.algorithm_args

    def score(self, x, y):
        return self.pipeline.score(x, y)

    def predict(self, x):
        return self.pipeline.predict(x)
