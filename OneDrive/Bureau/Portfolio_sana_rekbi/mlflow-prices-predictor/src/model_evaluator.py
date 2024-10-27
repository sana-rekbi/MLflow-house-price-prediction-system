# import logging
# from abc import ABC, abstractmethod

# import statsmodels.api as sm
# from sklearn.metrics import mean_squared_error, r2_score

# # Setup logging configuration
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Abstract Base Class for Model Evaluation Strategy
# # -------------------------------------------------
# # This class defines a common interface for different model evaluation strategies.
# # Subclasses must implement the evaluate_model method.
# class ModelEvaluationStrategy(ABC):
#     @abstractmethod
#     def evaluate_model(self, model, X_test, y_test):
#         """
#         Abstract method to evaluate a model.

#         Parameters:
#         model: The trained model to evaluate.
#         X_test (pd.DataFrame or np.array): The testing data features.
#         y_test (pd.Series or np.array): The testing data labels/target.

#         Returns:
#         dict: A dictionary containing evaluation metrics.
#         """
#         pass


# # Concrete Strategy for Regression Model Evaluation
# # -------------------------------------------------
# # This strategy implements evaluation for regression models.
# class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
#     def evaluate_model(self, model, X_test, y_test):
#         """
#         Evaluates a regression model using R-squared and Mean Squared Error.

#         Parameters:
#         model: The trained regression model to evaluate.
#         X_test (pd.DataFrame or np.array): The testing data features.
#         y_test (pd.Series or np.array): The testing data labels/target.

#         Returns:
#         dict: A dictionary containing R-squared and Mean Squared Error.
#         """
#         logging.info("Adding constant to test data for intercept.")
#         X_test = sm.add_constant(
#             X_test, has_constant="add"
#         )  # Adds a constant term to the test data

#         logging.info("Predicting using the trained model.")
#         y_pred = model.predict(X_test)

#         logging.info("Calculating evaluation metrics.")
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         metrics = {"Mean Squared Error": mse, "R-Squared": r2}

#         logging.info(f"Model Evaluation Metrics: {metrics}")
#         return metrics


# # Context Class for Model Evaluation
# # ----------------------------------
# # This class uses a ModelEvaluationStrategy to evaluate a model.
# class ModelEvaluator:
#     def __init__(self, strategy: ModelEvaluationStrategy):
#         """
#         Initializes the ModelEvaluator with a specific model evaluation strategy.

#         Parameters:
#         strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
#         """
#         self._strategy = strategy

#     def set_strategy(self, strategy: ModelEvaluationStrategy):
#         """
#         Sets a new strategy for the ModelEvaluator.

#         Parameters:
#         strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
#         """
#         logging.info("Switching model evaluation strategy.")
#         self._strategy = strategy

#     def evaluate(self, model, X_test, y_test):
#         """
#         Executes the model evaluation using the current strategy.

#         Parameters:
#         model: The trained model to evaluate.
#         X_test (pd.DataFrame or np.array): The testing data features.
#         y_test (pd.Series or np.array): The testing data labels/target.

#         Returns:
#         dict: A dictionary containing evaluation metrics.
#         """
#         logging.info("Evaluating the model using the selected strategy.")
#         return self._strategy.evaluate_model(model, X_test, y_test)


# # Example usage
# if __name__ == "__main__":
#     # Example trained model and data (replace with actual trained model and data)
#     # model = trained_ols_model
#     # X_test = test_data_features
#     # y_test = test_data_target

#     # Initialize model evaluator with a specific strategy
#     # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
#     # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)

#     pass


import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass


# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluates a regression model using R-squared and Mean Squared Error.

        Parameters:
        model (RegressorMixin): The trained regression model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing R-squared and Mean Squared Error.
        """
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Squared Error": mse, "R-Squared": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass
