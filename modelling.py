import optuna
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class Modelling:
    """
    A class to handle machine learning model initialization and hyperparameter tuning using Optuna.

    Attributes:
        model (Any): The initialized machine learning model.
        available_models (Dict[str, Any]): A dictionary mapping model names to their respective classes.
    """

    def __init__(self):
        """
        Initializes the Modelling class and defines available models for initialization.
        """
        # No model initialized yet
        self.model = None

        # Dictionary to hold available model classes for initialization
        self.available_models = {
            'RandomForestClassifier': RandomForestClassifier, 
            'XGBClassifier': XGBClassifier,
            'LGBMClassifier': LGBMClassifier,
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
        }


    def initialize_model(self, model_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Initializes and returns a machine learning model based on the model_name.

        Args:
            model_name (str): Name of the model to initialize.
            params (Optional[Dict[str, Any]]): Hyperparameters for the model. Defaults to None.

        Returns:
            Any: An instance of the selected model.

        Raises:
            ValueError: If the model_name is not in available_models.
        """

        # Check if the model name is valid
        if model_name not in self.available_models:
            print(f"The {model_name} is not yet available, you can select a following model: {list(self.available_models.keys())}")
            raise ValueError(f"{model_name} is not a valid model.")
        
        # Initialize and return the model with given parameters, or empty dictionary if None
        return self.available_models[model_name](**(params or {}))
    
    def predict_test_set(self, model: Any, X_test: pd.DataFrame) -> np.ndarray:

        """
        Predicts the target variable for the given test set using the provided model.

        Args:
            model (Any): The trained model used for making predictions.
            X_test (pd.DataFrame): The test set features for which predictions are to be made.

        Returns:
            np.ndarray: The predicted values for the test set.
        """
        predictions = model.predict(X_test)
        return predictions

    def fit_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Fits the provided model to the training data.

        Args:
            model (Any): The model to be trained.
            X_train (pd.DataFrame): The training set features.
            y_train (pd.Series): The target variable corresponding to the training set features.

        Returns:
            Any: The fitted model.
        """
        return model.fit(X_train, y_train)

        
    def tune_model(self, model_name: str, X_train: Any, y_train: Any, n_trials: int) -> Dict[str, Any]:
        """
        Tunes the specified model's hyperparameters using Optuna.

        Args:
            model_name (str): Name of the model to tune.
            X_train (Any): Training features.
            y_train (Any): Training labels.
            n_trials (int): Number of trials for hyperparameter tuning.

        Returns:
            Dict[str, Any]: The best parameters found during tuning.

        Raises:
            ValueError: If the model_name is not in available_models.
        """

        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Check if the model name is valid
        if model_name not in self.available_models:
            print(f"The {model_name} is not yet available, you can select a following model: {list(self.available_models.keys())}")
            raise ValueError(f"{model_name} is not a valid model.")

        # Choose the appropriate tuning method based on model name
        if model_name == 'RandomForestClassifier':
            return self.tune_random_forest(X_train, y_train, n_trials)

        elif model_name == 'XGBClassifier':
            return self.tune_xgb(X_train, y_train, n_trials)

        elif model_name == 'LGBMClassifier':
            return self.tune_lgbm(X_train, y_train, n_trials)
        
        elif model_name == 'HistGradientBoostingClassifier':
            return self.tune_hgbc(X_train, y_train, n_trials)
        
        elif model_name =='GradientBoostingClassifier':
            return self.tune_gbc(X_train, y_train, n_trials)

        else:
            raise ValueError(f"{model_name} is not a valid model.")

    def tune_random_forest(self, X_train: Any, y_train: Any, n_trials: int) -> Dict[str, Any]:
        """
        Tunes hyperparameters for the RandomForestClassifier.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            n_trials (int): Number of trials for hyperparameter tuning.

        Returns:
            Dict[str, Any]: The best parameters found during tuning.
        """

        # Define hyperparameters to tune
        def rfc_objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 100)

            # Initialize model with suggested hyperparameters
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                criterion = criterion,
                max_leaf_nodes = max_leaf_nodes,
                random_state=42,
            )

             # Perform cross-validation and return mean accuracy
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        # Create and run an Optuna study
        rfc_study = optuna.create_study(direction="maximize")
        rfc_study.optimize(rfc_objective, n_trials=n_trials)

        print("Best parameters for RandomForestClassifier:", rfc_study.best_params)
        print("Best score for RandomForestClassifier:", rfc_study.best_value)

        return rfc_study.best_params

    def tune_xgb(self, X_train: Any, y_train: Any, n_trials: int) -> Dict[str, Any]:
        """
        Tunes hyperparameters for the XGBClassifier.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            n_trials (int): Number of trials for hyperparameter tuning.

        Returns:
            Dict[str, Any]: The best parameters found during tuning.
        """
        def xgb_objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            gamma = trial.suggest_float("gamma", 0, 5)
            reg_alpha = trial.suggest_float("reg_alpha", 0, 10)
            reg_lambda = trial.suggest_float("reg_lambda", 0, 10)
            scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 1.0)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
            
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
                min_child_weight=min_child_weight,
                random_state=42,
            )

            

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        xgb_study = optuna.create_study(direction="maximize")
        xgb_study.optimize(xgb_objective, n_trials=n_trials)

        print("Best parameters for XGBClassifier:", xgb_study.best_params)
        print("Best score for XGBClassifier:", xgb_study.best_value)

        return xgb_study.best_params

    def tune_lgbm(self, X_train: Any, y_train: Any, n_trials: int) -> Dict[str, Any]:
        """
        Tunes hyperparameters for the LGBMClassifier.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            n_trials (int): Number of trials for hyperparameter tuning.

        Returns:
            Dict[str, Any]: The best parameters found during tuning.
        """
        def lgbm_objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", -1, 15)  # -1 means no limit
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            num_leaves = trial.suggest_int("num_leaves", 2, 256)
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 0, 10)
            reg_lambda = trial.suggest_float("reg_lambda", 0, 10)

            lgbm = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42, 
                verbose=-1
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(lgbm, X_train, y_train, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        lgbm_study = optuna.create_study(direction="maximize")
        lgbm_study.optimize(lgbm_objective, n_trials=n_trials)

        print("Best parameters for LGBMClassifier:", lgbm_study.best_params)
        print("Best score for LGBMClassifier:", lgbm_study.best_value)

        return lgbm_study.best_params
    
    def tune_hgbc(self, X_train: Any, y_train: Any, n_trials: int) -> Dict[str, Any]:
        """
        Tunes hyperparameters for the HGBClassifier.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training labels.
            n_trials (int): Number of trials for hyperparameter tuning.

        Returns:
            Dict[str, Any]: The best parameters found during tuning.
        """

        def hgbc_objective(trial):

            learning_rate =  trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 1000)
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 255)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 200)
            l2_regularization = trial.suggest_float("l2_regularization", 0.0, 10.0)
            max_bins = trial.suggest_int("max_bins", 50, 255)
            early_stopping = trial.suggest_categorical("early_stopping", [True, False])
            scoring = "accuracy"  # Setting scoring to accuracy

            hgbc = HistGradientBoostingClassifier (
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                max_bins=max_bins,
                early_stopping=early_stopping,
                scoring=scoring)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(hgbc, X_train, y_train, cv=cv, scoring="accuracy", n_jobs = -1).mean()
            return score

        hgbc_study = optuna.create_study(direction="maximize")
        hgbc_study.optimize(hgbc_objective, n_trials=n_trials)

        print("Best parameters for HGBClassifier:", hgbc_study.best_params)
        print("Best score for HGBClassifier:", hgbc_study.best_value)

        return hgbc_study.best_params
    
    def tune_gbc(self, X_train: Any, y_train: Any, n_trials: int) -> Dict[str, Any]:
        
        def gbc_objective(trial):
           
           n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
           learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
           max_depth = trial.suggest_int("max_depth", 2, 10)
           subsample = trial.suggest_float("subsample", 0.5, 1.0)
           min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
           min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
           
           model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf)
           
           cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
           score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs = -1).mean()
           return score

        gbc_study = optuna.create_study(direction="maximize")
        gbc_study.optimize(gbc_objective, n_trials=n_trials)

        print("Best parameters for GradientBoostingClassifier:", gbc_study.best_params)
        print("Best score for GradientBoostingClassifier:", gbc_study.best_value)

        return gbc_study.best_params
