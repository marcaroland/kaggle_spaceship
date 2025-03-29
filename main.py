from modelling import Modelling
from feature_engineering import FeatureEngineering

from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import pandas as pd
from utils import load_data

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

FeatureEngineer = FeatureEngineering()
Modeller = Modelling()

imputed_train = FeatureEngineer.missing_value_imputation(train.drop('Transported', axis=1), 5, ['Spa', 'RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Age'])
processed_train = FeatureEngineer.feature_pipeline(imputed_train)
processed_train = pd.concat([processed_train, pd.DataFrame(train['Transported'].astype(int))], axis=1)

imputed_test = FeatureEngineer.missing_value_imputation(test, 5, ['Spa', 'RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Age'])
processed_test = FeatureEngineer.feature_pipeline(imputed_test)

FEATURES = processed_train.drop('Transported', axis=1).columns
TARGET = ['Transported']

X_train, y_train = FeatureEngineer.data_split(processed_train, features=FEATURES, target=TARGET)
X_train = pd.get_dummies(X_train, columns=['Deck', 'Cabin_part'], dtype=int)
X_train.drop(columns=['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name',  'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck_B', 'Deck_G', 'Deck_T', 'Deck_D', 'Deck_C', 'Deck_A'], inplace=True)
X_train[['PassengerId', 'FAMILY_ID']] = X_train[['PassengerId', 'FAMILY_ID']].astype('int64')


X_test = processed_test.copy()
X_test = pd.get_dummies(X_test, columns=['Deck', 'Cabin_part'], dtype=int)
X_test.drop(columns=['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck_B', 'Deck_G', 'Deck_T', 'Deck_D', 'Deck_C', 'Deck_A'], inplace=True)
X_test['PassengerId'] = X_test['PassengerId'].astype('int64')
X_test['FAMILY_ID'] = X_test['FAMILY_ID'].astype('int64')

gbc_best_optuna_params = Modeller.tune_model('GradientBoostingClassifier', X_train=X_train, y_train=y_train.values.ravel(), n_trials=3)
xgb_best_optuna_params = Modeller.tune_model('XGBClassifier', X_train=X_train, y_train=y_train.values.ravel(), n_trials=3)
rfr_best_optuna_params = Modeller.tune_model('RandomForestClassifier', X_train=X_train, y_train=y_train.values.ravel(), n_trials=3)
lgbm_best_optuna_params = Modeller.tune_model('LGBMClassifier', X_train=X_train, y_train=y_train.values.ravel(), n_trials=3)
hgbc_best_optuna_params = Modeller.tune_model('HistGradientBoostingClassifier', X_train=X_train, y_train=y_train.values.ravel(), n_trials=3)

clf1 = RandomForestClassifier(**rfr_best_optuna_params)
clf2 = LGBMClassifier(**lgbm_best_optuna_params)
clf3 = XGBClassifier(**xgb_best_optuna_params)
clf4 = HistGradientBoostingClassifier(**hgbc_best_optuna_params)
clf5 = GradientBoostingClassifier(**gbc_best_optuna_params)
eclf1 = VotingClassifier(estimators=[('rfr', clf1), ('lgb', clf2), ('xgb', clf3), ('hgbc', clf4), ('sgd', clf5)], voting='soft', weights=[0.75, 0.90, 0.85, 0.95, 0.80]) # rf 80, lgb 81.1, hgb 81.3 xgb 81.1
eclf1.fit(X_train, y_train)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test['Transported'] = Modeller.predict_test_set(model = eclf1, X_test=X_test[X_train.columns])
test['Transported'] = test['Transported'].apply(lambda x: True if x == 1 else False)
submission = test[['PassengerId', 'Transported']]
submission.to_csv("submission.csv", index=False)