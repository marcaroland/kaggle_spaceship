import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

class FeatureEngineering:
    
    def ship_related_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Splits 'Cabin' into 'Deck' and 'Cabin_part'.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing the 'Cabin' column.

        Returns:
            pd.DataFrame: The updated DataFrame with 'Deck' and 'Cabin_part' columns.
        """

        # Split 'Cabin' column by '/' and assign first and third parts to 'Deck' and 'Cabin_part'
        data[['Deck', 'Cabin_part']] = data['Cabin'].str.split("/", expand=True).iloc[:, [0, 2]]


        return data

    def passenger_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates passenger-related features including group size and family details.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing passenger information.

        Returns:
            pd.DataFrame: The updated DataFrame with additional passenger features.
        """
        # Count number of passengers per 'Cabin' and assign result to a new column
        data['NUMBER_OF_PASSENGERS_PER_CABIN'] = data.groupby('Cabin')['PassengerId'].transform('count')

        # Count passengers per group by splitting 'PassengerId' and assigning result to a new column
        data['NUMBER_OF_PASSENGERS_PER_GROUP'] = data['PassengerId'].str.split("_").str[0].map(
            data['PassengerId'].str.split("_").str[0].value_counts())

        # Calculate mean age per group and assign to a new column
        data['Avg_Age_Per_Group'] = data.groupby('NUMBER_OF_PASSENGERS_PER_GROUP')['Age'].transform('mean')
        

        # Number of people in a family
        data['FAMILY'] = data.groupby(by='FAMILY_ID')['FAMILY_ID'].transform('count')

        # Number of passengers in CryoSleep per HomePlanet
        data['Cryo_Count'] = data.groupby(by='HomePlanet')['CryoSleep'].transform('count')

        return data

    def service_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates service-related spending features more efficiently.

        Args:
            data (pd.DataFrame): Input DataFrame containing service spending information.

        Returns:
            pd.DataFrame: Updated DataFrame with service-related features.
        """
        service_columns = ['RoomService', 'Spa', 'FoodCourt', 'ShoppingMall', 'VRDeck']

        # Compute aggregate spending features at different levels
        passenger_spending = data.groupby('PassengerId')[service_columns].agg(['mean'])
        family_spending = data.groupby('FAMILY_ID')[service_columns].agg(['mean'])
        deck_spending = data.groupby('Deck')[service_columns].agg(['mean'])
        family_spending_sum = data.groupby('FAMILY_ID')[service_columns].agg(['sum'])

        passenger_spending.columns = [f"{stat}_Spending_On_{col}_Pass" for col, stat in passenger_spending.columns]
        family_spending.columns = [f"{stat}_Spending_On_{col}_Fam" for col, stat in family_spending.columns]
        deck_spending.columns = [f"{stat}_Spending_On_{col}_Deck" for col, stat in deck_spending.columns]
        family_spending_sum.columns = [f"{stat}_Spending_On_{col}_Fam_SUM" for col, stat in family_spending_sum.columns]

        data['TotalSpending'] = data[service_columns].sum(axis=1)
        data["MeanSpending_HomePlanet"] = data.groupby("HomePlanet")["TotalSpending"].transform("mean")
        data["MeanSpending_Destination"] = data.groupby("Destination")["TotalSpending"].transform("mean")
        data["TotalSpending_Family"] = data.groupby("FAMILY")["TotalSpending"].transform("sum")
        data['SPEND_PER_FAMILY'] = data['TotalSpending'] / data['FAMILY']
        data["isHighSpender"] = (data["TotalSpending"] > data["TotalSpending"].median()).astype(int)

        # Merge computed statistics back to the original DataFrame
        data = data.merge(passenger_spending, on='PassengerId', how='left')
        data = data.merge(family_spending, on='FAMILY_ID', how='left')
        data = data.merge(deck_spending, on='Deck', how='left')
        data = data.merge(family_spending_sum, on='FAMILY_ID', how='left')

        # Create binary and categorical features for age column
        data['isMinor'] = (data['Age'] < 18).astype(int)
        data['Age_Cat'] = pd.cut(data['Age'], bins=[0, 12, 17, 64, float('inf')], labels=['Child', 'Teen', 'Adult', 'Senior'])

        # Convert categorical features to dummy variables (optional)
        data = pd.get_dummies(data, columns=['Age_Cat'], dtype=int)

        return data
    
    def destination_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates destination-related features.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing destination related features.

        Returns:
            pd.DataFrame: The updated DataFrame with destination related features.
        """
        # Count the number of occurrences of each 'Destination' by 'HomePlanet'
        data['Destination_Count_By_HomePlanet'] = data.groupby('Destination')['HomePlanet'].transform('count')

        return data 

    
    def missing_value_imputation(self, data: pd.DataFrame, n_neighbors: int, columns_to_impute: list) -> pd.DataFrame:
        """Imputes missing values using the KNNImputer.
        
        Args:
            data (pd.DataFrame): The DataFrame containing missing values.
            n_neighbors (int): Number of neighbors for KNNImputer.
            columns_to_impute (list): Columns to be imputed.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed.
        """
        # Extracting family id
        data['FAMILY_ID']  = data['PassengerId'].str[:4]
        
        # Create a mapping from family_id to known homeplanet (excluding NaNs)
        family_planet_map = data.groupby('FAMILY_ID')['HomePlanet'].apply(lambda x: x.dropna().unique())
        
        # Fill missing values if a family member has a known homeplanet
        data['HomePlanet'] = data.apply(lambda row: family_planet_map[row['FAMILY_ID']][0] 
                                    if pd.isna(row['HomePlanet']) and len(family_planet_map[row['FAMILY_ID']]) > 0 
                                    else row['HomePlanet'], axis=1)
    
        # Initialize the KNN imputer with the specified number of neighbors
        imputer = KNNImputer(n_neighbors=n_neighbors)


        # Select only the columns to be imputed
        data_to_impute = data[columns_to_impute]

        # Fit and transform the data for imputation, then update original data
        data_imputed = pd.DataFrame(imputer.fit_transform(data_to_impute), columns=columns_to_impute, index=data.index)
        data.update(data_imputed)

        # Identify categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns

        # Initialize SimpleImputer for categorical features
        cat_imputer = SimpleImputer(strategy='most_frequent')

        # Impute categorical features using mode and ensure DataFrame structure is maintained
        imputed_cat_data = pd.DataFrame(
            cat_imputer.fit_transform(data[cat_cols]),
            columns=cat_cols,
            index=data.index
        )

        # Convert the categorical columns back to string type (if needed)
        data = data.astype(str)

        # Combine imputed categorical features with the rest of the DataFrame
        data = pd.concat([imputed_cat_data, data.drop(columns=cat_cols)], axis=1)

        data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].astype('float32')
        
        return data

    def feature_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Runs the full feature engineering pipeline on data.
        
        Args:
            data (pd.DataFrame): The input DataFrame to process.

        Returns:
            pd.DataFrame: The processed DataFrame with all features engineered.
        """
        # Apply each feature engineering function to the data
        data_processed = self.ship_related_features(data)
        data_processed = self.passenger_features(data_processed)
        data_processed = self.service_features(data_processed)
        data_processed = self.destination_features(data_processed)

        return data_processed
    
    def data_split(self, data: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
        """Splits the data into features and target.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing features and target.
            features (list[str]): The list of feature column names.
            target (str): The name of the target column.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and target Series.
        """
        # Select the feature columns and target column from data
        X_train = data[features]
        y_train = data[target]

        return X_train, y_train
    
    def get_dummies(self, data: pd.DataFrame, dtype: type) -> pd.DataFrame:
        """Creates dummy variables for categorical features.
        
        Args:
            data (pd.DataFrame): The input DataFrame to create dummy variables from.
            dtype (type): The desired data type for the resulting dummy variables.

        Returns:
            pd.DataFrame: The DataFrame with dummy variables added.
        """
        # Convert categorical columns to dummy/one-hot encoded columns
        dummied_data = pd.get_dummies(data, dtype=dtype)
        
        return dummied_data