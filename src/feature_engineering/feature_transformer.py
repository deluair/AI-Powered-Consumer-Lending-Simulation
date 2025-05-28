import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from ..utils.logger import get_logger

logger = get_logger(__name__)

class FeatureTransformer:
    def __init__(self, numerical_cols=None, categorical_cols=None, passthrough_cols=None):
        self.numerical_cols = numerical_cols if numerical_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.passthrough_cols = passthrough_cols if passthrough_cols else []
        self.preprocessor = None
        self._fit_status = False

    def _identify_columns(self, df: pd.DataFrame):
        """Automatically identify column types if not provided."""
        if not self.numerical_cols and not self.categorical_cols:
            logger.info("Attempting to auto-identify numerical and categorical columns.")
            self.numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Exclude passthrough columns from auto-identified lists
            if self.passthrough_cols:
                self.numerical_cols = [col for col in self.numerical_cols if col not in self.passthrough_cols]
                self.categorical_cols = [col for col in self.categorical_cols if col not in self.passthrough_cols]
            
            logger.info(f"Identified numerical columns: {self.numerical_cols}")
            logger.info(f"Identified categorical columns: {self.categorical_cols}")

    def fit(self, df: pd.DataFrame, y=None):
        """Fits the preprocessor on the dataframe."""
        logger.info("Fitting FeatureTransformer...")
        self._identify_columns(df.copy()) # Use a copy to avoid modifying original df during type checks

        transformers = []
        if self.numerical_cols:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, self.numerical_cols))
        
        if self.categorical_cols:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')), # or a constant like 'missing'
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_cols))
        
        if self.passthrough_cols:
            transformers.append(('pass', 'passthrough', self.passthrough_cols))

        if not transformers:
            logger.warning("No columns specified for transformation or passthrough. Preprocessor will be empty.")
            # Create a dummy preprocessor that does nothing if no columns are to be processed
            self.preprocessor = Pipeline([('dummy', 'passthrough')]) 
        else:
            self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop') # drop any cols not specified

        self.preprocessor.fit(df)
        self._fit_status = True
        logger.info("FeatureTransformer fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataframe using the fitted preprocessor."""
        if not self._fit_status:
            raise RuntimeError("Transformer has not been fitted yet. Call fit() first.")
        logger.info("Transforming data...")
        
        # Ensure columns are in the same order as during fit, and only include known columns
        # This is a simplified way; more robust handling might be needed for missing/extra columns.
        cols_to_process = []
        if self.numerical_cols: cols_to_process.extend(self.numerical_cols)
        if self.categorical_cols: cols_to_process.extend(self.categorical_cols)
        if self.passthrough_cols: cols_to_process.extend(self.passthrough_cols)
        
        # Filter df to include only columns the preprocessor expects
        # df_filtered = df[cols_to_process].copy()

        transformed_data = self.preprocessor.transform(df)
        
        # Get feature names after transformation (especially for OneHotEncoder)
        feature_names = []
        try:
            if self.preprocessor and hasattr(self.preprocessor, 'get_feature_names_out'):
                 feature_names = self.preprocessor.get_feature_names_out()
            else: # Fallback for older scikit-learn or custom preprocessor
                # This part is tricky and depends on the transformers used.
                # For ColumnTransformer with standard components:
                for name, trans, columns in self.preprocessor.transformers_:
                    if trans == 'passthrough':
                        feature_names.extend(columns)
                    elif hasattr(trans, 'get_feature_names_out'):
                        feature_names.extend(trans.get_feature_names_out(columns))
                    else: # Simple case for one-hot encoder in a pipeline
                        if name == 'cat' and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                            feature_names.extend(trans.named_steps['onehot'].get_feature_names_out(columns))
                        else:
                             feature_names.extend(columns) # Fallback, may not be correct for all transformers
        except Exception as e:
            logger.warning(f"Could not automatically get feature names: {e}. Using generic names.")
            feature_names = [f"feature_{i}" for i in range(transformed_data.shape[1])]

        transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
        logger.info(f"Data transformed. Shape: {transformed_df.shape}")
        return transformed_df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fits and then transforms the dataframe."""
        self.fit(df, y)
        return self.transform(df)

if __name__ == '__main__':
    # Sample Data
    data = {
        'age': [25, 30, np.nan, 35, 22],
        'income': [50000, 60000, 75000, np.nan, 45000],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'city': ['NY', 'LA', 'NY', 'SF', 'LA'],
        'user_id': ['id1', 'id2', 'id3', 'id4', 'id5']
    }
    sample_df = pd.DataFrame(data).set_index('user_id')

    logger.info("Original DataFrame:")
    logger.info(sample_df)

    # Define columns
    # Option 1: Specify columns
    # transformer = FeatureTransformer(
    #     numerical_cols=['age', 'income'],
    #     categorical_cols=['gender', 'city']
    # )

    # Option 2: Auto-detect (passthrough 'user_id' if it wasn't index)
    transformer = FeatureTransformer() # Auto-detects numerical and categorical
    # If user_id was a column: transformer = FeatureTransformer(passthrough_cols=['user_id'])

    transformed_df = transformer.fit_transform(sample_df)

    logger.info("\nTransformed DataFrame:")
    logger.info(transformed_df.head())
    logger.info(f"\nTransformed DataFrame columns: {transformed_df.columns.tolist()}")

    # Example with passthrough
    data_pass = {
        'age': [25, 30, np.nan, 35, 22],
        'income': [50000, 60000, 75000, np.nan, 45000],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'user_id_col': ['id1', 'id2', 'id3', 'id4', 'id5']
    }
    sample_df_pass = pd.DataFrame(data_pass)
    transformer_pass = FeatureTransformer(passthrough_cols=['user_id_col'])
    transformed_df_pass = transformer_pass.fit_transform(sample_df_pass)
    logger.info("\nTransformed DataFrame with passthrough:")
    logger.info(transformed_df_pass.head())