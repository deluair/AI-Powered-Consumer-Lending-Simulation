import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Example model
from sklearn.metrics import roc_auc_score, classification_report

from ..config import MODELS_DIR
from ..utils.logger import get_logger
from ..feature_engineering.feature_transformer import FeatureTransformer

logger = get_logger(__name__)

class TraditionalModel:
    def __init__(self, model_name="traditional_logistic_regression.joblib"):
        self.model_name = model_name
        self.model_path = MODELS_DIR / self.model_name
        self.model = LogisticRegression(solver='liblinear', random_state=42) # Example
        self.feature_transformer = None # Will be set during training
        self._is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, 
              numerical_cols=None, categorical_cols=None, test_size=0.2):
        """Trains the traditional model."""
        logger.info(f"Starting training for {self.model_name}...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # Initialize and fit feature transformer
        self.feature_transformer = FeatureTransformer(numerical_cols=numerical_cols, categorical_cols=categorical_cols)
        X_train_transformed = self.feature_transformer.fit_transform(X_train)
        X_test_transformed = self.feature_transformer.transform(X_test)
        
        logger.info(f"Transformed training data shape: {X_train_transformed.shape}")

        self.model.fit(X_train_transformed, y_train)
        self._is_trained = True
        logger.info(f"{self.model_name} trained successfully.")

        # Evaluate on test set
        y_pred_proba = self.model.predict_proba(X_test_transformed)[:, 1]
        y_pred = self.model.predict(X_test_transformed)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        logger.info(f"Test Set AUC: {auc:.4f}")
        logger.info(f"Test Set Classification Report:\n{report}")
        
        self.save_model()
        return {"auc": auc, "classification_report": report}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not self._is_trained:
            logger.warning("Model is not trained yet. Attempting to load from disk.")
            if not self.load_model():
                 raise RuntimeError("Model is not trained and could not be loaded from disk.")
        
        if not self.feature_transformer:
            raise RuntimeError("Feature transformer is not available. Model may not have been trained or loaded correctly.")

        X_transformed = self.feature_transformer.transform(X)
        predictions = self.model.predict(X_transformed)
        return pd.Series(predictions, index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Predicts probabilities on new data."""
        if not self._is_trained:
            logger.warning("Model is not trained yet. Attempting to load from disk.")
            if not self.load_model():
                raise RuntimeError("Model is not trained and could not be loaded from disk.")

        if not self.feature_transformer:
            raise RuntimeError("Feature transformer is not available. Model may not have been trained or loaded correctly.")

        X_transformed = self.feature_transformer.transform(X)
        probabilities = self.model.predict_proba(X_transformed)[:, 1] # Probability of positive class
        return pd.Series(probabilities, index=X.index, name="probability_default")

    def save_model(self):
        """Saves the trained model and feature transformer to disk."""
        if not self._is_trained or not self.feature_transformer:
            logger.error("Cannot save model. Model or feature transformer not available.")
            return
        
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        model_payload = {
            'model': self.model,
            'feature_transformer': self.feature_transformer
        }
        joblib.dump(model_payload, self.model_path)
        logger.info(f"Model and transformer saved to {self.model_path}")

    def load_model(self) -> bool:
        """Loads the model and feature transformer from disk."""
        if self.model_path.exists():
            try:
                model_payload = joblib.load(self.model_path)
                self.model = model_payload['model']
                self.feature_transformer = model_payload['feature_transformer']
                self._is_trained = True # Assume loaded model is trained
                logger.info(f"Model and transformer loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model from {self.model_path}: {e}")
                return False
        else:
            logger.warning(f"Model file not found at {self.model_path}")
            return False

if __name__ == '__main__':
    # Example Usage (requires synthetic_data_generator to be runnable for dummy data)
    from ..data_ingestion.synthetic_data_generator import SyntheticDataGenerator
    
    logger.info("Running TraditionalModel example...")
    # Generate some dummy data
    sdg = SyntheticDataGenerator(random_seed=42)
    # Focus on traditional profiles for this model
    data_df = sdg.generate_dataset(num_traditional=500, num_thin_file=0, num_credit_invisible=0)
    
    # Define features and target (simplified for example)
    # In a real scenario, these would be more carefully selected traditional credit features
    traditional_features = ['credit_score', 'months_credit_history', 'num_credit_accounts', 
                            'credit_utilization_ratio', 'debt_to_income_ratio', 'annual_income', 'age']
    categorical_features = [] # Add any categorical traditional features here
    
    # Ensure all features are present and handle missing data if necessary
    # For this example, we'll drop rows with NaNs in key features for simplicity
    data_df = data_df.dropna(subset=traditional_features + ['defaulted'])
    
    if data_df.empty or len(data_df) < 50: # Need enough data to split
        logger.error("Not enough valid data to run example. Exiting.")
        exit()

    X = data_df[traditional_features]
    y = data_df['defaulted']

    trad_model = TraditionalModel()
    training_results = trad_model.train(X, y, numerical_cols=traditional_features, categorical_cols=categorical_features)
    
    logger.info(f"Training results: {training_results}")

    # Test prediction
    if not X.empty:
        sample_X = X.head()
        predictions = trad_model.predict(sample_X)
        probabilities = trad_model.predict_proba(sample_X)
        logger.info(f"Sample Predictions:\n{predictions}")
        logger.info(f"Sample Probabilities:\n{probabilities}")
    else:
        logger.warning("No data available to make sample predictions.")

    # Test loading model
    loaded_model = TraditionalModel()
    if loaded_model.load_model():
        logger.info("Model loaded successfully for a new instance.")
        if not X.empty:
            loaded_predictions = loaded_model.predict(X.head())
            logger.info(f"Predictions from loaded model:\n{loaded_predictions}")
    else:
        logger.error("Failed to load the saved model.")