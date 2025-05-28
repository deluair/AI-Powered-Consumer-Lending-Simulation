import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier # Example: XGBoost for potentially complex relationships
from sklearn.metrics import roc_auc_score, classification_report

from ..config import MODELS_DIR
from ..utils.logger import get_logger
from ..feature_engineering.feature_transformer import FeatureTransformer
from ..feature_engineering.alternative_feature_extractor import AlternativeFeatureExtractor

logger = get_logger(__name__)

class AlternativeModel:
    def __init__(self, model_name="alternative_xgb_model.joblib"):
        self.model_name = model_name
        self.model_path = MODELS_DIR / self.model_name
        self.model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # Example
        self.feature_transformer = None # For numerical/categorical alternative features
        # self.alt_feature_extractor = AlternativeFeatureExtractor() # If raw alt data needs processing first
        self._is_trained = False

    def train(self, X_alt: pd.DataFrame, y: pd.Series, 
              numerical_alt_cols=None, categorical_alt_cols=None, test_size=0.2):
        """
        Trains the model using alternative data features.
        Assumes X_alt contains already extracted alternative features.
        """
        logger.info(f"Starting training for {self.model_name} with alternative features...")
        
        X_train, X_test, y_train, y_test = train_test_split(X_alt, y, test_size=test_size, random_state=42, stratify=y)
        
        logger.info(f"Alt Training data shape: {X_train.shape}, Alt Test data shape: {X_test.shape}")

        # Initialize and fit feature transformer for alternative features
        # These might be different from traditional features (e.g., text-derived, scores)
        self.feature_transformer = FeatureTransformer(numerical_cols=numerical_alt_cols, 
                                                    categorical_cols=categorical_alt_cols)
        X_train_transformed = self.feature_transformer.fit_transform(X_train)
        X_test_transformed = self.feature_transformer.transform(X_test)
        
        logger.info(f"Transformed Alt training data shape: {X_train_transformed.shape}")

        self.model.fit(X_train_transformed, y_train)
        self._is_trained = True
        logger.info(f"{self.model_name} trained successfully.")

        # Evaluate on test set
        y_pred_proba = self.model.predict_proba(X_test_transformed)[:, 1]
        y_pred = self.model.predict(X_test_transformed)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        logger.info(f"Alt Model Test Set AUC: {auc:.4f}")
        logger.info(f"Alt Model Test Set Classification Report:\n{report}")
        
        self.save_model()
        return {"auc": auc, "classification_report": report}

    def predict(self, X_alt: pd.DataFrame) -> pd.Series:
        """Makes predictions on new alternative data features."""
        if not self._is_trained:
            logger.warning("Alt Model is not trained yet. Attempting to load from disk.")
            if not self.load_model():
                 raise RuntimeError("Alt Model is not trained and could not be loaded from disk.")
        
        if not self.feature_transformer:
            raise RuntimeError("Alt Feature transformer is not available.")

        X_alt_transformed = self.feature_transformer.transform(X_alt)
        predictions = self.model.predict(X_alt_transformed)
        return pd.Series(predictions, index=X_alt.index, name="alt_prediction")

    def predict_proba(self, X_alt: pd.DataFrame) -> pd.Series:
        """Predicts probabilities on new alternative data features."""
        if not self._is_trained:
            logger.warning("Alt Model is not trained yet. Attempting to load from disk.")
            if not self.load_model(): # Ensure load_model sets self._is_trained and self.feature_transformer
                raise RuntimeError("Alt Model is not trained and could not be loaded from disk.")

        if not self.feature_transformer:
            raise RuntimeError("Alt Feature transformer is not available.")

        X_alt_transformed = self.feature_transformer.transform(X_alt)
        probabilities = self.model.predict_proba(X_alt_transformed)[:, 1]
        return pd.Series(probabilities, index=X_alt.index, name="alt_probability_default")

    def save_model(self):
        """Saves the trained alternative model and its feature transformer."""
        if not self._is_trained or not self.feature_transformer:
            logger.error("Cannot save Alt model. Model or feature transformer not available.")
            return
        
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        model_payload = {
            'model': self.model,
            'feature_transformer': self.feature_transformer
        }
        joblib.dump(model_payload, self.model_path)
        logger.info(f"Alternative model and transformer saved to {self.model_path}")

    def load_model(self) -> bool:
        """Loads the alternative model and its feature transformer."""
        if self.model_path.exists():
            try:
                model_payload = joblib.load(self.model_path)
                self.model = model_payload['model']
                self.feature_transformer = model_payload['feature_transformer']
                self._is_trained = True
                logger.info(f"Alternative model and transformer loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading alternative model from {self.model_path}: {e}")
                return False
        else:
            logger.warning(f"Alternative model file not found at {self.model_path}")
            return False

if __name__ == '__main__':
    from ..data_ingestion.synthetic_data_generator import SyntheticDataGenerator
    # Assuming AlternativeFeatureExtractor is in the feature_engineering module
    from ..feature_engineering.alternative_feature_extractor import AlternativeFeatureExtractor

    logger.info("Running AlternativeModel example...")
    sdg = SyntheticDataGenerator(random_seed=42)
    # Generate data that would have alternative features (thin-file, invisible)
    data_df = sdg.generate_dataset(num_traditional=100, num_thin_file=300, num_credit_invisible=200)
    data_df = data_df.set_index('user_id') # Ensure user_id is index for joining

    # Simulate extracting alternative features
    # In a real pipeline, these would come from various sources and be processed by AlternativeFeatureExtractor
    # For this example, we'll use some columns directly from synthetic data that represent alt features
    alt_feature_cols_numeric = [
        'digital_footprint_score', 'social_media_longevity_years', 
        'avg_monthly_utility_bill', 'residential_stability_years', 'rent_payment_ontime_pct'
    ]
    alt_feature_cols_categorical = [
        'utility_payment_rating', 'mobile_plan_type', 'utility_payment_consistency'
    ]
    all_alt_features = alt_feature_cols_numeric + alt_feature_cols_categorical

    # Create a DataFrame of just alternative features
    # Handle missing features by filling with a placeholder or dropping
    X_alternative_features = data_df[all_alt_features].copy()
    X_alternative_features.fillna({
        'digital_footprint_score': X_alternative_features['digital_footprint_score'].median(),
        'social_media_longevity_years': X_alternative_features['social_media_longevity_years'].median(),
        'avg_monthly_utility_bill': X_alternative_features['avg_monthly_utility_bill'].median(),
        'residential_stability_years': X_alternative_features['residential_stability_years'].median(),
        'rent_payment_ontime_pct': X_alternative_features['rent_payment_ontime_pct'].median(),
        'utility_payment_rating': 'Unknown',
        'mobile_plan_type': 'Unknown',
        'utility_payment_consistency': 'Unknown'
    }, inplace=True)

    y = data_df['defaulted']

    if X_alternative_features.empty or len(X_alternative_features) < 50:
        logger.error("Not enough valid alternative feature data to run example. Exiting.")
        exit()

    alt_model = AlternativeModel()
    alt_model.train(X_alternative_features, y, 
                    numerical_alt_cols=alt_feature_cols_numeric, 
                    categorical_alt_cols=alt_feature_cols_categorical)

    # Test prediction
    sample_X_alt = X_alternative_features.head()
    alt_predictions = alt_model.predict(sample_X_alt)
    alt_probabilities = alt_model.predict_proba(sample_X_alt)
    logger.info(f"Sample Alt Predictions:\n{alt_predictions}")
    logger.info(f"Sample Alt Probabilities:\n{alt_probabilities}")

    # Test loading model
    loaded_alt_model = AlternativeModel()
    if loaded_alt_model.load_model():
        logger.info("Alternative model loaded successfully for a new instance.")
        loaded_alt_predictions = loaded_alt_model.predict(sample_X_alt)
        logger.info(f"Predictions from loaded Alt model:\n{loaded_alt_predictions}")
    else:
        logger.error("Failed to load the saved alternative model.")