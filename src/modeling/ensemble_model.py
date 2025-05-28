import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Meta-learner example
from sklearn.metrics import roc_auc_score, classification_report

from ..config import MODELS_DIR
from ..utils.logger import get_logger
from .traditional_model import TraditionalModel # Assuming it's in the same directory
from .alternative_model import AlternativeModel # Assuming it's in the same directory

logger = get_logger(__name__)

class EnsembleModel:
    def __init__(self, traditional_model_path=None, alternative_model_path=None, 
                 model_name="ensemble_meta_learner.joblib"):
        self.model_name = model_name
        self.model_path = MODELS_DIR / self.model_name
        
        self.traditional_model = TraditionalModel()
        if traditional_model_path:
            self.traditional_model.model_path = traditional_model_path # Override default path if provided
        
        self.alternative_model = AlternativeModel()
        if alternative_model_path:
            self.alternative_model.model_path = alternative_model_path # Override default path if provided
            
        self.meta_learner = LogisticRegression(solver='liblinear', random_state=42) # Example meta-learner
        self._is_trained = False

    def _load_base_models(self):
        """Loads the pre-trained base models."""
        trad_loaded = self.traditional_model.load_model()
        alt_loaded = self.alternative_model.load_model()
        if not trad_loaded:
            logger.error("Failed to load the traditional base model for the ensemble.")
            # raise RuntimeError("Traditional base model could not be loaded.")
            return False
        if not alt_loaded:
            logger.error("Failed to load the alternative base model for the ensemble.")
            # raise RuntimeError("Alternative base model could not be loaded.")
            return False
        logger.info("Base models loaded successfully for ensemble.")
        return True

    def _get_base_model_predictions(self, X_traditional: pd.DataFrame, X_alternative: pd.DataFrame) -> pd.DataFrame:
        """Generates predictions from the base models."""
        if not self.traditional_model._is_trained or not self.alternative_model._is_trained:
            if not self._load_base_models(): # Attempt to load if not already loaded
                 raise RuntimeError("Base models are not trained or loaded.")

        # Ensure indices match for proper alignment if X_traditional and X_alternative are from the same source df
        # This assumes X_traditional and X_alternative have the same index
        if not X_traditional.index.equals(X_alternative.index):
            logger.warning("Indices of traditional and alternative data do not match. Ensure data corresponds to the same users.")
            # Attempt to align by reindexing alternative data to traditional data's index, if sizes match
            if len(X_traditional) == len(X_alternative):
                X_alternative = X_alternative.reindex(X_traditional.index)
            else:
                raise ValueError("Cannot align traditional and alternative data due to different lengths and mismatched indices.")

        trad_pred_proba = self.traditional_model.predict_proba(X_traditional)
        alt_pred_proba = self.alternative_model.predict_proba(X_alternative)
        
        # Combine predictions into a new feature set for the meta-learner
        # Ensure the series have the same index before concatenation
        base_predictions = pd.concat([trad_pred_proba.rename('trad_model_proba'), 
                                      alt_pred_proba.rename('alt_model_proba')], axis=1)
        return base_predictions

    def train_meta_learner(self, X_traditional: pd.DataFrame, X_alternative: pd.DataFrame, y: pd.Series, test_size=0.2):
        """Trains the meta-learner on predictions from base models."""
        logger.info(f"Starting training for ensemble meta-learner {self.model_name}...")

        if not self._load_base_models(): # Ensure base models are loaded
            logger.error("Cannot train meta-learner because base models failed to load.")
            return None

        # Split data first to avoid leakage from base model predictions on the same data used for meta-learner training
        # This is a simplified approach. A more robust way is k-fold cross-validation for base model predictions.
        X_trad_train, X_trad_val, X_alt_train, X_alt_val, y_train, y_val = train_test_split(
            X_traditional, X_alternative, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Meta-learner training set size: {len(y_train)}, Validation set size: {len(y_val)}")

        # Get predictions from base models on the training portion for the meta-learner
        # Ideally, these base models should be trained on a fold different from what they predict on for meta-training
        # For simplicity here, we use the already loaded (potentially fully trained) base models.
        # This can lead to overfitting if base models saw this data. Proper stacking uses out-of-fold predictions.
        meta_features_train = self._get_base_model_predictions(X_trad_train, X_alt_train)
        meta_features_val = self._get_base_model_predictions(X_trad_val, X_alt_val)
        
        logger.info(f"Meta-features for training shape: {meta_features_train.shape}")

        self.meta_learner.fit(meta_features_train, y_train)
        self._is_trained = True
        logger.info(f"Ensemble meta-learner {self.model_name} trained successfully.")

        # Evaluate on validation set
        y_pred_proba = self.meta_learner.predict_proba(meta_features_val)[:, 1]
        y_pred = self.meta_learner.predict(meta_features_val)
        
        auc = roc_auc_score(y_val, y_pred_proba)
        report = classification_report(y_val, y_pred)
        logger.info(f"Ensemble Meta-Learner Validation Set AUC: {auc:.4f}")
        logger.info(f"Ensemble Meta-Learner Validation Set Classification Report:\n{report}")
        
        self.save_model()
        return {"auc": auc, "classification_report": report}

    def predict(self, X_traditional: pd.DataFrame, X_alternative: pd.DataFrame) -> pd.Series:
        """Makes final predictions using the ensemble."""
        if not self._is_trained:
            logger.warning("Ensemble meta-learner is not trained. Attempting to load.")
            if not self.load_model():
                raise RuntimeError("Ensemble meta-learner is not trained and could not be loaded.")

        meta_features = self._get_base_model_predictions(X_traditional, X_alternative)
        predictions = self.meta_learner.predict(meta_features)
        return pd.Series(predictions, index=meta_features.index, name="ensemble_prediction")

    def predict_proba(self, X_traditional: pd.DataFrame, X_alternative: pd.DataFrame) -> pd.Series:
        """Predicts final probabilities using the ensemble."""
        if not self._is_trained:
            logger.warning("Ensemble meta-learner is not trained. Attempting to load.")
            if not self.load_model():
                raise RuntimeError("Ensemble meta-learner is not trained and could not be loaded.")

        meta_features = self._get_base_model_predictions(X_traditional, X_alternative)
        probabilities = self.meta_learner.predict_proba(meta_features)[:, 1]
        return pd.Series(probabilities, index=meta_features.index, name="ensemble_probability_default")

    def save_model(self):
        """Saves the trained meta-learner to disk."""
        if not self._is_trained:
            logger.error("Cannot save meta-learner. Model not trained.")
            return
        
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Only save the meta-learner; base models are saved by their respective classes
        joblib.dump(self.meta_learner, self.model_path)
        logger.info(f"Ensemble meta-learner saved to {self.model_path}")

    def load_model(self) -> bool:
        """Loads the meta-learner from disk. Base models must be loaded separately or be loadable by their classes."""
        if self.model_path.exists():
            try:
                self.meta_learner = joblib.load(self.model_path)
                # Attempt to load base models as well, as they are needed for prediction
                if not self._load_base_models():
                    logger.warning("Meta-learner loaded, but base models failed to load. Predictions might fail.")
                    # self._is_trained should reflect if the *ensemble* is ready
                    # If base models can't load, the ensemble isn't fully ready.
                    self._is_trained = False 
                    return False
                self._is_trained = True
                logger.info(f"Ensemble meta-learner loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading ensemble meta-learner from {self.model_path}: {e}")
                self._is_trained = False
                return False
        else:
            logger.warning(f"Ensemble meta-learner file not found at {self.model_path}")
            self._is_trained = False
            return False

if __name__ == '__main__':
    from ..data_ingestion.synthetic_data_generator import SyntheticDataGenerator
    logger.info("Running EnsembleModel example...")

    # This example assumes TraditionalModel and AlternativeModel have been trained and saved.
    # For a self-contained example, we'd need to train them first or use mock models.
    # Here, we'll try to load them. Ensure their .joblib files exist from their own script runs.

    # 1. Generate or load data
    sdg = SyntheticDataGenerator(random_seed=42)
    data_df = sdg.generate_dataset(num_traditional=200, num_thin_file=150, num_credit_invisible=100)
    data_df = data_df.set_index('user_id')

    # Define feature sets (must match what base models were trained on)
    # These are simplified examples from the base model scripts
    traditional_features_num = ['credit_score', 'months_credit_history', 'num_credit_accounts', 
                                'credit_utilization_ratio', 'debt_to_income_ratio', 'annual_income', 'age']
    # Clean data for traditional model features
    X_trad_full = data_df[traditional_features_num].copy().dropna()
    
    alt_feature_cols_numeric = [
        'digital_footprint_score', 'social_media_longevity_years', 
        'avg_monthly_utility_bill', 'residential_stability_years', 'rent_payment_ontime_pct'
    ]
    alt_feature_cols_categorical = [
        'utility_payment_rating', 'mobile_plan_type', 'utility_payment_consistency'
    ]
    all_alt_features = alt_feature_cols_numeric + alt_feature_cols_categorical
    X_alt_full = data_df[all_alt_features].copy()
    X_alt_full.fillna({
        'digital_footprint_score': X_alt_full['digital_footprint_score'].median(),
        'social_media_longevity_years': X_alt_full['social_media_longevity_years'].median(),
        'avg_monthly_utility_bill': X_alt_full['avg_monthly_utility_bill'].median(),
        'residential_stability_years': X_alt_full['residential_stability_years'].median(),
        'rent_payment_ontime_pct': X_alt_full['rent_payment_ontime_pct'].median(),
        'utility_payment_rating': 'Unknown',
        'mobile_plan_type': 'Unknown',
        'utility_payment_consistency': 'Unknown'
    }, inplace=True)

    # Align data and target variable
    common_index = X_trad_full.index.intersection(X_alt_full.index)
    if common_index.empty:
        logger.error("No common users between traditional and alternative feature sets after cleaning. Exiting example.")
        exit()

    X_trad_aligned = X_trad_full.loc[common_index]
    X_alt_aligned = X_alt_full.loc[common_index]
    y_aligned = data_df.loc[common_index, 'defaulted']

    if len(y_aligned) < 50: # Need enough data for meta-learner training/validation
        logger.error("Not enough aligned data to run ensemble example. Exiting.")
        exit()

    # 2. Initialize EnsembleModel (it will try to load base models)
    ensemble = EnsembleModel()
    
    # Check if base models loaded successfully before proceeding to train meta-learner
    if not ensemble.traditional_model._is_trained or not ensemble.alternative_model._is_trained:
        logger.error("Base models for ensemble could not be loaded. Ensure they are trained and saved.")
        logger.error("Run traditional_model.py and alternative_model.py first if their .joblib files are missing.")
        # As a fallback for the example, let's try to train them if they are not loaded.
        # This is NOT ideal for a real stacking setup but makes the example more runnable.
        logger.info("Attempting to train base models as they were not loaded...")
        if not ensemble.traditional_model._is_trained:
            logger.info("Training traditional base model...")
            # Need to pass the correct y for the X_trad_full data
            y_for_trad = data_df.loc[X_trad_full.index, 'defaulted']
            ensemble.traditional_model.train(X_trad_full, y_for_trad, numerical_cols=traditional_features_num)
        
        if not ensemble.alternative_model._is_trained:
            logger.info("Training alternative base model...")
            y_for_alt = data_df.loc[X_alt_full.index, 'defaulted']
            ensemble.alternative_model.train(X_alt_full, y_for_alt, 
                                             numerical_alt_cols=alt_feature_cols_numeric, 
                                             categorical_alt_cols=alt_feature_cols_categorical)
        
        # Re-check if base models are now trained
        if not ensemble.traditional_model._is_trained or not ensemble.alternative_model._is_trained:
            logger.error("Still failed to get base models ready. Exiting example.")
            exit()
        logger.info("Base models trained/loaded.")

    # 3. Train the meta-learner
    logger.info("Training ensemble meta-learner...")
    ensemble.train_meta_learner(X_trad_aligned, X_alt_aligned, y_aligned)

    # 4. Test prediction
    if not X_trad_aligned.empty and not X_alt_aligned.empty:
        sample_X_trad = X_trad_aligned.head()
        sample_X_alt = X_alt_aligned.head()
        
        ensemble_predictions = ensemble.predict(sample_X_trad, sample_X_alt)
        ensemble_probabilities = ensemble.predict_proba(sample_X_trad, sample_X_alt)
        logger.info(f"Sample Ensemble Predictions:\n{ensemble_predictions}")
        logger.info(f"Sample Ensemble Probabilities:\n{ensemble_probabilities}")
    else:
        logger.warning("No aligned data available for sample predictions.")

    # 5. Test loading the ensemble model (meta-learner + attempts to load base models)
    loaded_ensemble = EnsembleModel()
    if loaded_ensemble.load_model():
        logger.info("Ensemble model loaded successfully for a new instance.")
        if not X_trad_aligned.empty and not X_alt_aligned.empty:
            loaded_ensemble_predictions = loaded_ensemble.predict(X_trad_aligned.head(), X_alt_aligned.head())
            logger.info(f"Predictions from loaded ensemble model:\n{loaded_ensemble_predictions}")
    else:
        logger.error("Failed to load the saved ensemble model.")