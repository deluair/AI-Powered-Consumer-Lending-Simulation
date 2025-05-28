import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from ..utils.logger import get_logger
from ..config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, SYNTHETIC_DATA_DIR
from ..data_ingestion.synthetic_data_generator import SyntheticDataGenerator
from ..data_ingestion.traditional_data_loader import load_traditional_data
from ..data_ingestion.alternative_data_loader import load_alternative_data
from ..feature_engineering.feature_transformer import FeatureTransformer
from ..feature_engineering.alternative_feature_extractor import AlternativeFeatureExtractor
from ..modeling.traditional_model import TraditionalModel
from ..modeling.alternative_model import AlternativeModel
from ..modeling.ensemble_model import EnsembleModel
from ..modeling.explainability import ModelExplainer
from ..modeling.bias_fairness import BiasFairnessAnalysis

logger = get_logger(__name__)

class LendingSimulator:
    def __init__(self, config=None):
        """
        Initializes the Lending Simulator.
        :param config: A dictionary containing simulation parameters and configurations.
                       (e.g., data sources, model paths, simulation scenarios)
        """
        self.config = config if config else {}
        self.traditional_model = None
        self.alternative_model = None
        self.ensemble_model = None
        self.feature_transformer = None # This might be part of each model wrapper
        self.alt_feature_extractor = None
        self.simulation_results = None
        self.raw_data = None
        self.processed_data = None
        self.applicants_data = None # Data used for a specific simulation run

        self._load_models()
        self._initialize_components()
        logger.info("LendingSimulator initialized.")

    def _load_models(self):
        """
        Loads the pre-trained models specified in the configuration.
        Models are expected to be instances of TraditionalModel, AlternativeModel, EnsembleModel wrappers.
        """
        logger.info("Loading models...")
        try:
            # Load Traditional Model
            trad_model_path = MODELS_DIR / self.config.get("traditional_model_name", "traditional_credit_model.joblib")
            if trad_model_path.exists():
                self.traditional_model = TraditionalModel.load(trad_model_path)
                logger.info(f"Traditional model loaded from {trad_model_path}")
            else:
                logger.warning(f"Traditional model file not found at {trad_model_path}. Traditional model not loaded.")

            # Load Alternative Model
            alt_model_path = MODELS_DIR / self.config.get("alternative_model_name", "alternative_credit_model.joblib")
            if alt_model_path.exists():
                self.alternative_model = AlternativeModel.load(alt_model_path)
                logger.info(f"Alternative model loaded from {alt_model_path}")
            else:
                logger.warning(f"Alternative model file not found at {alt_model_path}. Alternative model not loaded.")

            # Load Ensemble Model
            ens_model_path = MODELS_DIR / self.config.get("ensemble_model_name", "ensemble_model.joblib")
            if ens_model_path.exists():
                self.ensemble_model = EnsembleModel.load(ens_model_path)
                # The ensemble model should internally load its base models if needed, or be configured with them.
                # For this simulator, we assume the ensemble can operate independently or uses paths from its own config.
                logger.info(f"Ensemble model loaded from {ens_model_path}")
            else:
                logger.warning(f"Ensemble model file not found at {ens_model_path}. Ensemble model not loaded.")
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            # Decide if this is a critical error or if simulation can proceed with available models

    def _initialize_components(self):
        """
        Initializes other components like feature extractors.
        """
        self.alt_feature_extractor = AlternativeFeatureExtractor()
        # FeatureTransformer is often part of the model wrappers, so it's loaded with them.
        # If a global transformer is needed, initialize it here.
        logger.info("Other simulation components initialized.")

    def load_applicant_data(self, data_source_type='synthetic', file_path=None, generation_params=None):
        """
        Loads applicant data for the simulation.
        :param data_source_type: 'synthetic', 'traditional_csv', 'alternative_sources', 'custom_df'.
        :param file_path: Path to CSV file if data_source_type is 'traditional_csv'.
        :param generation_params: Dict for synthetic data generation if 'synthetic'.
        """
        logger.info(f"Loading applicant data from source: {data_source_type}")
        if data_source_type == 'synthetic':
            params = generation_params or self.config.get('synthetic_data_params', {})
            sdg = SyntheticDataGenerator(random_seed=params.get('random_seed'))
            self.applicants_data = sdg.generate_dataset(
                num_traditional=params.get('num_traditional', 100),
                num_thin_file=params.get('num_thin_file', 50),
                num_credit_invisible=params.get('num_credit_invisible', 50)
            )
            logger.info(f"Generated {len(self.applicants_data)} synthetic applicant profiles.")
        elif data_source_type == 'traditional_csv':
            if not file_path:
                file_path = PROCESSED_DATA_DIR / self.config.get('applicant_data_file', 'applicants.csv')
            self.applicants_data = load_traditional_data(str(file_path))
            logger.info(f"Loaded {len(self.applicants_data)} applicants from {file_path}.")
        elif data_source_type == 'alternative_sources':
            # This would involve loading multiple alternative data files and merging
            # For now, let's assume it returns a combined DataFrame
            self.applicants_data = load_alternative_data() # Or provide a file_name if needed
            logger.info(f"Loaded {len(self.applicants_data)} applicants from alternative sources.")
        elif data_source_type == 'custom_df' and isinstance(file_path, pd.DataFrame):
            self.applicants_data = file_path
            logger.info(f"Loaded {len(self.applicants_data)} custom DataFrame as applicant data.")
        else:
            logger.error(f"Unsupported data_source_type: {data_source_type}")
            raise ValueError(f"Unsupported data_source_type: {data_source_type}")
        
        # Basic preprocessing or validation
        if self.applicants_data is not None and 'applicant_id' not in self.applicants_data.columns:
            self.applicants_data['applicant_id'] = [f'app_{i}' for i in range(len(self.applicants_data))]
        
        # Store a copy for processing
        self.processed_data = self.applicants_data.copy() if self.applicants_data is not None else None

    def preprocess_data_for_models(self):
        """
        Applies feature engineering steps to the loaded applicant data.
        This method assumes self.applicants_data is populated.
        The actual transformation is handled by the model wrappers' internal transformers.
        This method primarily focuses on extracting alternative features if needed.
        """
        if self.processed_data is None:
            logger.error("Applicant data not loaded. Cannot preprocess.")
            return

        logger.info("Preprocessing data for models...")
        # Alternative feature extraction (if applicable and model needs it)
        # This step might be model-specific. For an ensemble or alternative model, it's relevant.
        if self.config.get('extract_alternative_features', False) and self.alt_feature_extractor:
            logger.info("Extracting alternative features...")
            # Assuming alt_feature_extractor can take the base applicant data and add features
            # This is a placeholder; actual implementation depends on how alt features are structured
            # For example, it might need paths to raw alternative data files linked by applicant_id
            self.processed_data = self.alt_feature_extractor.extract_all_features(self.processed_data)
            logger.info("Alternative features extracted and merged.")
        
        # Further transformations (scaling, encoding) are typically handled by the
        # FeatureTransformer within each model wrapper (TraditionalModel, AlternativeModel, EnsembleModel)
        # when their predict() method is called. So, no explicit global transformation here unless necessary.
        logger.info("Data preprocessing for model input is largely handled by individual model wrappers.")


    def run_simulation(self, model_type='ensemble'):
        """
        Runs the lending simulation using the specified model type.
        :param model_type: 'traditional', 'alternative', or 'ensemble'.
        """
        if self.processed_data is None:
            logger.error("Applicant data not loaded or preprocessed. Cannot run simulation.")
            return None

        logger.info(f"Running simulation with model type: {model_type}...")
        
        model_to_use = None
        if model_type == 'traditional':
            model_to_use = self.traditional_model
        elif model_type == 'alternative':
            model_to_use = self.alternative_model
        elif model_type == 'ensemble':
            model_to_use = self.ensemble_model
        else:
            logger.error(f"Invalid model_type: {model_type}. Choose 'traditional', 'alternative', or 'ensemble'.")
            return None

        if model_to_use is None:
            logger.error(f"Model for type '{model_type}' is not loaded. Cannot run simulation.")
            return None

        # The model wrappers (TraditionalModel, etc.) should handle their own feature transformation internally.
        # We pass the `processed_data` which might include original traditional features and extracted alternative features.
        # The model's internal FeatureTransformer will select relevant columns and process them.
        
        # Make predictions
        # The `predict` method of model wrappers should return an array of predictions (0 or 1)
        # The `predict_proba` method should return an array of probabilities for the positive class
        try:
            predictions = model_to_use.predict(self.processed_data)
            probabilities = model_to_use.predict_proba(self.processed_data)[:, 1] # Probability of class 1 (e.g., default)
        except Exception as e:
            logger.error(f"Error during model prediction with {model_type} model: {e}", exc_info=True)
            # Potentially log self.processed_data.head() or columns for debugging
            logger.error(f"Data columns provided to model: {self.processed_data.columns.tolist()}")
            return None

        # Store results
        self.simulation_results = self.applicants_data.copy() # Start with original applicant data
        self.simulation_results[f'{model_type}_prediction'] = predictions
        self.simulation_results[f'{model_type}_probability_default'] = probabilities # Assuming positive class is 'default'
        
        # Example: Add a simple loan decision rule based on probability
        # This is highly simplified; real decision logic would be more complex.
        threshold = self.config.get('decision_threshold', 0.5) # Default threshold
        self.simulation_results[f'{model_type}_loan_decision'] = np.where(probabilities < threshold, 'Approved', 'Declined')

        logger.info(f"Simulation completed. Results generated for {len(self.simulation_results)} applicants.")
        self.save_simulation_results(filename_prefix=f"simulation_run_{model_type}")
        return self.simulation_results

    def perform_explainability_analysis(self, data_sample: pd.DataFrame, model_type='ensemble', instance_index=0):
        """
        Performs model explainability analysis (SHAP, LIME) on a sample of data.
        :param data_sample: DataFrame sample (in original feature format) to explain.
        :param model_type: Which model's explanations are needed.
        :param instance_index: Index of a specific instance in data_sample for LIME.
        """
        if data_sample.empty:
            logger.warning("Data sample for explainability is empty. Skipping analysis.")
            return

        logger.info(f"Performing explainability analysis for {model_type} model...")
        model_wrapper = getattr(self, f"{model_type}_model", None)
        if not model_wrapper or not hasattr(model_wrapper, 'model') or not hasattr(model_wrapper, 'feature_transformer'):
            logger.error(f"{model_type} model or its components not available for explainability.")
            return

        actual_model = model_wrapper.model
        feature_transformer = model_wrapper.feature_transformer
        original_feature_names = model_wrapper.original_input_features # Expect this attribute in model wrappers
        
        explainer = ModelExplainer(actual_model, feature_transformer, original_feature_names)

        # SHAP (using a background sample from the data_sample or a dedicated background set)
        # For KernelExplainer, background data is important.
        background_sample_shap = data_sample.sample(min(50, len(data_sample)), random_state=42)
        explain_sample_shap = data_sample.sample(min(10, len(data_sample)), random_state=1) # Explain a few instances
        
        shap_values, _ = explainer.explain_with_shap(
            background_sample_shap, 
            explain_sample_shap, 
            save_plot_path=f"{model_type}_shap_summary.png"
        )
        logger.info(f"SHAP analysis complete for {model_type} model.")

        # LIME (for a single instance)
        if instance_index < len(data_sample):
            instance_for_lime = data_sample.iloc[instance_index]
            lime_explanation = explainer.explain_with_lime(data_sample, instance_for_lime, num_features=10)
            # Log or save LIME explanation
            lime_report_path = REPORTS_DIR / f"{model_type}_lime_instance_{instance_index}.txt"
            with open(lime_report_path, 'w') as f:
                f.write(f"LIME Explanation for instance {instance_index} using {model_type} model:\n")
                for feature, weight in lime_explanation.as_list(label=1): # Assuming label 1 is 'default'
                    f.write(f"{feature}: {weight:.4f}\n")
            logger.info(f"LIME explanation for instance {instance_index} ({model_type} model) saved to {lime_report_path}.")
        else:
            logger.warning("Instance index for LIME is out of bounds for the provided data sample.")

    def perform_bias_fairness_analysis(self, test_data: pd.DataFrame, true_labels: pd.Series, 
                                       sensitive_feature: str, model_type='ensemble'):
        """
        Performs bias and fairness analysis.
        :param test_data: DataFrame with features (including sensitive_feature) for fairness evaluation.
        :param true_labels: Series of true labels corresponding to test_data.
        :param sensitive_feature: Name of the sensitive column in test_data.
        :param model_type: Model to evaluate for fairness.
        """
        logger.info(f"Performing bias/fairness analysis for {model_type} model on feature '{sensitive_feature}'...")
        model_wrapper = getattr(self, f"{model_type}_model", None)
        if not model_wrapper:
            logger.error(f"{model_type} model not available for bias/fairness analysis.")
            return

        # The BiasFairnessAnalysis class expects the model wrapper itself, as it calls predict/predict_proba
        bias_analyzer = BiasFairnessAnalysis(
            model=model_wrapper, 
            X_test=test_data, 
            y_test=true_labels, 
            sensitive_feature_name=sensitive_feature,
            favorable_label=0, # Assuming 0 = not default (favorable)
            unfavorable_label=1  # Assuming 1 = default (unfavorable)
        )
        metrics = bias_analyzer.calculate_aif_metrics()
        if metrics:
            logger.info(f"Fairness metrics for {model_type} model ({sensitive_feature}): {metrics}")
            # Save metrics to a report
            report_path = REPORTS_DIR / f"{model_type}_fairness_report_{sensitive_feature}.txt"
            with open(report_path, 'w') as f:
                f.write(f"Fairness Report for {model_type} model, Sensitive Feature: {sensitive_feature}\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            logger.info(f"Fairness report saved to {report_path}")
        return metrics

    def save_simulation_results(self, filename_prefix='simulation_results'):
        """
        Saves the simulation results to a CSV file.
        """
        if self.simulation_results is not None:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = REPORTS_DIR / f"{filename_prefix}_{timestamp}.csv"
            self.simulation_results.to_csv(file_path, index=False)
            logger.info(f"Simulation results saved to {file_path}")
        else:
            logger.warning("No simulation results to save.")

if __name__ == '__main__':
    logger.info("Starting LendingSimulator example run...")
    
    # Basic configuration for the simulator
    sim_config = {
        'synthetic_data_params': {
            'num_traditional': 200,
            'num_thin_file': 50,
            'num_credit_invisible': 50,
            'random_seed': 42
        },
        'traditional_model_name': 'traditional_credit_model.joblib', # Ensure these models are trained and saved
        'alternative_model_name': 'alternative_credit_model.joblib',
        'ensemble_model_name': 'ensemble_model.joblib',
        'extract_alternative_features': True, # If alt features should be generated/used
        'decision_threshold': 0.3 # Example threshold for loan approval
    }

    simulator = LendingSimulator(config=sim_config)

    # 1. Load Applicant Data (using synthetic for this example)
    simulator.load_applicant_data(data_source_type='synthetic')

    if simulator.applicants_data is not None and not simulator.applicants_data.empty:
        # 2. Preprocess Data (Alternative feature extraction happens here if configured)
        # The main transformations (scaling/encoding) are within model wrappers.
        simulator.preprocess_data_for_models()

        # 3. Run Simulation (e.g., with ensemble model)
        results_df = simulator.run_simulation(model_type='ensemble')

        if results_df is not None:
            logger.info("\n--- Simulation Results (Sample) ---")
            print(results_df.head())

            # 4. Perform Explainability (on a sample of the input data)
            # Ensure 'defaulted' or the target variable is present if needed by explainers for context
            # The data_sample should be in the original feature space before model-specific transformations.
            sample_for_explain = simulator.applicants_data.sample(min(20, len(simulator.applicants_data)), random_state=123)
            # Define the original features the model was trained on (or expects)
            # This should align with what the model wrappers' FeatureTransformer expects.
            # For simplicity, assuming all columns in `sample_for_explain` are relevant original features.
            # This part needs careful alignment with how model wrappers store their expected input features.
            if simulator.ensemble_model and hasattr(simulator.ensemble_model, 'original_input_features'):
                 simulator.perform_explainability_analysis(sample_for_explain, model_type='ensemble', instance_index=0)
            else:
                logger.warning("Ensemble model or its original_input_features attribute not found, skipping explainability.")

            # 5. Perform Bias/Fairness Analysis
            # Requires a test set with true labels and a sensitive feature.
            # For this example, let's use a subset of generated data and assume 'demographic_group' is sensitive.
            if 'defaulted' in simulator.applicants_data.columns and 'demographic_group' in simulator.applicants_data.columns:
                fairness_test_data = simulator.applicants_data.sample(frac=0.3, random_state=7)
                true_labels_for_fairness = fairness_test_data['defaulted']
                # Ensure the model wrappers are compatible with BiasFairnessAnalysis
                if simulator.ensemble_model:
                    simulator.perform_bias_fairness_analysis(
                        test_data=fairness_test_data.drop(columns=['defaulted']),
                        true_labels=true_labels_for_fairness,
                        sensitive_feature='demographic_group', # Make sure this column exists
                        model_type='ensemble'
                    )
                else:
                    logger.warning("Ensemble model not available, skipping bias/fairness analysis.")
            else:
                logger.warning("Required columns ('defaulted', 'demographic_group') not in applicant data. Skipping bias/fairness analysis.")
        else:
            logger.error("Simulation did not produce results.")
    else:
        logger.error("Failed to load or generate applicant data for the simulation.")

    logger.info("LendingSimulator example run finished.")