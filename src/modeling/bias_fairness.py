import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing
# from fairlearn.metrics import MetricFrame, count, selection_rate, false_positive_rate, true_positive_rate
# from fairlearn.postprocessing import ThresholdOptimizer
# from fairlearn.reductions import ExponentiatedGradient, GridSearch

from ..utils.logger import get_logger
from ..config import REPORTS_DIR

logger = get_logger(__name__)

class BiasFairnessAnalysis:
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                 sensitive_feature_name: str, favorable_label=1, unfavorable_label=0):
        """
        Initialize BiasFairnessAnalysis.
        :param model: Trained model object with a predict() method.
        :param X_test: Test features DataFrame.
        :param y_test: Test true labels Series.
        :param sensitive_feature_name: Name of the sensitive feature column in X_test.
        :param favorable_label: Label value considered favorable (e.g., 1 for 'loan approved' or 'not default').
        :param unfavorable_label: Label value considered unfavorable (e.g., 0 for 'loan denied' or 'default').
        """
        self.model = model
        self.X_test_orig = X_test.copy() # Keep original for reference
        self.y_test = y_test.copy()
        self.sensitive_feature_name = sensitive_feature_name
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label

        if self.sensitive_feature_name not in self.X_test_orig.columns:
            raise ValueError(f"Sensitive feature '{self.sensitive_feature_name}' not found in X_test columns.")

        self.y_pred = self.model.predict(self.X_test_orig)
        self.y_pred_proba = self.model.predict_proba(self.X_test_orig)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Prepare AIF360 dataset
        self.aif_dataset_test = self._prepare_aif_dataset(self.X_test_orig, self.y_test, self.y_pred)
        logger.info("BiasFairnessAnalysis initialized.")

    def _prepare_aif_dataset(self, X_df: pd.DataFrame, y_series: pd.Series, y_pred_series: pd.Series = None):
        """
        Prepares an AIF360 BinaryLabelDataset.
        """
        df_combined = X_df.copy()
        df_combined['true_label'] = y_series
        if y_pred_series is not None:
            df_combined['predicted_label'] = y_pred_series
        
        # AIF360 expects protected attributes to be numeric for some algorithms/metrics
        # If your sensitive feature is categorical, you might need to map it to numeric values
        # For simplicity, we assume it's already suitable or will be handled by AIF360's one-hot encoding if needed.
        
        # Identify privileged and unprivileged groups based on the sensitive feature
        # This is a simplified assumption; real-world scenarios might need more nuanced definitions.
        # Example: if 'gender' is 0 for female (unprivileged) and 1 for male (privileged)
        # For now, let's assume the sensitive feature has two distinct values, and we pick one as privileged.
        # A more robust way would be to define these based on domain knowledge or user input.
        sensitive_values = df_combined[self.sensitive_feature_name].unique()
        if len(sensitive_values) < 2:
            logger.warning(f"Sensitive feature '{self.sensitive_feature_name}' has fewer than 2 unique values. Metrics might not be meaningful.")
            # Defaulting to a dummy privileged/unprivileged structure if only one value exists
            privileged_groups = [{self.sensitive_feature_name: sensitive_values[0]}] if len(sensitive_values) > 0 else []
            unprivileged_groups = [] 
        else:
            # Heuristic: consider the group with the higher mean of the favorable true label as privileged
            # This is just one way; often this is predefined.
            group_means = df_combined.groupby(self.sensitive_feature_name)['true_label'].mean()
            privileged_value = group_means.idxmax() if not group_means.empty else sensitive_values[0]
            unprivileged_value = sensitive_values[0] if sensitive_values[0] != privileged_value else sensitive_values[1]
            
            privileged_groups = [{self.sensitive_feature_name: privileged_value}]
            unprivileged_groups = [{self.sensitive_feature_name: unprivileged_value}]
            logger.info(f"Identified privileged group: {privileged_groups}, unprivileged group: {unprivileged_groups} for '{self.sensitive_feature_name}'.")

        # Create BinaryLabelDataset
        # Note: AIF360's BinaryLabelDataset can take the DataFrame directly.
        # It will try to infer categorical features if `protected_attribute_names` are non-numeric.
        dataset = BinaryLabelDataset(
            df=df_combined,
            label_names=['true_label'],
            protected_attribute_names=[self.sensitive_feature_name],
            favorable_label=self.favorable_label,
            unfavorable_label=self.unfavorable_label,
            privileged_protected_attributes=privileged_groups, # This might need adjustment based on feature type
            unprivileged_protected_attributes=unprivileged_groups
        )
        
        # If predictions are available, create a dataset for the predicted labels as well
        if 'predicted_label' in df_combined.columns:
            dataset_pred = dataset.copy()
            dataset_pred.labels = df_combined[['predicted_label']].values
            return dataset, dataset_pred # Return both true and predicted label datasets
        
        return dataset, None

    def calculate_aif_metrics(self):
        """
        Calculates various fairness metrics using AIF360.
        """
        if self.aif_dataset_test[1] is None:
            logger.error("Predicted labels dataset not available for AIF360 metrics.")
            return None

        dataset_true, dataset_pred = self.aif_dataset_test
        
        # Assuming privileged_groups and unprivileged_groups were correctly set during _prepare_aif_dataset
        # If not, ClassificationMetric might require them explicitly.
        # For simplicity, we rely on the groups defined in the BinaryLabelDataset object.
        privileged_groups = dataset_true.privileged_groups
        unprivileged_groups = dataset_true.unprivileged_groups

        if not privileged_groups or not unprivileged_groups:
            logger.warning("Privileged or unprivileged groups not properly defined. AIF360 metrics might be incomplete.")
            # Fallback if groups are not set in dataset (should not happen with current _prepare_aif_dataset)
            # This part is tricky as it depends on how sensitive_feature_name is encoded.
            # For now, we assume _prepare_aif_dataset handles it.

        metric_pred = ClassificationMetric(
            dataset_true, dataset_pred, 
            unprivileged_groups=unprivileged_groups, 
            privileged_groups=privileged_groups
        )

        metrics = {
            'Statistical Parity Difference': metric_pred.statistical_parity_difference(),
            'Average Odds Difference': metric_pred.average_odds_difference(),
            'Equal Opportunity Difference': metric_pred.equal_opportunity_difference(),
            'Disparate Impact': metric_pred.disparate_impact(),
            'Theil Index': metric_pred.theil_index(),
            'False Positive Rate Difference': metric_pred.false_positive_rate_difference(),
            'True Positive Rate Difference': metric_pred.true_positive_rate_difference() # Same as EOD
        }
        logger.info(f"AIF360 Fairness Metrics: {metrics}")
        return metrics

    def apply_reweighing(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Applies reweighing preprocessing technique to the training data.
        Returns a new X_train, y_train, and sample_weights.
        This method should be called *before* training the model.
        """
        logger.info("Applying Reweighing pre-processing...")
        aif_dataset_train, _ = self._prepare_aif_dataset(X_train, y_train)
        
        RW = Reweighing(unprivileged_groups=aif_dataset_train.unprivileged_groups,
                        privileged_groups=aif_dataset_train.privileged_groups)
        
        dataset_train_reweighted = RW.fit_transform(aif_dataset_train)
        
        # Extract reweighted data and weights
        # AIF360 dataset features are numpy arrays. We need to convert back to DataFrame.
        X_train_reweighted_df = pd.DataFrame(dataset_train_reweighted.features, columns=X_train.columns)
        y_train_reweighted_series = pd.Series(dataset_train_reweighted.labels.ravel(), name=y_train.name)
        sample_weights = dataset_train_reweighted.instance_weights
        
        logger.info("Reweighing applied. Model should be trained with these instance weights.")
        return X_train_reweighted_df, y_train_reweighted_series, sample_weights

    def apply_eq_odds_postprocessing(self, unprivileged_groups=None, privileged_groups=None):
        """
        Applies Equalized Odds Postprocessing.
        This method modifies the predictions of the *already trained* model.
        Returns new y_pred. Note: This requires y_pred_proba.
        """
        if self.y_pred_proba is None:
            logger.error("predict_proba is required for EqOddsPostprocessing but not available from the model.")
            return self.y_pred # Return original predictions

        logger.info("Applying Equalized Odds Post-processing...")
        dataset_true, dataset_pred_orig = self.aif_dataset_test

        # Use groups from dataset if not provided
        unprivileged_groups = unprivileged_groups or dataset_true.unprivileged_groups
        privileged_groups = privileged_groups or dataset_true.privileged_groups

        if not privileged_groups or not unprivileged_groups:
             logger.error("Privileged/Unprivileged groups must be defined for EqOddsPostprocessing.")
             return self.y_pred

        # EqOddsPostprocessing needs scores (probabilities) for the favorable class
        # We need to ensure dataset_pred_orig has scores if the model provides them.
        # If not, it might try to use labels, which is less ideal.
        dataset_pred_with_scores = dataset_pred_orig.copy()
        dataset_pred_with_scores.scores = self.y_pred_proba.reshape(-1,1) # AIF360 expects 2D array for scores

        EOPP = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups,
                                    seed=42) # Seed for reproducibility
        
        EOPP = EOPP.fit(dataset_true, dataset_pred_with_scores) # Fit on true labels and predicted scores/labels
        dataset_pred_transformed = EOPP.predict(dataset_pred_with_scores) # Transform predictions
        
        y_pred_transformed = dataset_pred_transformed.labels.ravel()
        logger.info("Equalized Odds Post-processing applied.")
        return y_pred_transformed

    # Placeholder for Fairlearn integration (can be more complex)
    # def calculate_fairlearn_metrics(self):
    #     logger.info("Calculating Fairlearn metrics...")
    #     grouped_on_sensitive = MetricFrame(metrics=count,
    #                                        y_true=self.y_test,
    #                                        y_pred=self.y_pred,
    #                                        sensitive_features=self.X_test_orig[self.sensitive_feature_name])
    #     logger.info(f"Counts by group:\n{grouped_on_sensitive.by_group}")
    #     # Add more metrics like selection_rate, false_positive_rate etc.
    #     return grouped_on_sensitive.by_group

if __name__ == '__main__':
    from ..data_ingestion.synthetic_data_generator import SyntheticDataGenerator
    from ..modeling.traditional_model import TraditionalModel # Using TraditionalModel as an example
    from sklearn.model_selection import train_test_split

    logger.info("Running BiasFairnessAnalysis example...")
    sdg = SyntheticDataGenerator(random_seed=42)
    # Generate data with a potential sensitive feature, e.g., 'demographic_group'
    # For simplicity, let's add a binary demographic group to the synthetic data
    data_df = sdg.generate_dataset(num_traditional=500, num_thin_file=0, num_credit_invisible=0)
    data_df['demographic_group'] = np.random.choice([0, 1], size=len(data_df), p=[0.6, 0.4]) # 0: Unprivileged, 1: Privileged (example)

    trad_features_num = ['credit_score', 'months_credit_history', 'num_credit_accounts', 
                         'credit_utilization_ratio', 'debt_to_income_ratio', 'annual_income', 'age']
    trad_features_cat = [] # Add 'demographic_group' if it were to be one-hot encoded by FeatureTransformer
    sensitive_feature = 'demographic_group'
    
    # Ensure no NaNs in features or target
    all_features_for_model = trad_features_num + trad_features_cat
    # If sensitive_feature is not part of model features, it's only used for analysis
    # If it IS part of model features, it will be transformed by FeatureTransformer
    
    data_df = data_df.dropna(subset=all_features_for_model + [sensitive_feature, 'defaulted'])
    if len(data_df) < 100:
        logger.error("Not enough data for bias/fairness example after NaN drop. Exiting.")
        exit()

    X_original = data_df[all_features_for_model + ([sensitive_feature] if sensitive_feature not in all_features_for_model else [])]
    y = data_df['defaulted']
    
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- Option 1: Train model WITHOUT sensitive feature, then analyze --- 
    # model_features_train = X_train_orig.drop(columns=[sensitive_feature], errors='ignore')
    # model_features_test = X_test_orig.drop(columns=[sensitive_feature], errors='ignore')
    
    # --- Option 2: Train model WITH sensitive feature (potentially problematic) ---
    # For this example, let's assume the sensitive feature is NOT directly used for modeling
    # but is present in X_test_orig for fairness evaluation.
    # If it *were* used, it should be in trad_features_num or trad_features_cat.
    model_input_features_train = X_train_orig[all_features_for_model]
    model_input_features_test = X_test_orig[all_features_for_model]

    # Train a simple model
    example_model_wrapper = TraditionalModel(model_name="bias_trad_model.joblib")
    example_model_wrapper.train(model_input_features_train, y_train, 
                                numerical_cols=trad_features_num, 
                                categorical_cols=trad_features_cat)
    
    # Initialize BiasFairnessAnalysis
    # Pass X_test_orig which includes the sensitive feature for analysis
    # The model itself might not have been trained on it.
    bias_analyzer = BiasFairnessAnalysis(
        model=example_model_wrapper, # The wrapper which has predict/predict_proba
        X_test=X_test_orig, # Contains sensitive_feature for analysis
        y_test=y_test,
        sensitive_feature_name=sensitive_feature,
        favorable_label=0, # Assuming 0 = not defaulted (favorable)
        unfavorable_label=1  # Assuming 1 = defaulted (unfavorable)
    )

    logger.info("\n--- Initial Fairness Metrics ---")
    initial_metrics = bias_analyzer.calculate_aif_metrics()
    if initial_metrics:
        for metric, value in initial_metrics.items():
            print(f"{metric}: {value:.4f}")

    # --- Example: Apply Reweighing (Pre-processing) ---
    # This would typically be done *before* the main model training.
    # For demonstration, we'll show how to get weights, then one would retrain.
    logger.info("\n--- Applying Reweighing (Pre-processing) Demo ---")
    # We need the sensitive feature in X_train_orig for reweighing
    _, _, train_sample_weights = bias_analyzer.apply_reweighing(X_train_orig, y_train)
    logger.info(f"Generated {len(train_sample_weights)} sample weights for training.")
    # Now, one would retrain `example_model_wrapper` using these `train_sample_weights`
    # e.g., example_model_wrapper.train(..., sample_weight=train_sample_weights)
    # After retraining, re-evaluate fairness.

    # --- Example: Apply Equalized Odds Post-processing ---
    if example_model_wrapper.model_type == 'sklearn' and hasattr(example_model_wrapper.model, 'predict_proba'):
        logger.info("\n--- Applying Equalized Odds Post-processing ---")
        y_pred_transformed_eopp = bias_analyzer.apply_eq_odds_postprocessing()
        
        # Re-evaluate fairness with transformed predictions
        # Need to update the bias_analyzer's y_pred and re-prepare AIF dataset for new predictions
        bias_analyzer.y_pred = y_pred_transformed_eopp
        bias_analyzer.aif_dataset_test = bias_analyzer._prepare_aif_dataset(bias_analyzer.X_test_orig, bias_analyzer.y_test, bias_analyzer.y_pred)
        
        logger.info("\n--- Fairness Metrics After Equalized Odds Post-processing ---")
        eopp_metrics = bias_analyzer.calculate_aif_metrics()
        if eopp_metrics:
            for metric, value in eopp_metrics.items():
                print(f"{metric}: {value:.4f}")
    else:
        logger.warning("Skipping Equalized Odds Post-processing example as model does not have predict_proba or is not scikit-learn type.")

    logger.info("BiasFairnessAnalysis example finished.")