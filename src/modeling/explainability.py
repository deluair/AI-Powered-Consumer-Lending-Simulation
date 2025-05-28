import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import joblib

from ..config import MODELS_DIR, REPORTS_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ModelExplainer:
    def __init__(self, model, feature_transformer, feature_names, mode='classification'):
        """
        Initialize the explainer.
        :param model: The trained model object (e.g., scikit-learn model).
        :param feature_transformer: The fitted FeatureTransformer object used for preprocessing.
        :param feature_names: List of original feature names before transformation.
        :param mode: 'classification' or 'regression'.
        """
        self.model = model
        self.feature_transformer = feature_transformer
        self.original_feature_names = feature_names # Names before any transformation
        self.transformed_feature_names = None # Names after transformation, to be set by transformer
        self.mode = mode
        self.shap_explainer = None
        self.lime_explainer = None

        if not hasattr(self.feature_transformer, 'transform') or not hasattr(self.feature_transformer, 'preprocessor'):
            raise ValueError("feature_transformer must be a valid preprocessor like scikit-learn's ColumnTransformer or our custom FeatureTransformer.")
        
        # Attempt to get transformed feature names
        try:
            # This is a bit fragile and depends on the FeatureTransformer's implementation
            # We need a dummy transform to get the output feature names if not directly available
            dummy_data = pd.DataFrame(np.zeros((1, len(self.original_feature_names))), columns=self.original_feature_names)
            # Fill with plausible dummy values for categorical features if transformer expects them
            for col in self.feature_transformer.categorical_cols:
                if col in dummy_data.columns:
                    # Get first category from one-hot encoder if possible, or use a placeholder
                    try:
                        cat_pipeline = dict(self.feature_transformer.preprocessor.named_transformers_)['cat']
                        ohe_categories = cat_pipeline.named_steps['onehot'].categories_
                        col_idx_in_cat_cols = self.feature_transformer.categorical_cols.index(col)
                        dummy_data[col] = ohe_categories[col_idx_in_cat_cols][0]
                    except:
                        dummy_data[col] = 'category_A' # Fallback dummy category
            
            transformed_dummy = self.feature_transformer.transform(dummy_data)
            self.transformed_feature_names = transformed_dummy.columns.tolist()
        except Exception as e:
            logger.warning(f"Could not automatically determine transformed feature names: {e}. SHAP/LIME might use generic names.")
            # Fallback: if model has feature_names_in_ attribute (like scikit-learn >= 1.0)
            if hasattr(self.model, 'feature_names_in_'):
                 self.transformed_feature_names = self.model.feature_names_in_.tolist()
            else:
                # This is a last resort and might not be accurate
                # self.transformed_feature_names = [f"feature_{i}" for i in range(transformed_dummy.shape[1] if 'transformed_dummy' in locals() else 10)]
                pass # Let SHAP/LIME try to infer or use default names

        logger.info("ModelExplainer initialized.")

    def _preprocess_for_lime(self, data_row: pd.Series):
        """
        Preprocesses a single row of data for LIME, which expects numpy array.
        LIME needs the data in the transformed state that the model consumes.
        """
        # Convert Series to DataFrame for the transformer
        data_df_row = pd.DataFrame([data_row])
        transformed_row_df = self.feature_transformer.transform(data_df_row)
        return transformed_row_df.iloc[0].to_numpy()

    def explain_with_shap(self, X_background: pd.DataFrame, X_explain: pd.DataFrame, 
                            save_plot_path=None, plot_type='summary'):
        """
        Generates SHAP explanations.
        :param X_background: DataFrame used to initialize the SHAP explainer (background data).
        :param X_explain: DataFrame of instances to explain.
        :param save_plot_path: Optional path to save SHAP plot.
        :param plot_type: 'summary', 'waterfall', 'force' (for single instance in X_explain for waterfall/force)
        """
        logger.info("Generating SHAP explanations...")
        
        # Transform data using the pipeline
        X_background_transformed = self.feature_transformer.transform(X_background)
        X_explain_transformed = self.feature_transformer.transform(X_explain)

        # Use the transformed feature names if available, otherwise SHAP might use default names
        if self.transformed_feature_names:
            X_background_transformed.columns = self.transformed_feature_names
            X_explain_transformed.columns = self.transformed_feature_names

        # SHAP explainer: KernelExplainer is model-agnostic but can be slow.
        # TreeExplainer for tree-based models (XGBoost, LightGBM, RandomForest) is faster.
        # DeepExplainer for neural networks.
        # LinearExplainer for linear models.
        if hasattr(self.model, 'coef_'): # Likely a linear model
            self.shap_explainer = shap.LinearExplainer(self.model, X_background_transformed)
        elif hasattr(self.model, 'feature_importances_'): # Likely a tree-based model
            self.shap_explainer = shap.TreeExplainer(self.model, X_background_transformed)
        else: # Fallback to KernelExplainer (slower)
            # KernelExplainer needs a predict_proba function for classification
            predict_fn = self.model.predict_proba if self.mode == 'classification' else self.model.predict
            self.shap_explainer = shap.KernelExplainer(predict_fn, X_background_transformed)

        shap_values = self.shap_explainer.shap_values(X_explain_transformed)
        logger.info("SHAP values generated.")

        # For classification, shap_values can be a list (one per class)
        # We usually care about the SHAP values for the positive class
        shap_values_for_plot = shap_values
        if self.mode == 'classification' and isinstance(shap_values, list):
            shap_values_for_plot = shap_values[1] # Assuming positive class is index 1

        # Plotting (optional)
        if plot_type == 'summary':
            shap.summary_plot(shap_values_for_plot, X_explain_transformed, show=False)
        elif plot_type == 'waterfall' and len(X_explain_transformed) == 1:
            # Waterfall plot is for a single instance
            shap.waterfall_plot(self.shap_explainer.expected_value[1] if self.mode == 'classification' and isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) and len(self.shap_explainer.expected_value)>1 else self.shap_explainer.expected_value , 
                                shap_values_for_plot[0], X_explain_transformed.iloc[0], show=False)
        elif plot_type == 'force' and len(X_explain_transformed) == 1:
            shap.force_plot(self.shap_explainer.expected_value[1] if self.mode == 'classification' and isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) and len(self.shap_explainer.expected_value)>1 else self.shap_explainer.expected_value,
                            shap_values_for_plot[0], X_explain_transformed.iloc[0], matplotlib=True, show=False)
        
        if save_plot_path:
            import matplotlib.pyplot as plt
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            plot_file = REPORTS_DIR / save_plot_path
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP plot saved to {plot_file}")

        return shap_values, X_explain_transformed # Return transformed data as well for context

    def explain_with_lime(self, X_train_original: pd.DataFrame, data_row_original: pd.Series, num_features=10):
        """
        Generates LIME explanation for a single instance.
        :param X_train_original: DataFrame of the original training data (untransformed) for LIME to sample from.
        :param data_row_original: A single instance (Pandas Series) in its original, untransformed state.
        :param num_features: Number of features to include in the explanation.
        """
        logger.info(f"Generating LIME explanation for instance: {data_row_original.name if hasattr(data_row_original, 'name') else 'Unnamed instance'}...")

        # LIME needs the training data in its original form to sample from, 
        # and then it applies transformations internally via the predict_fn.
        
        # Create LIME explainer
        # It needs original feature names and categorical feature indices
        categorical_indices = [X_train_original.columns.get_loc(col) for col in self.feature_transformer.categorical_cols if col in X_train_original.columns]
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_original.to_numpy(),
            feature_names=self.original_feature_names,
            class_names=['Not Default', 'Default'] if self.mode == 'classification' else None,
            categorical_features=categorical_indices,
            mode=self.mode,
            random_state=42
        )

        # LIME's predict_fn needs to take data in the original format (or a format LIME can sample from),
        # transform it, and then get predictions from the model.
        def lime_predict_fn(data_original_format_numpy):
            # Convert numpy array back to DataFrame with original feature names
            data_original_df = pd.DataFrame(data_original_format_numpy, columns=self.original_feature_names)
            # Transform the data using the pipeline
            data_transformed_df = self.feature_transformer.transform(data_original_df)
            if self.mode == 'classification':
                return self.model.predict_proba(data_transformed_df)
            else:
                return self.model.predict(data_transformed_df)
        
        # Explain the instance (original format)
        explanation = self.lime_explainer.explain_instance(
            data_row=data_row_original.to_numpy(),
            predict_fn=lime_predict_fn,
            num_features=num_features
        )
        logger.info("LIME explanation generated.")
        return explanation

if __name__ == '__main__':
    from ..data_ingestion.synthetic_data_generator import SyntheticDataGenerator
    from ..modeling.traditional_model import TraditionalModel # Using TraditionalModel as an example
    from sklearn.model_selection import train_test_split

    logger.info("Running ModelExplainer example...")
    sdg = SyntheticDataGenerator(random_seed=42)
    data_df = sdg.generate_dataset(num_traditional=300, num_thin_file=0, num_credit_invisible=0)
    
    trad_features_num = ['credit_score', 'months_credit_history', 'num_credit_accounts', 
                         'credit_utilization_ratio', 'debt_to_income_ratio', 'annual_income', 'age']
    trad_features_cat = [] # Example: add 'employment_status' if it were used and categorical
    
    data_df = data_df.dropna(subset=trad_features_num + ['defaulted'])
    if len(data_df) < 50: 
        logger.error("Not enough data for explainer example.")
        exit()

    X_original = data_df[trad_features_num + trad_features_cat]
    y = data_df['defaulted']
    
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, test_size=0.3, random_state=42, stratify=y)

    # Train a simple model for demonstration
    example_model_wrapper = TraditionalModel(model_name="explainer_trad_model.joblib")
    # The train method of TraditionalModel already handles feature transformation internally
    example_model_wrapper.train(X_train_orig, y_train, numerical_cols=trad_features_num, categorical_cols=trad_features_cat)
    
    # Now, initialize the explainer with the trained model and its transformer
    explainer = ModelExplainer(
        model=example_model_wrapper.model, # The actual scikit-learn model object
        feature_transformer=example_model_wrapper.feature_transformer, # The fitted FeatureTransformer
        feature_names=X_original.columns.tolist(), # Original feature names
        mode='classification'
    )

    # SHAP Explanations
    # Use a subset of X_train_orig as background for SHAP KernelExplainer (can be slow)
    # For TreeExplainer, background data might not be strictly needed or can be X_train_transformed
    # X_background_shap = shap.sample(X_train_orig, 50) # Sample 50 instances for background
    X_background_shap = X_train_orig.sample(min(50, len(X_train_orig)), random_state=42) 

    instances_to_explain_shap = X_test_orig.sample(min(5, len(X_test_orig)), random_state=1)
    
    logger.info(f"Explaining {len(instances_to_explain_shap)} instances with SHAP.")
    shap_values, X_explained_transformed_shap = explainer.explain_with_shap(
        X_background_shap, 
        instances_to_explain_shap, 
        save_plot_path="shap_summary_plot.png", 
        plot_type='summary'
    )
    # print("\nSHAP Values (first instance, positive class):")
    # if explainer.mode == 'classification' and isinstance(shap_values, list):
    #     print(pd.Series(shap_values[1][0], index=X_explained_transformed_shap.columns))
    # else:
    #     print(pd.Series(shap_values[0], index=X_explained_transformed_shap.columns))

    if len(instances_to_explain_shap) > 0:
        explainer.explain_with_shap(X_background_shap, instances_to_explain_shap.head(1), save_plot_path="shap_waterfall_plot.png", plot_type='waterfall')

    # LIME Explanation for one instance
    if not X_test_orig.empty:
        instance_to_explain_lime = X_test_orig.iloc[0]
        logger.info(f"\nExplaining one instance with LIME (Original features):\n{instance_to_explain_lime}")
        
        lime_explanation = explainer.explain_with_lime(X_train_orig, instance_to_explain_lime, num_features=5)
        print("\nLIME Explanation (Top 5 features for positive class):")
        for feature, weight in lime_explanation.as_list(label=1): # label=1 for positive class 'Default'
            print(f"{feature}: {weight:.4f}")
        
        # To save LIME plot (optional)
        # fig = lime_explanation.as_pyplot_figure(label=1)
        # fig.savefig(REPORTS_DIR / "lime_explanation.png")
        # logger.info(f"LIME plot saved to {REPORTS_DIR / 'lime_explanation.png'}")
    else:
        logger.warning("Skipping LIME example as X_test_orig is empty.")

    logger.info("ModelExplainer example finished.")