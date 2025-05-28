from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import FileResponse
import pandas as pd
import os
from typing import List, Dict, Any, Optional

from ...simulation.lending_simulator import LendingSimulator
from ...simulation.scenarios import ScenarioManager
from ...simulation.portfolio_analyzer import PortfolioAnalyzer
from ...data_ingestion.synthetic_data_generator import SyntheticDataGenerator # For generating base data
from ...config import (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, 
                       SYNTHETIC_DATA_GENERATION_CONFIG, SIMULATION_CONFIG)
from ...utils.logger import get_logger
from ..schemas import (SimulationRunConfig, ScenarioConfig, PortfolioAnalysisRequest,
                       SyntheticDataConfig, SimulationResponse, PortfolioMetricsResponse,
                       SegmentedPortfolioMetricsResponse)

logger = get_logger(__name__)
router = APIRouter()

# --- Helper Functions ---

def get_simulator_instance(model_type: str = 'ensemble') -> LendingSimulator:
    """Helper to initialize LendingSimulator with specified model type."""
    try:
        simulator = LendingSimulator(
            traditional_model_path=os.path.join(MODELS_DIR, SIMULATION_CONFIG.get('traditional_model_filename', 'traditional_model.joblib')),
            alternative_model_path=os.path.join(MODELS_DIR, SIMULATION_CONFIG.get('alternative_model_filename', 'alternative_model.joblib')),
            ensemble_model_path=os.path.join(MODELS_DIR, SIMULATION_CONFIG.get('ensemble_model_filename', 'ensemble_model.joblib'))
        )
        simulator.load_models(model_type=model_type)
        return simulator
    except Exception as e:
        logger.error(f"Error initializing LendingSimulator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize simulator: {str(e)}")

def load_applicant_data(data_file_name: Optional[str] = None) -> pd.DataFrame:
    """Loads applicant data from processed_data or generates synthetic if not provided."""
    if data_file_name:
        file_path = os.path.join(PROCESSED_DATA_DIR, data_file_name)
        if not os.path.exists(file_path):
            logger.error(f"Data file {file_path} not found.")
            raise HTTPException(status_code=404, detail=f"Data file '{data_file_name}' not found in processed data directory.")
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")
    else:
        logger.info("No data file provided, generating synthetic data for simulation.")
        sdg_config = SYNTHETIC_DATA_GENERATION_CONFIG
        sdg = SyntheticDataGenerator(random_seed=sdg_config.get('random_seed', 42))
        # Use a small default dataset for on-the-fly generation if no file is given
        return sdg.generate_dataset(
            num_traditional=sdg_config.get('num_traditional_profiles_simulation_default', 500),
            num_thin_file=sdg_config.get('num_thin_file_profiles_simulation_default', 100),
            num_credit_invisible=sdg_config.get('num_credit_invisible_profiles_simulation_default', 50)
        )

# --- API Endpoints ---

@router.post("/run_simulation", response_model=SimulationResponse)
async def run_simulation_endpoint(config: SimulationRunConfig = Body(...)):
    """
    Runs a lending simulation based on the provided configuration.
    
    - **data_file_name**: Optional. Name of the CSV file in `processed_data` directory.
                          If not provided, default synthetic data will be generated.
    - **model_type**: Model to use ('traditional', 'alternative', 'ensemble'). Default: 'ensemble'.
    - **scenario**: Optional. Configuration for applying a scenario before simulation.
        - **name**: Name of the scenario (e.g., 'economic_downturn').
        - **params**: Dictionary of parameters for the scenario.
    - **output_file_name**: Optional. If provided, simulation results will be saved to this CSV file in `reports` dir.
    - **run_explainability**: Whether to run SHAP explainability (can be slow). Default: False.
    - **run_bias_fairness**: Whether to run bias/fairness analysis. Default: False.
        - **protected_feature**: Name of the protected feature column for bias analysis.
        - **privileged_group**: Value representing the privileged group in the protected feature.
    """
    try:
        logger.info(f"Received simulation run request: {config}")
        simulator = get_simulator_instance(model_type=config.model_type)
        applicant_data = load_applicant_data(config.data_file_name)

        if config.scenario:
            logger.info(f"Applying scenario: {config.scenario.name} with params: {config.scenario.params}")
            scenario_manager = ScenarioManager(base_data=applicant_data)
            try:
                applicant_data = scenario_manager.apply_scenario(
                    scenario_name=config.scenario.name,
                    params=config.scenario.params or {}
                )
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        simulation_results_df = simulator.run_simulation(
            applicant_data=applicant_data,
            model_type_override=config.model_type # Ensure correct model is used
        )

        # Add actual_default for portfolio analysis (can be from original data or simulated)
        # For this example, if 'defaulted' exists in input, use it as 'actual_default'
        # Otherwise, simulate it for approved loans (e.g. based on predicted proba + randomness or a fixed rate)
        if 'defaulted' in applicant_data.columns:
            simulation_results_df['actual_default'] = applicant_data['defaulted']
        else: # Simulate actual_default for approved loans if not present
            approved_mask = simulation_results_df['decision'] == 'approved'
            # Simple simulation: 10% of approved loans default (replace with more sophisticated logic if needed)
            num_approved = approved_mask.sum()
            default_indices = simulation_results_df[approved_mask].sample(frac=0.1, random_state=config.random_seed or 42).index
            simulation_results_df['actual_default'] = 0
            simulation_results_df.loc[default_indices, 'actual_default'] = 1
            simulation_results_df.loc[simulation_results_df['decision'] == 'rejected', 'actual_default'] = pd.NA

        # Ensure required columns for PortfolioAnalyzer are present
        # These might come from applicant_data or be added if not present
        if 'loan_amount' not in simulation_results_df.columns and 'loan_amount' in applicant_data.columns:
            simulation_results_df = simulation_results_df.merge(applicant_data[['applicant_id', 'loan_amount']], on='applicant_id', how='left')
        elif 'loan_amount' not in simulation_results_df.columns:
             # Add a default if missing, for analyzer to work. This should ideally be in input data.
            simulation_results_df['loan_amount'] = 10000 
        
        if 'interest_rate' not in simulation_results_df.columns and 'interest_rate' in applicant_data.columns:
            simulation_results_df = simulation_results_df.merge(applicant_data[['applicant_id', 'interest_rate']], on='applicant_id', how='left')
        elif 'interest_rate' not in simulation_results_df.columns:
            simulation_results_df['interest_rate'] = 0.05 # Default annual rate

        if 'loan_term_months' not in simulation_results_df.columns and 'loan_term_months' in applicant_data.columns:
            simulation_results_df = simulation_results_df.merge(applicant_data[['applicant_id', 'loan_term_months']], on='applicant_id', how='left')
        elif 'loan_term_months' not in simulation_results_df.columns:
            simulation_results_df['loan_term_months'] = 36 # Default term

        # Fill NaNs that might cause issues in PortfolioAnalyzer for key financial columns on approved loans
        for col in ['loan_amount', 'interest_rate', 'loan_term_months']:
            if col in simulation_results_df.columns:
                approved_mask = simulation_results_df['decision'] == 'approved'
                if simulation_results_df.loc[approved_mask, col].isnull().any():
                    default_val = 10000 if col == 'loan_amount' else (0.05 if col == 'interest_rate' else 36)
                    logger.warning(f"Filling NaNs in '{col}' for approved loans with default value {default_val}.")
                    simulation_results_df.loc[approved_mask & simulation_results_df[col].isnull(), col] = default_val

        output_path = None
        if config.output_file_name:
            if not os.path.exists(REPORTS_DIR):
                os.makedirs(REPORTS_DIR)
            output_path = os.path.join(REPORTS_DIR, config.output_file_name)
            simulation_results_df.to_csv(output_path, index=False)
            logger.info(f"Simulation results saved to {output_path}")

        # Explainability and Bias/Fairness (Optional)
        shap_summary_plot_path: Optional[str] = None
        bias_fairness_report: Optional[Dict[str, Any]] = None

        if config.run_explainability:
            logger.info("Running SHAP explainability analysis...")
            # Ensure data for explainer is preprocessed correctly
            # This might require access to the feature transformer used by the model
            # For simplicity, assuming simulator handles preprocessing internally for explainability
            # Or, we pass the raw features that the model was trained on.
            # This part needs careful handling of feature transformations.
            # For now, let's assume the simulator's explainer can handle the applicant_data directly
            # or the relevant subset of features.
            try:
                # This needs the data that went *into* the model prediction step
                # If applicant_data was modified by scenario, use that.
                # The simulator's explainer should ideally use the same features as the model.
                shap_values_df, shap_plot = simulator.explain_predictions(
                    data_for_explanation=applicant_data, # Or a preprocessed version if needed
                    model_type_override=config.model_type,
                    top_n_features=10,
                    output_plot_path=os.path.join(REPORTS_DIR, "shap_summary_plot.png") if config.output_file_name else None
                )
                if shap_plot and config.output_file_name:
                    shap_summary_plot_path = os.path.join(REPORTS_DIR, "shap_summary_plot.png")
                logger.info("SHAP explainability analysis completed.")
            except Exception as e:
                logger.warning(f"Could not run SHAP explainability: {e}", exc_info=True)

        if config.run_bias_fairness and config.bias_fairness_config:
            logger.info("Running bias and fairness analysis...")
            if not config.bias_fairness_config.protected_feature or \
               not config.bias_fairness_config.privileged_group:
                raise HTTPException(status_code=400, detail="Protected feature and privileged group must be specified for bias analysis.")
            
            if config.bias_fairness_config.protected_feature not in applicant_data.columns:
                 raise HTTPException(status_code=400, detail=f"Protected feature '{config.bias_fairness_config.protected_feature}' not found in dataset.")

            try:
                # Bias analysis needs: features, true labels (if available for training context), predicted labels
                # For simulation, we use predicted decisions as 'labels'
                # True labels ('defaulted') are for evaluating model fairness if it was trained on them.
                # Here, we analyze fairness of *decisions* from the simulation.
                bias_fairness_report = simulator.analyze_bias_fairness(
                    data_with_predictions=simulation_results_df.merge(applicant_data[[config.bias_fairness_config.protected_feature, 'applicant_id']], on='applicant_id'),
                    protected_feature=config.bias_fairness_config.protected_feature,
                    favorable_outcome_value='approved', # Decision is favorable if 'approved'
                    privileged_groups=[{config.bias_fairness_config.protected_feature: config.bias_fairness_config.privileged_group}],
                    # actual_labels_col='defaulted' # if analyzing fairness of default prediction model
                    predictions_col='decision' # Analyzing fairness of approval decisions
                )
                logger.info("Bias and fairness analysis completed.")
            except Exception as e:
                logger.warning(f"Could not run bias/fairness analysis: {e}", exc_info=True)
                bias_fairness_report = {"error": str(e)}

        return SimulationResponse(
            message="Simulation completed successfully.",
            num_records_processed=len(simulation_results_df),
            model_used=config.model_type,
            scenario_applied=config.scenario.name if config.scenario else None,
            results_file_path=output_path,
            shap_summary_plot_path=shap_summary_plot_path,
            bias_fairness_report=bias_fairness_report
        )

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception in run_simulation: {http_exc.detail}")
        raise http_exc
    except ValueError as ve:
        logger.error(f"Value error in run_simulation: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in run_simulation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/analyze_portfolio", response_model=PortfolioMetricsResponse)
async def analyze_portfolio_endpoint(request: PortfolioAnalysisRequest = Body(...)):
    """
    Analyzes a portfolio based on simulation results stored in a CSV file.

    - **results_file_name**: Name of the CSV file in `reports` directory containing simulation results.
    - **loan_amount_col**: Column name for loan amounts. Default: 'loan_amount'.
    - **interest_rate_col**: Column name for annual interest rates. Default: 'interest_rate'.
    - **decision_col**: Column name for loan decisions. Default: 'decision'.
    - **actual_outcome_col**: Column name for actual default status. Default: 'actual_default'.
    - **loan_term_months_col**: Column name for loan term in months. Default: 'loan_term_months'.
    """
    try:
        logger.info(f"Received portfolio analysis request for file: {request.results_file_name}")
        file_path = os.path.join(REPORTS_DIR, request.results_file_name)
        if not os.path.exists(file_path):
            logger.error(f"Results file {file_path} not found.")
            raise HTTPException(status_code=404, detail=f"Results file '{request.results_file_name}' not found.")

        results_df = pd.read_csv(file_path)
        
        analyzer = PortfolioAnalyzer(
            simulation_results=results_df,
            loan_amount_col=request.loan_amount_col,
            interest_rate_col=request.interest_rate_col,
            decision_col=request.decision_col,
            actual_outcome_col=request.actual_outcome_col,
            loan_term_months_col=request.loan_term_months_col
        )
        
        summary_report = analyzer.generate_summary_report()
        return PortfolioMetricsResponse(metrics=summary_report)

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        logger.error(f"Value error in analyze_portfolio: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in analyze_portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/analyze_portfolio_segmented", response_model=SegmentedPortfolioMetricsResponse)
async def analyze_portfolio_segmented_endpoint(
    request: PortfolioAnalysisRequest = Body(...),
    segment_col: str = Query(..., description="Column name to segment the portfolio by (e.g., 'credit_score_band').")
):
    """
    Analyzes a portfolio segmented by a specified column.
    Uses the same request body as `/analyze_portfolio` plus a `segment_col` query parameter.
    """
    try:
        logger.info(f"Received segmented portfolio analysis request for file: {request.results_file_name}, segment by: {segment_col}")
        file_path = os.path.join(REPORTS_DIR, request.results_file_name)
        if not os.path.exists(file_path):
            logger.error(f"Results file {file_path} not found.")
            raise HTTPException(status_code=404, detail=f"Results file '{request.results_file_name}' not found.")

        results_df = pd.read_csv(file_path)
        
        if segment_col not in results_df.columns:
            raise HTTPException(status_code=400, detail=f"Segment column '{segment_col}' not found in the results file.")

        analyzer = PortfolioAnalyzer(
            simulation_results=results_df,
            loan_amount_col=request.loan_amount_col,
            interest_rate_col=request.interest_rate_col,
            decision_col=request.decision_col,
            actual_outcome_col=request.actual_outcome_col,
            loan_term_months_col=request.loan_term_months_col
        )
        
        segmented_report = analyzer.analyze_by_segment(segment_col=segment_col)
        # Convert segment keys to string if they are not (e.g. numpy int64)
        string_keyed_report = {str(k): v for k, v in segmented_report.items()}
        return SegmentedPortfolioMetricsResponse(segmented_metrics=string_keyed_report)

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        logger.error(f"Value error in analyze_portfolio_segmented: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in analyze_portfolio_segmented: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/available_scenarios", response_model=List[str])
async def list_available_scenarios():
    """
    Lists the names of predefined scenarios available in the ScenarioManager.
    """
    # Create a dummy ScenarioManager to access scenario names
    # This is a bit of a hack; ideally, scenario names would be statically available
    # or ScenarioManager would have a class method for this.
    try:
        # Need some dummy data to initialize ScenarioManager
        dummy_df = pd.DataFrame({'A': [1]}) 
        manager = ScenarioManager(base_data=dummy_df)
        return list(manager.scenarios.keys())
    except Exception as e:
        logger.error(f"Error listing available scenarios: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve scenario list.")

# Placeholder for an endpoint to get a sample scenario config
@router.get("/scenario_template/{scenario_name}", response_model=Dict[str, Any])
async def get_scenario_template(scenario_name: str):
    """
    Provides a template/example parameters for a given scenario.
    (This is a conceptual endpoint; actual parameters depend on scenario implementation)
    """
    # These are just examples, actual params are defined in ScenarioManager methods
    templates = {
        "economic_downturn": {
            "income_reduction_factor": 0.9, 
            "dti_increase_factor": 1.1,
            "credit_score_impact_segment_frac": 0.2,
            "credit_score_reduction_points": 50,
            "random_seed": 42
        },
        "improved_economy": {
            "income_increase_factor": 1.1, 
            "dti_reduction_factor": 0.9,
            "credit_score_increase_points": 20
        },
        "shift_in_demographics": {
            "feature_to_shift": "age", # or 'region', etc.
            "shift_params": {"new_mean": 30, "new_std": 8} # Example for 'age'
            # For categorical: "shift_params": {'target_category': 'North', 'increase_proportion_by': 0.1}
        },
        "interest_rate_hike": {
            "dti_impact_factor": 1.05
        }
    }
    if scenario_name not in templates:
        raise HTTPException(status_code=404, detail=f"Template for scenario '{scenario_name}' not found.")
    return templates[scenario_name]