import typer
import uvicorn
import os
import pandas as pd 

from src.utils.logger import get_logger
from src.config import (
    API_HOST, API_PORT, API_RELOAD, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
    SYNTHETIC_DATA_GENERATION_CONFIG, TRADITIONAL_MODEL_CONFIG,
    ALTERNATIVE_MODEL_CONFIG, ENSEMBLE_MODEL_CONFIG, SIMULATION_CONFIG
)

# Conditional imports for functionality to avoid loading everything at once

app = typer.Typer(help="CLI for the AI-Powered Consumer Lending Simulation project.")
logger = get_logger(__name__)

@app.command()
def run_api(
    host: str = typer.Option(API_HOST, "--host", "-h", help="Host to run the API on."),
    port: int = typer.Option(API_PORT, "--port", "-p", help="Port to run the API on."),
    reload: bool = typer.Option(API_RELOAD, "--reload", help="Enable auto-reload for development.")
):
    """Starts the FastAPI application server."""
    logger.info(f"Starting API server on {host}:{port} with reload={'enabled' if reload else 'disabled'}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)

@app.command()
def generate_synthetic_data(
    num_traditional: int = typer.Option(SYNTHETIC_DATA_GENERATION_CONFIG.get('num_traditional_profiles', 1000), help="Number of traditional profiles."),
    num_thin_file: int = typer.Option(SYNTHETIC_DATA_GENERATION_CONFIG.get('num_thin_file_profiles', 200), help="Number of thin-file profiles."),
    num_credit_invisible: int = typer.Option(SYNTHETIC_DATA_GENERATION_CONFIG.get('num_credit_invisible_profiles', 100), help="Number of credit-invisible profiles."),
    output_filename: str = typer.Option(SYNTHETIC_DATA_GENERATION_CONFIG.get('default_output_filename', 'synthetic_lending_data.csv'), help="Output CSV filename."),
    output_dir: str = typer.Option(RAW_DATA_DIR, help="Directory to save the generated data.")
):
    """Generates synthetic lending data and saves it to a CSV file."""
    from src.data_ingestion.synthetic_data_generator import SyntheticDataGenerator # Local import
    logger.info(f"Generating synthetic data: {num_traditional}T, {num_thin_file}TF, {num_credit_invisible}CI")
    generator = SyntheticDataGenerator(random_seed=SYNTHETIC_DATA_GENERATION_CONFIG.get('random_seed', 42))
    dataset = generator.generate_dataset(num_traditional, num_thin_file, num_credit_invisible)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_filename)
    dataset.to_csv(output_path, index=False)
    logger.info(f"Synthetic data saved to {output_path}")

@app.command()
def train_model(
    model_type: str = typer.Argument(..., help="Type of model to train ('traditional', 'alternative', 'ensemble')."),
    data_path: str = typer.Option(os.path.join(PROCESSED_DATA_DIR, SIMULATION_CONFIG.get('default_training_data_filename','processed_synthetic_data.csv')), help="Path to the training data CSV file."),
    output_dir: str = typer.Option(MODELS_DIR, help="Directory to save the trained model.")
):
    """Trains a specified model type using data from data_path and saves it."""
    logger.info(f"Starting training for {model_type} model using data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}. Please generate or place data first.")
        raise typer.Exit(code=1)
        
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Failed to load training data from {data_path}: {e}")
        raise typer.Exit(code=1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_type == 'traditional':
        from src.modeling.traditional_model import TraditionalModel # Local import
        model_config = TRADITIONAL_MODEL_CONFIG
        model = TraditionalModel(
            target_column=model_config.get('target_column', 'defaulted'),
            test_size=model_config.get('test_size', 0.2),
            random_state=model_config.get('random_state', 42)
        )
        model.train(data)
        model_filename = model_config.get('output_filename', 'traditional_model.joblib')
        model.save_model(os.path.join(output_dir, model_filename))
        logger.info(f"Traditional model trained and saved to {os.path.join(output_dir, model_filename)}")
    elif model_type == 'alternative':
        from src.modeling.alternative_model import AlternativeModel # Local import
        model_config = ALTERNATIVE_MODEL_CONFIG
        model = AlternativeModel(
            target_column=model_config.get('target_column', 'defaulted'),
            test_size=model_config.get('test_size', 0.2),
            random_state=model_config.get('random_state', 42)
        )
        # Alternative model might need specific feature columns
        # This needs to be handled based on how AlternativeModel expects data
        model.train(data) 
        model_filename = model_config.get('output_filename', 'alternative_model.joblib')
        model.save_model(os.path.join(output_dir, model_filename))
        logger.info(f"Alternative model trained and saved to {os.path.join(output_dir, model_filename)}")
    elif model_type == 'ensemble':
        from src.modeling.ensemble_model import EnsembleModel # Local import
        ensemble_config = ENSEMBLE_MODEL_CONFIG
        traditional_model_path = os.path.join(MODELS_DIR, TRADITIONAL_MODEL_CONFIG.get('output_filename', 'traditional_model.joblib'))
        alternative_model_path = os.path.join(MODELS_DIR, ALTERNATIVE_MODEL_CONFIG.get('output_filename', 'alternative_model.joblib'))

        if not os.path.exists(traditional_model_path) or not os.path.exists(alternative_model_path):
            logger.error("Base models (traditional and/or alternative) not found. Train them first.")
            raise typer.Exit(code=1)

        model = EnsembleModel(
            traditional_model_path=traditional_model_path,
            alternative_model_path=alternative_model_path,
            target_column=ensemble_config.get('target_column', 'defaulted'),
            test_size=ensemble_config.get('test_size', 0.2),
            random_state=ensemble_config.get('random_state', 42)
        )
        model.train_meta_learner(data)
        model_filename = ensemble_config.get('output_filename', 'ensemble_model.joblib')
        model.save_model(os.path.join(output_dir, model_filename))
        logger.info(f"Ensemble model trained and saved to {os.path.join(output_dir, model_filename)}")
    else:
        logger.error(f"Unsupported model type: {model_type}. Choose from 'traditional', 'alternative', 'ensemble'.")
        raise typer.Exit(code=1)

@app.command()
def run_simulation_cli(
    data_file_name: str = typer.Option(None, help="Name of the CSV file in processed_data. If None, uses default synthetic data."),
    model_type: str = typer.Option('ensemble', help="Model to use ('traditional', 'alternative', 'ensemble')."),
    scenario_name: str = typer.Option(None, help="Name of scenario to apply (e.g., 'economic_downturn')."),
    # Scenario params would be complex for CLI, better handled via API or config file for CLI
    output_file_name: str = typer.Option("cli_simulation_results.csv", help="Output CSV filename for simulation results in reports dir.")
):
    """Runs a lending simulation from the CLI (simplified version)."""
    from src.simulation.lending_simulator import LendingSimulator # Local import
    from src.simulation.scenarios import ScenarioManager # Local import
    from src.api.routers.simulation_router import load_applicant_data # Re-use loader

    logger.info(f"Running CLI simulation with model: {model_type}, data: {data_file_name or 'default synthetic'}, scenario: {scenario_name or 'None'}")

    try:
        simulator = LendingSimulator(
            traditional_model_path=os.path.join(MODELS_DIR, TRADITIONAL_MODEL_CONFIG.get('output_filename', 'traditional_model.joblib')),
            alternative_model_path=os.path.join(MODELS_DIR, ALTERNATIVE_MODEL_CONFIG.get('output_filename', 'alternative_model.joblib')),
            ensemble_model_path=os.path.join(MODELS_DIR, ENSEMBLE_MODEL_CONFIG.get('output_filename', 'ensemble_model.joblib'))
        )
        simulator.load_models(model_type=model_type)
        applicant_data = load_applicant_data(data_file_name) # Uses the same logic as API

        if scenario_name:
            logger.info(f"Applying scenario: {scenario_name}")
            # For CLI, scenario params might need to be predefined or loaded from a file
            # Using empty params for now, or a default set from config if available.
            scenario_params = SIMULATION_CONFIG.get('default_scenario_params', {}).get(scenario_name, {})
            scenario_manager = ScenarioManager(base_data=applicant_data)
            try:
                applicant_data = scenario_manager.apply_scenario(scenario_name, scenario_params)
            except ValueError as ve:
                logger.error(f"Scenario error: {ve}")
                raise typer.Exit(code=1)
        
        results_df = simulator.run_simulation(applicant_data, model_type_override=model_type)

        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
        output_path = os.path.join(REPORTS_DIR, output_file_name)
        results_df.to_csv(output_path, index=False)
        logger.info(f"CLI Simulation results saved to {output_path}")
        print(f"Simulation results saved to {output_path}")
        print(results_df.head())

    except Exception as e:
        logger.error(f"CLI Simulation failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def setup_project_dirs():
    """Creates necessary project directories if they don't exist."""
    dirs_to_create = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, "data/synthetic"]
    for proj_dir in dirs_to_create:
        if not os.path.exists(proj_dir):
            os.makedirs(proj_dir)
            logger.info(f"Created directory: {proj_dir}")
            # Add .gitkeep file to ensure empty dirs are tracked by Git
            with open(os.path.join(proj_dir, ".gitkeep"), "w") as f:
                pass # Create empty file
        else:
            logger.info(f"Directory already exists: {proj_dir}")
    logger.info("Project directories checked/created.")

if __name__ == "__main__":
    # Create directories first if running main.py directly for the first time
    # This is helpful for initial setup, though `setup-project-dirs` command is preferred.
    # For CLI usage, Typer handles this implicitly when commands are run.
    # setup_project_dirs() # Optionally run this on direct script execution
    app()