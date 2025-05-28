from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import joblib
import uuid
from datetime import datetime

from ...api.schemas import PredictionInput, PredictionOutput, CombinedFeaturesInput
from ...utils.logger import get_logger
from ...config import MODELS_DIR
from ...modeling.traditional_model import TraditionalModel
from ...modeling.alternative_model import AlternativeModel
from ...modeling.ensemble_model import EnsembleModel
from ...feature_engineering.feature_transformer import FeatureTransformer # Assuming it's needed for direct loading

logger = get_logger(__name__)
router = APIRouter()

# --- Model Loading --- 
# This is a simplified way to load models. In a production system, you might have a more robust
# model registry, versioning, and dynamic loading mechanism (e.g., loading from MLflow).

# Global model cache (simple approach)
loaded_models = {}

def get_model_path(model_name: str):
    return MODELS_DIR / model_name

def load_model_from_path(model_path, model_class=None):
    try:
        if model_class and hasattr(model_class, 'load'):
            model_instance = model_class.load(model_path) # Use class-specific load method
            logger.info(f"Model {model_path.name} loaded successfully using {model_class.__name__}.load().")
            return model_instance
        else:
            # Generic joblib load for scikit-learn models or models saved directly with joblib
            model = joblib.load(model_path)
            logger.info(f"Model {model_path.name} loaded successfully using joblib.load().")
            return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise HTTPException(status_code=503, detail=f"Model file {model_path.name} not found. The service might be temporarily unavailable or not fully configured.")
    except Exception as e:
        logger.error(f"Error loading model {model_path.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not load model {model_path.name}. Error: {e}")

async def get_prediction_model(model_type: str):
    model_name_map = {
        "traditional": "traditional_credit_model.joblib", # Default names from model classes
        "alternative": "alternative_credit_model.joblib",
        "ensemble": "ensemble_model.joblib"
    }
    model_class_map = {
        "traditional": TraditionalModel,
        "alternative": AlternativeModel,
        "ensemble": EnsembleModel
    }

    if model_type not in model_name_map:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}. Choose from 'traditional', 'alternative', 'ensemble'.")

    model_filename = model_name_map[model_type]
    model_path = get_model_path(model_filename)
    model_class = model_class_map.get(model_type)

    if model_filename not in loaded_models:
        logger.info(f"Model {model_filename} not in cache. Attempting to load...")
        # The model classes (TraditionalModel, AlternativeModel, EnsembleModel) should handle their own loading
        # including their internal components like the actual predictor and feature transformer.
        # We pass the expected class to load_model_from_path to use its .load() method.
        loaded_models[model_filename] = load_model_from_path(model_path, model_class=model_class)
    
    return loaded_models[model_filename]

# --- Helper to convert Pydantic input to DataFrame --- 
def pydantic_to_dataframe(features_input: CombinedFeaturesInput) -> pd.DataFrame:
    # This needs to match the structure expected by your models' predict methods.
    # The model classes (TraditionalModel, etc.) expect a DataFrame with original feature names.
    data_dict = {}
    if features_input.traditional_features:
        data_dict.update(features_input.traditional_features.dict())
    if features_input.alternative_features:
        data_dict.update(features_input.alternative_features.dict())
    
    # Ensure all expected features are present, fill with None or default if necessary
    # This depends on how your FeatureTransformer handles missing columns during transform
    # For now, we assume the input schema enforces required fields.
    
    return pd.DataFrame([data_dict])

# --- Prediction Endpoint --- 
@router.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def make_prediction(input_data: PredictionInput):
    """
    Make a prediction using the specified model type.

    - **features**: Applicant's features, including traditional and optionally alternative data.
    - **model_type**: Type of model to use ('traditional', 'alternative', 'ensemble'). Default is 'ensemble'.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Received prediction request {request_id} for model_type: {input_data.model_type}")
    logger.debug(f"Input features for request {request_id}: {input_data.features.dict()}")

    try:
        model_wrapper = await get_prediction_model(input_data.model_type)
    except HTTPException as e: # Catch exceptions from get_prediction_model
        logger.error(f"Failed to load model for request {request_id}: {e.detail}")
        raise e # Re-raise the HTTPException
    except Exception as e:
        logger.error(f"Unexpected error loading model for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during model loading: {e}")

    # Convert Pydantic input to DataFrame
    try:
        features_df = pydantic_to_dataframe(input_data.features)
    except Exception as e:
        logger.error(f"Error converting input to DataFrame for request {request_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feature input format: {e}")

    # Make prediction using the model wrapper's predict/predict_proba methods
    try:
        if hasattr(model_wrapper, 'predict') and hasattr(model_wrapper, 'predict_proba'):
            prediction = model_wrapper.predict(features_df)[0] # Assuming single prediction
            probability = model_wrapper.predict_proba(features_df)[0, 1] # Prob of positive class
            # Ensure types are Python native for JSON serialization
            prediction = int(prediction)
            probability = float(probability)
        else:
            # This case might occur if a raw scikit-learn model was loaded without a wrapper
            # Or if the wrapper class doesn't conform to the expected interface.
            logger.error(f"Loaded model for {input_data.model_type} does not have standard predict/predict_proba methods.")
            raise HTTPException(status_code=500, detail="Model interface error.")

    except Exception as e:
        logger.error(f"Error during prediction for request {request_id} with model {input_data.model_type}: {e}")
        # Log more details if possible, e.g., features_df.to_dict() if not too large
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

    model_identifier = f"{input_data.model_type}_model_api_loaded"
    if hasattr(model_wrapper, 'model_name') and model_wrapper.model_name:
        model_identifier = model_wrapper.model_name
    elif hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, '__class__'): # For raw sklearn models
        model_identifier = f"{input_data.model_type}_{model_wrapper.model.__class__.__name__}"

    response = PredictionOutput(
        prediction=prediction,
        probability=probability,
        model_used=model_identifier,
        request_id=request_id,
        timestamp=datetime.utcnow()
    )
    logger.info(f"Prediction successful for request {request_id}. Output: {response.dict()}")
    return response

# Example: Endpoint to reload a specific model (for development/testing)
@router.post("/reload_model/{model_type}", tags=["Model Management - Dev"], include_in_schema=True) # Set include_in_schema=False for production
async def reload_model_endpoint(model_type: str):
    """
    Development endpoint to manually reload a model from disk.
    """
    model_name_map = {
        "traditional": "traditional_credit_model.joblib",
        "alternative": "alternative_credit_model.joblib",
        "ensemble": "ensemble_model.joblib"
    }
    if model_type not in model_name_map:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")
    
    model_filename = model_name_map[model_type]
    if model_filename in loaded_models:
        del loaded_models[model_filename]
        logger.info(f"Model {model_filename} removed from cache. Will be reloaded on next request.")
        return {"message": f"Model {model_filename} will be reloaded on next use."}
    else:
        logger.info(f"Model {model_filename} not currently in cache. Will be loaded on next use.")
        return {"message": f"Model {model_filename} was not in cache. Will be loaded on next use."}


# To test this router, you would include it in your main FastAPI app:
# from .routers import prediction_router
# app.include_router(prediction_router.router, prefix="/api/v1") # Example prefix