from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- General Schemas ---
class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    version: str

class MessageResponse(BaseModel):
    message: str

# --- Prediction Schemas ---
# These would mirror the input features expected by your models

class TraditionalFeaturesInput(BaseModel):
    credit_score: int = Field(..., example=700, description="Applicant's credit score")
    months_credit_history: int = Field(..., example=120, description="Length of credit history in months")
    num_credit_accounts: int = Field(..., example=5, description="Number of active credit accounts")
    credit_utilization_ratio: float = Field(..., example=0.3, ge=0, le=1, description="Credit utilization ratio")
    debt_to_income_ratio: float = Field(..., example=0.4, ge=0, description="Debt to income ratio")
    annual_income: float = Field(..., example=60000, ge=0, description="Applicant's annual income")
    age: int = Field(..., example=35, ge=18, description="Applicant's age")
    # Add any other traditional features your model expects
    # employment_status: Optional[str] = Field(None, example="Employed")

class AlternativeFeaturesInput(BaseModel):
    # Example alternative features - adjust based on your actual features
    avg_utility_payment_timeliness: Optional[float] = Field(None, example=0.95, ge=0, le=1, description="Average utility payment timeliness score")
    digital_footprint_score: Optional[float] = Field(None, example=0.75, ge=0, le=1, description="Score based on digital footprint analysis")
    education_level: Optional[str] = Field(None, example="Bachelor's Degree", description="Highest level of education")
    # ... other alternative features

class CombinedFeaturesInput(BaseModel):
    traditional_features: TraditionalFeaturesInput
    alternative_features: Optional[AlternativeFeaturesInput] = None
    # You might also have a flat structure if preferred:
    # credit_score: int
    # ... and all other features directly here

class PredictionInput(BaseModel):
    # This could be a single instance or a list for batch predictions
    # For simplicity, let's assume a single instance for now
    # features: Dict[str, Any] # A more generic way to pass features
    features: CombinedFeaturesInput
    model_type: str = Field("ensemble", example="ensemble", description="Type of model to use: 'traditional', 'alternative', 'ensemble'")

    @validator('model_type')
    def model_type_must_be_valid(cls, value):
        if value not in ['traditional', 'alternative', 'ensemble']:
            raise ValueError("model_type must be 'traditional', 'alternative', or 'ensemble'")
        return value

class PredictionOutput(BaseModel):
    prediction: int = Field(..., example=0, description="Predicted class (e.g., 0 for not default, 1 for default)")
    probability: Optional[float] = Field(None, example=0.15, ge=0, le=1, description="Predicted probability of the positive class")
    model_used: str = Field(..., example="ensemble_v1.0", description="Identifier of the model version used")
    request_id: Optional[str] = Field(None, example="xyz-123", description="Unique ID for the prediction request")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# --- Data Ingestion Schemas (Example) ---
class FileUploadResponse(BaseModel):
    filename: str
    content_type: str
    size_kb: float
    message: str = "File uploaded successfully"

# --- Model Management Schemas (Example) ---
class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    description: Optional[str] = None
    trained_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None # e.g., {"auc": 0.85, "accuracy": 0.90}

class ModelListResponse(BaseModel):
    models: List[ModelInfo]

# --- Synthetic Data Generation Schemas ---
class SyntheticDataConfig(BaseModel):
    num_traditional: int = Field(100, ge=0, description="Number of traditional profiles to generate")
    num_thin_file: int = Field(50, ge=0, description="Number of thin-file profiles to generate")
    num_credit_invisible: int = Field(50, ge=0, description="Number of credit-invisible profiles to generate")
    output_filename: Optional[str] = Field("synthetic_lending_data.csv", description="Filename for the generated data")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class SyntheticDataResponse(BaseModel):
    message: str
    file_path: Optional[str] = None
    num_records_generated: int
    generation_time_seconds: Optional[float] = None

# Example of a more complex nested schema if needed
# class LoanApplicationInput(BaseModel):
#     applicant_id: str
#     application_data: CombinedFeaturesInput
#     requested_loan_amount: float

if __name__ == '__main__':
    # Example usage for validation and documentation generation
    sample_traditional_features = {
        "credit_score": 720,
        "months_credit_history": 60,
        "num_credit_accounts": 8,
        "credit_utilization_ratio": 0.25,
        "debt_to_income_ratio": 0.35,
        "annual_income": 75000,
        "age": 40
    }
    trad_input = TraditionalFeaturesInput(**sample_traditional_features)
    print(f"Traditional Input Example: {trad_input.json(indent=2)}")

    sample_combined_features = {
        "traditional_features": sample_traditional_features,
        "alternative_features": {
            "avg_utility_payment_timeliness": 0.98,
            "digital_footprint_score": 0.80
        }
    }
    comb_input = CombinedFeaturesInput(**sample_combined_features)
    print(f"\nCombined Input Example: {comb_input.json(indent=2)}")

    pred_input_payload = {
        "features": comb_input.dict(),
        "model_type": "ensemble"
    }
    pred_input = PredictionInput(**pred_input_payload)
    print(f"\nPrediction Input Payload Example: {pred_input.json(indent=2)}")

    pred_output = PredictionOutput(
        prediction=0,
        probability=0.123,
        model_used="ensemble_model_v1.2.3",
        request_id="abc-789"
    )
    print(f"\nPrediction Output Example: {pred_output.json(indent=2)}")

    synth_config = SyntheticDataConfig(num_traditional=10, num_thin_file=5, num_credit_invisible=5)
    print(f"\nSynthetic Data Config Example: {synth_config.json(indent=2)}")