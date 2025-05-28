from fastapi import APIRouter, HTTPException
import time
import os

from ...api.schemas import SyntheticDataConfig, SyntheticDataResponse
from ...utils.logger import get_logger
from ...config import SYNTHETIC_DATA_DIR, PROJECT_ROOT # Assuming SYNTHETIC_DATA_DIR is defined in config
from ...data_ingestion.synthetic_data_generator import SyntheticDataGenerator

logger = get_logger(__name__)
router = APIRouter()

@router.post("/generate_synthetic_data", response_model=SyntheticDataResponse, tags=["Synthetic Data"])
async def generate_synthetic_data_endpoint(config: SyntheticDataConfig):
    """
    Generate synthetic lending data based on the provided configuration.

    - **num_traditional**: Number of traditional credit profiles.
    - **num_thin_file**: Number of thin-file credit profiles.
    - **num_credit_invisible**: Number of credit-invisible profiles.
    - **output_filename**: Optional name for the output CSV file.
    - **random_seed**: Optional random seed for reproducibility.
    """
    logger.info(f"Received request to generate synthetic data with config: {config.dict()}")    
    start_time = time.time()

    try:
        sdg = SyntheticDataGenerator(random_seed=config.random_seed)
        dataset = sdg.generate_dataset(
            num_traditional=config.num_traditional,
            num_thin_file=config.num_thin_file,
            num_credit_invisible=config.num_credit_invisible
        )
        
        num_records_generated = len(dataset)
        if num_records_generated == 0:
            logger.warning("No records were generated with the given configuration.")
            raise HTTPException(status_code=400, detail="No records generated. Check configuration parameters.")

        # Ensure the directory exists
        SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        output_file = SYNTHETIC_DATA_DIR / config.output_filename
        sdg.save_dataset(dataset, output_file)
        
        end_time = time.time()
        generation_time = end_time - start_time

        # Get relative path for the response if needed, or absolute
        # For simplicity, returning absolute path for now.
        # relative_output_path = os.path.relpath(output_file, PROJECT_ROOT)

        logger.info(f"Synthetic data generated successfully: {output_file}. Records: {num_records_generated}. Time: {generation_time:.2f}s")
        return SyntheticDataResponse(
            message="Synthetic data generated successfully.",
            file_path=str(output_file),
            num_records_generated=num_records_generated,
            generation_time_seconds=round(generation_time, 2)
        )

    except ValueError as ve:
        logger.error(f"ValueError during synthetic data generation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during synthetic data generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Example of how to include this router in main.py:
# from .routers import synthetic_data_router
# app.include_router(synthetic_data_router.router, prefix="/data_tools", tags=["Data Tools"])