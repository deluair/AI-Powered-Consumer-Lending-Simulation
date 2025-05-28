from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..config import API_HOST, API_PORT, API_RELOAD, LOG_LEVEL, PROJECT_NAME, VERSION
from ..utils.logger import get_logger
from .routers import prediction_router, synthetic_data_router, simulation_router # Updated import
# from .routers import data_router, model_management_router # Kept for potential future use

logger = get_logger(__name__)

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="A FastAPI application for the Consumer Lending Simulation project."
)

# CORS (Cross-Origin Resource Sharing) Middleware
# Allows requests from all origins, methods, and headers. 
# Adjust origins for production environments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify allowed origins: ["http://localhost", "http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")
    # Initialize any resources here if needed (e.g., DB connections, ML models)
    # Example: load_models()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    # Clean up resources here if needed
    # Example: close_db_connections()

@app.get("/", tags=["General"])
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": f"Welcome to {PROJECT_NAME} API v{VERSION}"}

@app.get("/health", tags=["General"])
async def health_check():
    logger.debug("Health check endpoint accessed.")
    return {"status": "healthy", "version": VERSION}

# Include routers for different parts of the API
app.include_router(prediction_router.router, prefix="/api/v1/predict", tags=["Predictions"])
app.include_router(synthetic_data_router.router, prefix="/api/v1/data", tags=["Data Generation"])
app.include_router(simulation_router.router, prefix="/api/v1/simulation", tags=["Lending Simulation"])
# app.include_router(data_router.router, prefix="/data", tags=["Data Management"]) # Kept for potential future use
# app.include_router(model_management_router.router, prefix="/models", tags=["Model Management"]) # Kept for potential future use

# Example endpoint removed as actual routers are now included


if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {API_HOST}:{API_PORT} with reload={API_RELOAD} and log_level={LOG_LEVEL.lower()}")
    uvicorn.run(
        "main:app", 
        host=API_HOST, 
        port=API_PORT, 
        reload=API_RELOAD, 
        log_level=LOG_LEVEL.lower()
    )
    # To run this: python -m src.api.main
    # Or if uvicorn is installed globally: uvicorn src.api.main:app --reload