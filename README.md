# AI-Powered Consumer Lending Simulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A platform for simulating consumer lending decisions using AI, featuring synthetic data generation, advanced ML models, and a RESTful API.

## 🚀 Core Features

- **Synthetic Data Generation**: Create realistic applicant profiles.
- **Advanced ML Models**: Train traditional, alternative, and ensemble credit scoring models.
- **Lending Simulation**: Test lending strategies under various economic scenarios.
- **RESTful API**: Access functionalities programmatically via FastAPI.
- **CLI Interface**: Manage data, models, and simulations via command line.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/deluair/AI-Powered-Consumer-Lending-Simulation.git
    cd AI-Powered-Consumer-Lending-Simulation/lending_simulator_project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate
    
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: For Windows, if `shap` or `aif360` installation fails, you may need [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or to use WSL.*

4.  **Initialize project structure (optional, creates directories if not present):**
    ```bash
    python main.py setup-project-dirs
    ```

## 🖥️ Basic Usage (CLI)

Use `python main.py --help` to see all commands.

1.  **Generate Synthetic Data:**
    ```bash
    python main.py generate-synthetic-data --num-traditional 1000
    ```

2.  **Train a Model (e.g., traditional):**
    ```bash
    python main.py train-model traditional --data-path data/processed/processed_synthetic_data.csv
    ```

3.  **Run a Simulation:**
    ```bash
    python main.py run-simulation-cli --data-file-name processed_synthetic_data.csv --model-type traditional
    ```

4.  **Start the API Server:**
    ```bash
    python main.py run-api
    ```
    Access API docs at `http://127.0.0.1:8000/docs`.

## ⚙️ Configuration

- Main settings are in `src/config.py`.
- Create a `.env` file from `.env.example` for environment-specific settings.

## 🤝 Contributing

Contributions are welcome! Please refer to the full `README.md` in the repository for detailed guidelines (if available) or open an issue to discuss changes.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.