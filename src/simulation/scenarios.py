import pandas as pd
import numpy as np
from typing import Dict, Callable

from ..utils.logger import get_logger

logger = get_logger(__name__)

class ScenarioManager:
    def __init__(self, base_data: pd.DataFrame):
        """
        Initializes the ScenarioManager with base applicant data.
        :param base_data: The original DataFrame of applicant data to be modified by scenarios.
        """
        if not isinstance(base_data, pd.DataFrame):
            raise ValueError("base_data must be a pandas DataFrame.")
        self.base_data = base_data.copy() # Work on a copy
        self.scenarios: Dict[str, Callable[[pd.DataFrame, Dict], pd.DataFrame]] = {
            "economic_downturn": self._apply_economic_downturn,
            "improved_economy": self._apply_improved_economy,
            "shift_in_demographics": self._apply_demographic_shift,
            "interest_rate_hike": self._apply_interest_rate_hike,
            # Add more predefined scenarios here
        }
        logger.info("ScenarioManager initialized.")

    def apply_scenario(self, scenario_name: str, params: Dict = None) -> pd.DataFrame:
        """
        Applies a named scenario to a copy of the base data.
        :param scenario_name: The name of the scenario to apply (must be a key in self.scenarios).
        :param params: A dictionary of parameters specific to the scenario.
        :return: A new DataFrame with the scenario applied.
        """
        if scenario_name not in self.scenarios:
            logger.error(f"Scenario '{scenario_name}' not found.")
            raise ValueError(f"Scenario '{scenario_name}' not found. Available scenarios: {list(self.scenarios.keys())}")
        
        logger.info(f"Applying scenario: {scenario_name} with params: {params}")
        scenario_data = self.base_data.copy() # Always start from a fresh copy of base_data
        params = params or {}
        
        try:
            modified_data = self.scenarios[scenario_name](scenario_data, params)
            logger.info(f"Scenario '{scenario_name}' applied successfully.")
            return modified_data
        except Exception as e:
            logger.error(f"Error applying scenario '{scenario_name}': {e}", exc_info=True)
            # Return original data or raise, depending on desired behavior
            return self.base_data.copy() 

    def add_custom_scenario(self, scenario_name: str, scenario_function: Callable[[pd.DataFrame, Dict], pd.DataFrame]):
        """
        Adds a new custom scenario function.
        :param scenario_name: Name for the custom scenario.
        :param scenario_function: A function that takes a DataFrame and a params Dict, and returns a modified DataFrame.
        """
        if not callable(scenario_function):
            raise ValueError("scenario_function must be a callable.")
        self.scenarios[scenario_name] = scenario_function
        logger.info(f"Custom scenario '{scenario_name}' added.")

    # --- Predefined Scenario Implementations ---

    def _apply_economic_downturn(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Simulates an economic downturn.
        Effects:
        - Decrease in 'annual_income' by a certain percentage.
        - Increase in 'debt_to_income_ratio'.
        - Potential decrease in 'credit_score' for a segment of the population.
        - Higher probability of 'defaulted' (if 'defaulted' is a target to be simulated, not predicted).
        Params:
        - income_reduction_factor (float, e.g., 0.9 for 10% reduction)
        - dti_increase_factor (float, e.g., 1.1 for 10% increase)
        - credit_score_impact_segment_frac (float, e.g., 0.2 for 20% of population)
        - credit_score_reduction_points (int, e.g., 50 points)
        """
        logger.info("Applying economic downturn scenario...")
        modified_data = data.copy()

        income_reduction = params.get('income_reduction_factor', 0.9)
        dti_increase = params.get('dti_increase_factor', 1.1)
        cs_impact_frac = params.get('credit_score_impact_segment_frac', 0.2)
        cs_reduction = params.get('credit_score_reduction_points', 50)

        if 'annual_income' in modified_data.columns:
            modified_data['annual_income'] *= income_reduction
            modified_data['annual_income'] = modified_data['annual_income'].clip(lower=0) # Income cannot be negative
        
        if 'debt_to_income_ratio' in modified_data.columns:
            modified_data['debt_to_income_ratio'] *= dti_increase
        
        if 'credit_score' in modified_data.columns and cs_impact_frac > 0:
            impacted_indices = modified_data.sample(frac=cs_impact_frac, random_state=params.get('random_seed')).index
            modified_data.loc[impacted_indices, 'credit_score'] -= cs_reduction
            modified_data['credit_score'] = modified_data['credit_score'].clip(lower=300, upper=850) # Assuming standard score range

        # If 'defaulted' is part of the base_data and we want to simulate its change (not for model input features)
        # This is more for simulating ground truth changes, not for altering model inputs directly for prediction.
        # if 'defaulted' in modified_data.columns and params.get('increase_default_rate_factor'):
        #     # This is complex: would need to flip some non-defaulters to defaulters
        #     pass
        return modified_data

    def _apply_improved_economy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Simulates an improved economic climate.
        Effects:
        - Increase in 'annual_income'.
        - Decrease in 'debt_to_income_ratio'.
        - Potential increase in 'credit_score'.
        Params:
        - income_increase_factor (float, e.g., 1.1 for 10% increase)
        - dti_reduction_factor (float, e.g., 0.9 for 10% reduction)
        - credit_score_increase_points (int, e.g., 20 points)
        """
        logger.info("Applying improved economy scenario...")
        modified_data = data.copy()

        income_increase = params.get('income_increase_factor', 1.1)
        dti_reduction = params.get('dti_reduction_factor', 0.9)
        cs_increase = params.get('credit_score_increase_points', 20)

        if 'annual_income' in modified_data.columns:
            modified_data['annual_income'] *= income_increase
        
        if 'debt_to_income_ratio' in modified_data.columns:
            modified_data['debt_to_income_ratio'] *= dti_reduction
            modified_data['debt_to_income_ratio'] = modified_data['debt_to_income_ratio'].clip(lower=0)

        if 'credit_score' in modified_data.columns:
            modified_data['credit_score'] += cs_increase
            modified_data['credit_score'] = modified_data['credit_score'].clip(lower=300, upper=850)
        
        return modified_data

    def _apply_demographic_shift(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Simulates a shift in applicant demographics.
        Example: Increase in younger applicants or applicants from a specific region.
        This is highly dependent on the available features.
        Params:
        - feature_to_shift (str, e.g., 'age', 'region')
        - shift_params (dict, specific to the feature, e.g., for 'age': {'new_mean': 30, 'new_std': 5})
          or for categorical: {'target_category': 'Region_X', 'increase_proportion_by': 0.1}
        """
        logger.info("Applying demographic shift scenario...")
        modified_data = data.copy()
        feature = params.get('feature_to_shift')
        shift_params = params.get('shift_params', {})

        if not feature or feature not in modified_data.columns:
            logger.warning(f"Feature '{feature}' for demographic shift not found in data or not specified.")
            return modified_data

        if pd.api.types.is_numeric_dtype(modified_data[feature]):
            new_mean = shift_params.get('new_mean')
            new_std = shift_params.get('new_std', modified_data[feature].std())
            if new_mean is not None:
                current_mean = modified_data[feature].mean()
                modified_data[feature] = (modified_data[feature] - current_mean) + new_mean # Shift mean
                # Optionally, adjust std dev as well (more complex)
                # modified_data[feature] = new_mean + (modified_data[feature] - new_mean) * (new_std / current_std)
                if feature == 'age': # Example: clip age
                    modified_data[feature] = modified_data[feature].clip(lower=18).round()
        elif pd.api.types.is_categorical_dtype(modified_data[feature]) or modified_data[feature].dtype == 'object':
            target_category = shift_params.get('target_category')
            increase_proportion_by = shift_params.get('increase_proportion_by') # e.g., 0.1 for 10 percentage points
            if target_category and increase_proportion_by:
                # This is a simplified way to shift proportions. A more robust way might involve resampling.
                current_proportion = (modified_data[feature] == target_category).mean()
                num_to_change = int(len(modified_data) * increase_proportion_by)
                
                non_target_indices = modified_data[modified_data[feature] != target_category].index
                if len(non_target_indices) >= num_to_change > 0:
                    change_indices = np.random.choice(non_target_indices, size=num_to_change, replace=False)
                    modified_data.loc[change_indices, feature] = target_category
                else:
                    logger.warning(f"Not enough non-target samples to achieve desired shift for '{target_category}'.")
        else:
            logger.warning(f"Demographic shift for feature '{feature}' of type {modified_data[feature].dtype} not implemented.")
            
        return modified_data

    def _apply_interest_rate_hike(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Simulates an interest rate hike environment.
        This might affect DTI or affordability if those are calculated dynamically.
        If we have a feature like 'requested_loan_interest_rate_sensitivity', we could use it.
        For now, let's assume it primarily impacts 'debt_to_income_ratio' for new loan applications.
        Params:
        - dti_impact_factor (float, e.g., 1.05 for a 5% increase in DTI due to higher payments)
        """
        logger.info("Applying interest rate hike scenario...")
        modified_data = data.copy()
        dti_impact = params.get('dti_impact_factor', 1.05)

        if 'debt_to_income_ratio' in modified_data.columns:
            modified_data['debt_to_income_ratio'] *= dti_impact
        else:
            logger.warning("'debt_to_income_ratio' not in data, interest rate hike scenario may have limited effect.")
        
        return modified_data


if __name__ == '__main__':
    from ..data_ingestion.synthetic_data_generator import SyntheticDataGenerator

    logger.info("Running ScenarioManager example...")
    # Generate some base data
    sdg = SyntheticDataGenerator(random_seed=42)
    base_applicants = sdg.generate_dataset(num_traditional=100, num_thin_file=0, num_credit_invisible=0)
    base_applicants['age'] = np.random.randint(18, 65, size=len(base_applicants))
    base_applicants['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(base_applicants), p=[0.25,0.25,0.25,0.25])

    scenario_mgr = ScenarioManager(base_data=base_applicants)

    # Example 1: Economic Downturn
    downturn_params = {
        'income_reduction_factor': 0.85, 
        'dti_increase_factor': 1.2,
        'credit_score_impact_segment_frac': 0.3,
        'credit_score_reduction_points': 60,
        'random_seed': 123
    }
    downturn_data = scenario_mgr.apply_scenario('economic_downturn', downturn_params)
    logger.info("\n--- Economic Downturn Scenario Data (Sample) ---")
    print(downturn_data[['applicant_id', 'annual_income', 'debt_to_income_ratio', 'credit_score']].head())
    print("Original means:", base_applicants[['annual_income', 'debt_to_income_ratio', 'credit_score']].mean())
    print("Downturn means:", downturn_data[['annual_income', 'debt_to_income_ratio', 'credit_score']].mean())

    # Example 2: Demographic Shift (increase proportion of 'North' region)
    demographic_params = {
        'feature_to_shift': 'region',
        'shift_params': {'target_category': 'North', 'increase_proportion_by': 0.2} # Increase North proportion by 20pp
    }
    demographic_shift_data = scenario_mgr.apply_scenario('shift_in_demographics', demographic_params)
    logger.info("\n--- Demographic Shift Scenario Data (Region Counts) ---")
    print("Original Region Counts:\n", base_applicants['region'].value_counts(normalize=True))
    print("Shifted Region Counts:\n", demographic_shift_data['region'].value_counts(normalize=True))

    # Example 3: Custom Scenario
    def custom_policy_change_scenario(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        mod_data = data.copy()
        # Example: Stricter credit score cutoff for a segment
        if 'credit_score' in mod_data.columns and 'loan_product_type' in mod_data.columns:
            target_product = params.get('target_product', 'TypeA')
            new_min_score = params.get('new_min_score', 650)
            # This scenario doesn't filter, but could mark ineligible, or adjust a 'policy_score'
            # For simplicity, let's just log it. A real scenario would modify data for model input.
            logger.info(f"Custom: Applicants for {target_product} now notionally require score >= {new_min_score}")
        return mod_data

    scenario_mgr.add_custom_scenario('policy_change_A', custom_policy_change_scenario)
    # base_applicants['loan_product_type'] = np.random.choice(['TypeA', 'TypeB'], size=len(base_applicants))
    # custom_scenario_data = scenario_mgr.apply_scenario('policy_change_A', {'target_product': 'TypeA', 'new_min_score': 680})
    # print(custom_scenario_data.head())

    logger.info("ScenarioManager example finished.")