import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

from ..config import (
    SYNTHETIC_DATA_DIR,
    NUM_TRADITIONAL_PROFILES,
    NUM_THIN_FILE_PROFILES,
    NUM_CREDIT_INVISIBLE_PROFILES,
    TOTAL_SYNTHETIC_PROFILES
)
from ..utils.logger import get_logger

logger = get_logger(__name__)
fake = Faker()

class SyntheticDataGenerator:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        Faker.seed(self.random_seed)
        logger.info(f"SyntheticDataGenerator initialized with random_seed={random_seed}")

    def _generate_base_profile(self, user_id_prefix='USER_') -> dict:
        """Generates common attributes for any profile."""
        profile = {}
        profile['user_id'] = f"{user_id_prefix}{fake.uuid4()[:8]}"
        profile['age'] = np.random.randint(18, 75)
        profile['gender'] = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
        profile['zip_code'] = fake.zipcode()
        profile['employment_status'] = np.random.choice(
            ['Employed', 'Unemployed', 'Self-Employed', 'Student', 'Retired'], 
            p=[0.6, 0.1, 0.15, 0.1, 0.05]
        )
        profile['annual_income'] = max(0, np.random.normal(loc=60000, scale=30000))
        if profile['employment_status'] == 'Unemployed':
            profile['annual_income'] = max(0, np.random.normal(loc=15000, scale=5000))
        elif profile['employment_status'] == 'Student':
            profile['annual_income'] = max(0, np.random.normal(loc=10000, scale=3000))
        
        profile['education_level'] = np.random.choice(
            ['High School', 'Some College', "Bachelor's Degree", "Master's Degree", "PhD"],
            p=[0.25, 0.25, 0.3, 0.15, 0.05]
        )
        return profile

    def _generate_traditional_financial_profile(self) -> dict:
        """Generates a profile with traditional credit history."""
        profile = self._generate_base_profile(user_id_prefix='TRAD_')
        profile['profile_type'] = 'Traditional'
        profile['credit_score'] = np.random.randint(500, 850) # Skewed higher for traditional
        profile['months_credit_history'] = np.random.randint(24, 300)
        profile['num_credit_accounts'] = np.random.randint(2, 15)
        profile['credit_utilization_ratio'] = round(np.random.uniform(0.05, 0.8), 2)
        profile['debt_to_income_ratio'] = round(np.random.uniform(0.1, 0.6), 2)
        profile['has_mortgage'] = np.random.choice([True, False], p=[0.4, 0.6])
        profile['num_late_payments_last_12m'] = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
        profile['loan_purpose'] = fake.random_element(elements=('Debt Consolidation', 'Home Improvement', 'Car Purchase', 'Medical Expenses', 'Other'))
        profile['requested_loan_amount'] = round(np.random.uniform(1000, 50000), -2)
        # Simulate default probability based on credit score and DTI (simplified)
        default_prob = 0.02 + 0.5 * (1 - (profile['credit_score'] - 300)/550) + 0.3 * profile['debt_to_income_ratio']
        profile['defaulted'] = np.random.binomial(1, min(max(default_prob,0.01),0.95)) 
        return profile

    def _generate_thin_file_profile(self) -> dict:
        """Generates a profile for a thin-file consumer."""
        profile = self._generate_base_profile(user_id_prefix='THIN_')
        profile['profile_type'] = 'Thin-File'
        profile['credit_score'] = np.random.randint(300, 650) # Generally lower or non-existent
        profile['months_credit_history'] = np.random.randint(0, 24)
        profile['num_credit_accounts'] = np.random.randint(0, 3)
        profile['credit_utilization_ratio'] = None if profile['num_credit_accounts'] == 0 else round(np.random.uniform(0.0, 0.5), 2)
        profile['debt_to_income_ratio'] = round(np.random.uniform(0.0, 0.4), 2) if profile['annual_income'] > 0 else None
        profile['has_mortgage'] = False
        profile['num_late_payments_last_12m'] = np.random.choice([0, 1], p=[0.85, 0.15]) # Less history for late payments
        
        # Alternative data indicators (examples)
        profile['utility_payment_rating'] = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], p=[0.3, 0.4, 0.2, 0.1])
        profile['rent_payment_ontime_pct'] = round(np.random.uniform(0.7, 1.0), 2)
        profile['mobile_plan_type'] = np.random.choice(['Prepaid', 'Postpaid'])
        profile['years_with_current_employer'] = np.random.randint(0,5) if profile['employment_status'] in ['Employed', 'Self-Employed'] else 0
        profile['loan_purpose'] = fake.random_element(elements=('Personal Loan', 'Education', 'Small Business', 'Emergency', 'Other'))
        profile['requested_loan_amount'] = round(np.random.uniform(500, 15000), -2)
        # Simulate default probability (simplified, higher base for thin file)
        default_prob = 0.10 + 0.3 * (1- (profile.get('credit_score', 400) - 300)/350) if profile.get('credit_score') else 0.2
        if profile['utility_payment_rating'] == 'Poor': default_prob += 0.1
        if profile['rent_payment_ontime_pct'] < 0.85: default_prob += 0.1
        profile['defaulted'] = np.random.binomial(1, min(max(default_prob,0.02),0.98))
        return profile

    def _generate_credit_invisible_profile(self) -> dict:
        """Generates a profile for a credit-invisible individual."""
        profile = self._generate_base_profile(user_id_prefix='INVIS_')
        profile['profile_type'] = 'Credit-Invisible'
        profile['credit_score'] = None
        profile['months_credit_history'] = 0
        profile['num_credit_accounts'] = 0
        profile['credit_utilization_ratio'] = None
        profile['debt_to_income_ratio'] = None # Often rely on cash
        profile['has_mortgage'] = False
        profile['num_late_payments_last_12m'] = 0

        # Stronger emphasis on alternative data
        profile['digital_footprint_score'] = np.random.randint(1, 10) # 1-10 scale
        profile['social_media_longevity_years'] = np.random.randint(0, 10)
        profile['avg_monthly_utility_bill'] = round(np.random.uniform(50, 500), 2)
        profile['utility_payment_consistency'] = np.random.choice(['Consistent', 'Mostly Consistent', 'Inconsistent'], p=[0.5,0.3,0.2])
        profile['residential_stability_years'] = np.random.randint(0, 20)
        profile['income_source_verified'] = np.random.choice([True, False], p=[0.6, 0.4])
        profile['loan_purpose'] = fake.random_element(elements=('First Loan', 'Micro-Enterprise', 'Skill Development', 'Appliance Purchase', 'Other'))
        profile['requested_loan_amount'] = round(np.random.uniform(200, 5000), -2)
        # Simulate default probability (simplified, highest base for invisible)
        default_prob = 0.15 
        if profile['utility_payment_consistency'] == 'Inconsistent': default_prob += 0.15
        if not profile['income_source_verified']: default_prob += 0.1
        if profile['residential_stability_years'] < 1: default_prob += 0.05
        profile['defaulted'] = np.random.binomial(1, min(max(default_prob,0.05),0.99))
        return profile

    def generate_dataset(self, num_traditional=NUM_TRADITIONAL_PROFILES, 
                         num_thin_file=NUM_THIN_FILE_PROFILES, 
                         num_credit_invisible=NUM_CREDIT_INVISIBLE_PROFILES) -> pd.DataFrame:
        """Generates the full synthetic dataset."""
        logger.info(f"Generating dataset: {num_traditional} traditional, {num_thin_file} thin-file, {num_credit_invisible} credit-invisible.")
        
        profiles = []
        for _ in range(num_traditional):
            profiles.append(self._generate_traditional_financial_profile())
        for _ in range(num_thin_file):
            profiles.append(self._generate_thin_file_profile())
        for _ in range(num_credit_invisible):
            profiles.append(self._generate_credit_invisible_profile())
        
        df = pd.DataFrame(profiles)
        logger.info(f"Generated {len(df)} profiles in total.")
        return df

    def save_dataset(self, df: pd.DataFrame, filename: str = "synthetic_lending_data.csv") -> None:
        """Saves the generated dataset to a CSV file."""
        if not SYNTHETIC_DATA_DIR.exists():
            SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = SYNTHETIC_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")

if __name__ == '__main__':
    # This is an example of how to run the generator
    logger.info("Starting synthetic data generation process...")
    generator = SyntheticDataGenerator()
    
    # Generate a smaller dataset for quick testing
    # synthetic_df = generator.generate_dataset(num_traditional=100, num_thin_file=80, num_credit_invisible=50)
    synthetic_df = generator.generate_dataset() # Use default numbers from config
    
    print("\nSample of generated data:")
    print(synthetic_df.head())
    print("\nDataframe Info:")
    synthetic_df.info()
    print("\nProfile type distribution:")
    print(synthetic_df['profile_type'].value_counts())
    print("\nDefault rate by profile type:")
    print(synthetic_df.groupby('profile_type')['defaulted'].mean())
    
    generator.save_dataset(synthetic_df)
    logger.info("Synthetic data generation process finished.")

    # Further analysis could be done here, e.g.
    # print("\nBasic statistics:")
    # print(synthetic_df.describe(include='all'))