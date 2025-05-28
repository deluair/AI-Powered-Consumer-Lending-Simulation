import pandas as pd
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

class AlternativeFeatureExtractor:
    def __init__(self):
        """Initialize the extractor. May load NLP models or other resources here."""
        logger.info("AlternativeFeatureExtractor initialized.")
        # Example: self.nlp_model = spacy.load("en_core_web_sm")

    def extract_digital_footprint_features(self, df_digital: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features from digital footprint data.
        Example features: social media activity, online transaction patterns, etc.
        This is highly dependent on the raw digital footprint data available.
        """
        logger.info(f"Extracting digital footprint features from DataFrame with shape {df_digital.shape}")
        if df_digital.empty:
            logger.warning("Digital footprint DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        # Placeholder: Assume df_digital has 'user_id' and some raw text or categorical data
        # Example: 'social_media_posts', 'online_shopping_categories'
        features = df_digital[['user_id']].copy()

        # Example: Length of social media presence (if available)
        if 'social_media_longevity_years' in df_digital.columns:
            features['alt_social_longevity'] = df_digital['social_media_longevity_years']
        
        # Example: Consistency of digital identity (if available)
        if 'digital_identity_verification_patterns' in df_digital.columns:
            # This would require more complex logic based on the pattern data
            features['alt_identity_consistency_score'] = np.random.randint(1, 5, size=len(df_digital))

        logger.info(f"Generated digital footprint features. Shape: {features.shape}")
        return features.set_index('user_id')

    def extract_utility_payment_features(self, df_utility: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features from utility payment history.
        Example features: payment consistency, average bill amount, types of utilities.
        """
        logger.info(f"Extracting utility payment features from DataFrame with shape {df_utility.shape}")
        if df_utility.empty:
            logger.warning("Utility payment DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()
        
        features = df_utility[['user_id']].copy()

        if 'utility_payment_rating' in df_utility.columns:
             # Simple mapping for an example rating
            rating_map = {'Excellent': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Very Poor': 1}
            features['alt_utility_rating_numeric'] = df_utility['utility_payment_rating'].map(rating_map).fillna(0)

        if 'avg_monthly_utility_bill' in df_utility.columns:
            features['alt_avg_utility_bill'] = df_utility['avg_monthly_utility_bill']
        
        if 'utility_payment_consistency' in df_utility.columns:
            consistency_map = {'Consistent':3, 'Mostly Consistent':2, 'Inconsistent':1}
            features['alt_utility_consistency_score'] = df_utility['utility_payment_consistency'].map(consistency_map).fillna(0)

        logger.info(f"Generated utility payment features. Shape: {features.shape}")
        return features.set_index('user_id')

    def extract_geospatial_mobility_features(self, df_geo: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features from geospatial and mobility data.
        Example features: residential stability, commute patterns.
        """
        logger.info(f"Extracting geospatial/mobility features from DataFrame with shape {df_geo.shape}")
        if df_geo.empty:
            logger.warning("Geospatial DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        features = df_geo[['user_id']].copy()

        if 'residential_stability_years' in df_geo.columns:
            features['alt_residential_stability'] = df_geo['residential_stability_years']
        
        # Example: Commute distance (if available)
        # if 'commute_distance_km' in df_geo.columns:
        #     features['alt_commute_distance'] = df_geo['commute_distance_km']

        logger.info(f"Generated geospatial features. Shape: {features.shape}")
        return features.set_index('user_id')

    def combine_alternative_features(self, base_df: pd.DataFrame, list_of_alt_feature_dfs: list) -> pd.DataFrame:
        """
        Combines multiple alternative feature DataFrames with a base DataFrame (e.g., main applicant data).
        All DataFrames must be indexed by 'user_id'.
        """
        logger.info(f"Combining {len(list_of_alt_feature_dfs)} alternative feature DataFrames with base DataFrame.")
        
        df_combined = base_df.copy()
        if not df_combined.index.name == 'user_id':
             if 'user_id' in df_combined.columns:
                 df_combined = df_combined.set_index('user_id')
             else:
                 logger.error("Base DataFrame must have 'user_id' as index or column.")
                 raise ValueError("Base DataFrame must have 'user_id' as index or column.")

        for i, df_alt in enumerate(list_of_alt_feature_dfs):
            if df_alt.empty:
                logger.warning(f"Alternative feature DataFrame at index {i} is empty. Skipping.")
                continue
            if not df_alt.index.name == 'user_id':
                logger.error(f"Alternative feature DataFrame at index {i} must have 'user_id' as index.")
                # Optionally, try to set index if 'user_id' column exists
                # if 'user_id' in df_alt.columns:
                #    df_alt = df_alt.set_index('user_id')
                # else:
                raise ValueError(f"Alternative feature DataFrame at index {i} must have 'user_id' as index.")
            
            df_combined = df_combined.join(df_alt, how='left', rsuffix=f'_alt_{i}')
        
        logger.info(f"Combined DataFrame shape: {df_combined.shape}")
        return df_combined

if __name__ == '__main__':
    # Create dummy data for testing
    num_users = 5
    user_ids = [f'USER_{i:03d}' for i in range(num_users)]

    base_data = pd.DataFrame({'user_id': user_ids, 'age': np.random.randint(20, 60, num_users)})
    
    digital_data = pd.DataFrame({
        'user_id': user_ids,
        'social_media_longevity_years': np.random.randint(0, 10, num_users),
        'digital_identity_verification_patterns': ['pattern_A']*num_users # Simplified
    })

    utility_data = pd.DataFrame({
        'user_id': user_ids,
        'utility_payment_rating': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], num_users),
        'avg_monthly_utility_bill': np.random.uniform(50, 300, num_users),
        'utility_payment_consistency': np.random.choice(['Consistent', 'Inconsistent'], num_users)
    })

    geo_data = pd.DataFrame({
        'user_id': user_ids,
        'residential_stability_years': np.random.randint(1, 15, num_users)
    })

    extractor = AlternativeFeatureExtractor()

    feat_digital = extractor.extract_digital_footprint_features(digital_data)
    feat_utility = extractor.extract_utility_payment_features(utility_data)
    feat_geo = extractor.extract_geospatial_mobility_features(geo_data)

    print("\nDigital Features:")
    print(feat_digital.head())
    print("\nUtility Features:")
    print(feat_utility.head())
    print("\nGeo Features:")
    print(feat_geo.head())

    combined_features = extractor.combine_alternative_features(
        base_df=base_data.set_index('user_id'), 
        list_of_alt_feature_dfs=[feat_digital, feat_utility, feat_geo]
    )

    print("\nCombined Features:")
    print(combined_features.head())
    print(f"\nCombined features columns: {combined_features.columns.tolist()}")