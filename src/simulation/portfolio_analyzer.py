import pandas as pd
import numpy as np
from typing import Dict, Any, List

from ..utils.logger import get_logger

logger = get_logger(__name__)

class PortfolioAnalyzer:
    def __init__(self, simulation_results: pd.DataFrame, 
                 loan_amount_col: str = 'loan_amount',
                 interest_rate_col: str = 'interest_rate', # Annual interest rate (e.g., 0.05 for 5%)
                 decision_col: str = 'decision', # 'approved', 'rejected'
                 actual_outcome_col: str = 'actual_default', # 1 for default, 0 for paid
                 loan_term_months_col: str = 'loan_term_months'): # Loan term in months
        """
        Initializes the PortfolioAnalyzer with simulation results.

        :param simulation_results: DataFrame containing simulation outcomes.
                                   Expected columns include loan amounts, interest rates,
                                   loan decisions, actual default outcomes, and loan terms.
        :param loan_amount_col: Name of the column for loan amounts.
        :param interest_rate_col: Name of the column for annual interest rates.
        :param decision_col: Name of the column for loan approval decisions.
        :param actual_outcome_col: Name of the column for actual default status (1 or 0).
        :param loan_term_months_col: Name of the column for loan term in months.
        """
        if not isinstance(simulation_results, pd.DataFrame):
            raise ValueError("simulation_results must be a pandas DataFrame.")
        
        self.results = simulation_results.copy()
        self.loan_amount_col = loan_amount_col
        self.interest_rate_col = interest_rate_col
        self.decision_col = decision_col
        self.actual_outcome_col = actual_outcome_col
        self.loan_term_months_col = loan_term_months_col

        self._validate_columns()
        logger.info(f"PortfolioAnalyzer initialized with {len(self.results)} records.")

    def _validate_columns(self):
        """Validates that all necessary columns exist in the results DataFrame."""
        required_cols = [
            self.loan_amount_col, self.interest_rate_col, 
            self.decision_col, self.actual_outcome_col, self.loan_term_months_col
        ]
        missing_cols = [col for col in required_cols if col not in self.results.columns]
        if missing_cols:
            msg = f"Missing required columns in simulation_results: {missing_cols}"
            logger.error(msg)
            raise ValueError(msg)
        
        # Ensure 'approved' loans have necessary financial data
        approved_loans = self.results[self.results[self.decision_col] == 'approved']
        if approved_loans[self.loan_amount_col].isnull().any():
            logger.warning(f"Some approved loans have missing '{self.loan_amount_col}'. These will be handled as 0 or excluded.")
        if approved_loans[self.interest_rate_col].isnull().any():
            logger.warning(f"Some approved loans have missing '{self.interest_rate_col}'. These will be handled as 0 or excluded.")
        if approved_loans[self.loan_term_months_col].isnull().any():
            logger.warning(f"Some approved loans have missing '{self.loan_term_months_col}'. These will be handled as 0 or excluded.")

    def get_approved_loans(self) -> pd.DataFrame:
        """Returns a DataFrame of only approved loans."""
        return self.results[self.results[self.decision_col] == 'approved'].copy()

    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """
        Calculates overall portfolio metrics.
        """
        metrics = {}
        total_applications = len(self.results)
        approved_loans_df = self.get_approved_loans()
        num_approved_loans = len(approved_loans_df)

        metrics['total_applications'] = total_applications
        metrics['num_approved_loans'] = num_approved_loans
        metrics['approval_rate'] = (num_approved_loans / total_applications) if total_applications > 0 else 0

        if num_approved_loans > 0:
            metrics['total_loan_value_approved'] = approved_loans_df[self.loan_amount_col].sum()
            metrics['average_loan_amount_approved'] = approved_loans_df[self.loan_amount_col].mean()
            metrics['average_interest_rate_approved'] = approved_loans_df[self.interest_rate_col].mean() # Annual rate
            metrics['average_loan_term_months_approved'] = approved_loans_df[self.loan_term_months_col].mean()
        else:
            metrics['total_loan_value_approved'] = 0
            metrics['average_loan_amount_approved'] = 0
            metrics['average_interest_rate_approved'] = 0
            metrics['average_loan_term_months_approved'] = 0
        
        logger.info(f"Calculated overall metrics: {metrics}")
        return metrics

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculates risk-related metrics for the approved loan portfolio.
        """
        metrics = {}
        approved_loans_df = self.get_approved_loans()
        num_approved_loans = len(approved_loans_df)

        if num_approved_loans == 0:
            logger.info("No approved loans to calculate risk metrics.")
            return {
                'overall_default_rate': 0,
                'total_defaulted_value': 0,
                'num_defaulted_loans': 0,
                'loss_given_default_avg_estimate': 0, # Placeholder, LGD can be complex
                'expected_loss_estimate': 0 # Placeholder
            }

        defaulted_loans_df = approved_loans_df[approved_loans_df[self.actual_outcome_col] == 1]
        num_defaulted_loans = len(defaulted_loans_df)

        metrics['num_defaulted_loans'] = num_defaulted_loans
        metrics['overall_default_rate'] = (num_defaulted_loans / num_approved_loans) if num_approved_loans > 0 else 0
        metrics['total_defaulted_value'] = defaulted_loans_df[self.loan_amount_col].sum()
        
        # Loss Given Default (LGD) - simplified assumption (e.g., 100% loss of principal for defaulted loans)
        # In reality, LGD can vary and recovery amounts should be considered.
        # For this example, we assume LGD is 1 (100% of exposure at default).
        loss_given_default_estimate = 1.0 
        metrics['loss_given_default_avg_estimate'] = loss_given_default_estimate 
        
        # Expected Loss (EL) = Probability of Default (PD) * Exposure at Default (EAD) * Loss Given Default (LGD)
        # Here, PD is the overall_default_rate, EAD is total_loan_value_approved (simplified)
        # This is a portfolio-level EL estimate. Loan-level EL would be more precise.
        ead_total = approved_loans_df[self.loan_amount_col].sum()
        metrics['expected_loss_estimate'] = metrics['overall_default_rate'] * ead_total * loss_given_default_estimate
        
        logger.info(f"Calculated risk metrics: {metrics}")
        return metrics

    def _calculate_loan_profitability(self, row: pd.Series) -> float:
        """
        Simplified loan profitability calculation for a single loan.
        Assumes simple interest and full repayment if not defaulted.
        If defaulted, assumes loss of principal (or a fixed LGD could be applied).
        """
        if row[self.decision_col] != 'approved':
            return 0

        loan_amount = row[self.loan_amount_col]
        annual_interest_rate = row[self.interest_rate_col]
        term_months = row[self.loan_term_months_col]
        
        if pd.isna(loan_amount) or pd.isna(annual_interest_rate) or pd.isna(term_months) or term_months == 0:
            return 0 # Cannot calculate profitability

        term_years = term_months / 12.0

        if row[self.actual_outcome_col] == 1: # Defaulted
            # Simplified: loss of entire principal. Could be (principal * LGD_rate)
            return -loan_amount 
        else: # Paid
            # Simplified: total simple interest earned over the term
            total_interest_earned = loan_amount * annual_interest_rate * term_years
            return total_interest_earned

    def calculate_profitability_metrics(self) -> Dict[str, Any]:
        """
        Calculates profitability metrics for the approved loan portfolio.
        This is a simplified model.
        """
        metrics = {}
        approved_loans_df = self.get_approved_loans()
        num_approved_loans = len(approved_loans_df)

        if num_approved_loans == 0:
            logger.info("No approved loans to calculate profitability metrics.")
            return {
                'total_estimated_interest_earned': 0,
                'total_estimated_losses_from_defaults': 0,
                'net_estimated_profit': 0,
                'return_on_assets_approved_estimate': 0 # (Net Profit / Total Approved Loan Value)
            }

        # Calculate profitability for each loan
        approved_loans_df['estimated_profit'] = approved_loans_df.apply(self._calculate_loan_profitability, axis=1)

        total_interest_from_performing = approved_loans_df[approved_loans_df[self.actual_outcome_col] == 0]['estimated_profit'].sum()
        total_losses_from_defaults = approved_loans_df[approved_loans_df[self.actual_outcome_col] == 1]['estimated_profit'].sum() # Will be negative

        metrics['total_estimated_interest_earned_on_performing'] = total_interest_from_performing
        metrics['total_estimated_losses_from_defaults'] = abs(total_losses_from_defaults) # Make it positive for reporting loss amount
        metrics['net_estimated_profit'] = approved_loans_df['estimated_profit'].sum()
        
        total_approved_value = approved_loans_df[self.loan_amount_col].sum()
        if total_approved_value > 0:
            metrics['return_on_assets_approved_estimate'] = metrics['net_estimated_profit'] / total_approved_value
        else:
            metrics['return_on_assets_approved_estimate'] = 0
            
        logger.info(f"Calculated profitability metrics: {metrics}")
        return metrics

    def analyze_by_segment(self, segment_col: str) -> Dict[Any, Dict[str, Any]]:
        """
        Performs overall, risk, and profitability analysis for each segment in a given column.
        :param segment_col: The column name to segment the portfolio by (e.g., 'credit_score_band', 'income_group').
        :return: A dictionary where keys are segment values and values are dicts of metrics for that segment.
        """
        if segment_col not in self.results.columns:
            msg = f"Segment column '{segment_col}' not found in simulation_results."
            logger.error(msg)
            raise ValueError(msg)

        segmented_analysis = {}
        unique_segments = self.results[segment_col].unique()

        for segment_value in unique_segments:
            segment_data = self.results[self.results[segment_col] == segment_value]
            if segment_data.empty:
                logger.info(f"No data for segment: {segment_col} = {segment_value}. Skipping.")
                continue
            
            logger.info(f"Analyzing segment: {segment_col} = {segment_value}")
            segment_analyzer = PortfolioAnalyzer(
                simulation_results=segment_data,
                loan_amount_col=self.loan_amount_col,
                interest_rate_col=self.interest_rate_col,
                decision_col=self.decision_col,
                actual_outcome_col=self.actual_outcome_col,
                loan_term_months_col=self.loan_term_months_col
            )
            
            segment_metrics = {
                'overall': segment_analyzer.calculate_overall_metrics(),
                'risk': segment_analyzer.calculate_risk_metrics(),
                'profitability': segment_analyzer.calculate_profitability_metrics()
            }
            segmented_analysis[segment_value] = segment_metrics
            logger.info(f"Finished analysis for segment: {segment_col} = {segment_value}")
        
        return segmented_analysis

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive summary report including overall, risk, and profitability metrics.
        """
        logger.info("Generating summary report for the entire portfolio...")
        summary = {
            'overall_portfolio_metrics': self.calculate_overall_metrics(),
            'portfolio_risk_metrics': self.calculate_risk_metrics(),
            'portfolio_profitability_metrics': self.calculate_profitability_metrics()
        }
        logger.info("Summary report generated.")
        return summary


if __name__ == '__main__':
    logger.info("Running PortfolioAnalyzer example...")

    # Sample data for simulation results
    data = {
        'applicant_id': range(1, 101),
        'loan_amount': np.random.uniform(5000, 50000, 100).round(2),
        'interest_rate': np.random.uniform(0.03, 0.15, 100).round(4), # Annual rate
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], 100),
        'decision': np.random.choice(['approved', 'rejected'], 100, p=[0.7, 0.3]),
        'actual_default': np.zeros(100, dtype=int), # Initialize all as non-default
        'credit_score_band': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.2, 0.5, 0.3])
    }
    sample_results_df = pd.DataFrame(data)

    # For approved loans, assign some defaults
    approved_indices = sample_results_df[sample_results_df['decision'] == 'approved'].index
    num_to_default = int(len(approved_indices) * 0.1) # 10% default rate among approved
    if num_to_default > 0 and len(approved_indices) > 0:
        default_indices = np.random.choice(approved_indices, num_to_default, replace=False)
        sample_results_df.loc[default_indices, 'actual_default'] = 1
    
    # Ensure rejected loans don't have default status or financial impact in this context
    rejected_indices = sample_results_df[sample_results_df['decision'] == 'rejected'].index
    sample_results_df.loc[rejected_indices, 'actual_default'] = np.nan # Or 0, but NaN makes it clear they weren't at risk
    # For rejected loans, financial details might be present from application but won't contribute to portfolio performance
    # sample_results_df.loc[rejected_indices, ['loan_amount', 'interest_rate', 'loan_term_months']] = np.nan

    analyzer = PortfolioAnalyzer(simulation_results=sample_results_df)

    print("\n--- Overall Portfolio Summary ---")
    summary_report = analyzer.generate_summary_report()
    for key, value in summary_report.items():
        print(f"\n{key.replace('_', ' ').title()}:")
        for metric, val in value.items():
            print(f"  {metric.replace('_', ' ').title()}: {val:.4f}" if isinstance(val, float) else f"  {metric.replace('_', ' ').title()}: {val}")

    print("\n--- Analysis by Credit Score Band ---")
    segmented_report = analyzer.analyze_by_segment('credit_score_band')
    for segment, metrics_dict in segmented_report.items():
        print(f"\nSegment: Credit Score Band = {segment}")
        for category, cat_metrics in metrics_dict.items():
            print(f"  {category.title()} Metrics:")
            for metric, val in cat_metrics.items():
                print(f"    {metric.replace('_', ' ').title()}: {val:.4f}" if isinstance(val, float) else f"    {metric.replace('_', ' ').title()}: {val}")

    logger.info("PortfolioAnalyzer example finished.")