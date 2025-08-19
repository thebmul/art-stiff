"""
This module contains the logic for computing various arterial stiffness metrics
based on IVUS image data and pulse pressure waveforms.
"""

import numpy as np

class StiffnessCalculator:
    """
    A class to encapsulate methods for calculating different arterial stiffness
    indices. The methods are designed to be used with synchronized data
    from IVUS image processing and waveform analysis.
    """

# TODO: add the pressure gradient method in other methods where applicable? Or perhaps not
# TODO: look into units of measurement (double-check references)
# TODO: list references
# TODO: error handling for invalid inputs (e.g., negative areas or pressures)
# TODO: double-check formulas and methods in general is my Stiffness Index calculation correct?

    @staticmethod
    def calculate_pressure_gradient(
        systolic_pressure: float,
        diastolic_pressure: float
    ) -> float:
        """
        Calculates the pressure gradient (delta P) from the measurements of
        peak systolic and end diastolic pulse pressures.

        Formula: delta P = Ps - Pd

        Args:
            systolic_pressure (float): The peak systolic blood pressure (mmHg).
            diastolic_pressure (float): The end diastolic blood pressure (mmHg).

        Returns:
            float: The calculated pressure gradient (delta P) in mmHg. Returns 0 if
                    the diastolic pressure is greater than or equal to the systolic pressure.
        """

        if diastolic_pressure >= systolic_pressure:
            print("Warning: delta P is zero.")
            return 0.0
        
        return systolic_pressure - diastolic_pressure
    
    @staticmethod
    def calculate_vessel_diameter(
        vessel_area: float
    ) -> float:
        """
        Calculates the diameter (at peak systole or end diastole) from the
        cross-sectional lumen area.

        Formula: diameter = 2 * sqrt(area / pi)

        Args:
            vessel_area (float): The cross-sectional lumen area at peak systole (cm^2).

        Returns:
            float: The calculated vessel diameter. 
        """
        pi = np.pi
        
        return (2 * np.sqrt(vessel_area / pi))
    
    @staticmethod
    def calculate_distensibility(
        mean_systolic_area: float,
        mean_diastolic_area: float,
        mean_pressure_gradient: float
    ) -> float:
        """
        Calculates distensibility, a key indicator of arterial stiffness.

        Formula: (Change in mean Area * 100) / (mean Pressure Gradient * mean Diastolic Area)

        Args:
            systolic_area (float): The cross-sectional lumen area at peak systole (cm^2).
            diastolic_area (float): The cross-sectional lumen area at end diastole (cm^2).
            mean_pressure_gradient (float): The difference between peak systolic and end
                    diastolic blood pressure (mmHg).

        Returns:
            float: The calculated Distensibility. Returns 0 if the pressure
                    change is zero to prevent division by zero errors.
        """
        
        delta_area = mean_systolic_area - mean_diastolic_area
        
        if mean_pressure_gradient == 0.0:
            print("Distensibility cannot be calculated.")
            return 0.0

        return (delta_area * 100) / (mean_pressure_gradient * mean_diastolic_area)

    @staticmethod
    def calculate_compliance(
        mean_systolic_area: float,
        mean_diastolic_area: float,
        mean_pressure_gradient: float
    ) -> float:
        """
        Calculates Compliance, another key indicator of arterial stiffness. 

        Formula: (Change in mean Area) / (mean Pressure Gradient)

        Args:
            systolic_area (float): The cross-sectional lumen area at peak systole (cm^2).
            diastolic_area (float): The cross-sectional lumen area at end diastole (cm^2).
            mean_pressure_gradient (float): The difference between peak systolic and end
                    diastolic blood pressure (mmHg).

        Returns:
            float: The calculated Arterial Compliance. Returns 0 if the pressure
                    change is zero to prevent division by zero errors.
        """
        
        delta_area = mean_systolic_area - mean_diastolic_area

        if mean_pressure_gradient == 0.0:
            print("Compliance cannot be calculated.")
            return 0.0

        return delta_area / mean_pressure_gradient
        
    @staticmethod
    def calculate_stiffness_index(
        systolic_area: float,
        diastolic_area: float,
        mean_pressure_gradient: float
    ) -> float:
        """
        Calculates the Stiffness Index (β). This method implements the specific formula for 
        Stiffness Index. The formula is logarithmic and is yet another key measure of 
        arterial stiffness.

        Formula:
            β = ln(mean Pressure Gradient) / (Change in Diameter / mean Diastolic Diameter)

        Args:
            systolic_area (float): The cross-sectional lumen area at peak systole (cm^2).
            diastolic_area (float): The cross-sectional lumen area at end diastole (cm^2).
            mean_pressure_gradient (float): The difference between peak systolic and end
                    diastolic blood pressure (mmHg).

        Returns:
            float: The calculated Stiffness Index (β). Returns 0 if the change in mean 
                    Diameter is zero to prevent division by zero errors.
        """
        #pulse_pressure_ratio = mean_pressure_gradient
        systolic_diameter = StiffnessCalculator.calculate_vessel_diameter(systolic_area)
        diastolic_diameter = StiffnessCalculator.calculate_vessel_diameter(diastolic_area)

        area_strain = (systolic_diameter / diastolic_diameter) / diastolic_diameter
        
        # Handle potential division by zero and log errors
        if mean_pressure_gradient <= 0:
            print("Log Error: Stiffness Index cannot be calculated.")
            return 0.0
        
        if area_strain == 0:
            print("Warning: Change in mean Diameter (area strain) is zero. " \
                    "Stiffness Index cannot be calculated.")
            return 0.0
        
        return np.log(mean_pressure_gradient) / area_strain
    
    # TODO: Implement "Young's Elastic Modulus" metric calculation
    @staticmethod
    def calculate_elastic_modulus():
        """
        Placeholder for Elastic Modulus calculation.
        """
        pass

# Example usage 
if __name__ == "__main__":
    # Sample data
    sample_systolic_area = 1.05  # cm^2
    sample_diastolic_area = 0.98 # cm^2
    sample_systolic_pressure = 120 # mmHg
    sample_diastolic_pressure = 80 # mmHg

    # Create an instance of the calculator
    stiffness_analyzer = StiffnessCalculator()

    # Calculate and print the Distensibility Index
    distensibility = stiffness_analyzer.calculate_distensibility_index(
        sample_systolic_area,
        sample_diastolic_area,
        sample_systolic_pressure,
        sample_diastolic_pressure
    )
    print(f"Calculated Distensibility Index: {distensibility:.4f}")

    # Calculate and print the Beta Stiffness Index
    beta_index = stiffness_analyzer.calculate_beta_stiffness_index(
        sample_systolic_area,
        sample_diastolic_area,
        sample_systolic_pressure,
        sample_diastolic_pressure
    )
    print(f"Calculated Beta Stiffness Index: {beta_index:.4f}")
    
    # Calculate and print the Compliance
    compliance = stiffness_analyzer.calculate_compliance(
        sample_systolic_area,
        sample_diastolic_area,
        sample_systolic_pressure,
        sample_diastolic_pressure
    )
    print(f"Calculated Compliance: {compliance:.4f}")
