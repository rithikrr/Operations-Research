# #Food Wastage Optimization and Sensitivity Analysis

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Load the optimized data
file_path = "C:/Users/91770/Desktop/Prescriptive Analytics and Optimization/Project/Final/food_wastage_data.csv"
data = pd.read_csv(file_path)

# Add 'No Show Guests' and 'Actual Guests Present'
data['No Show Guests'] = (data['Number of Guests'] * 0.1).round().astype(int)
data['Actual Guests Present'] = data['Number of Guests'] - data['No Show Guests']
data['Cost per Unit'] = data['Pricing'].map({'Low': 200, 'Moderate': 400, 'High': 600})

# Optimization Model
cost_vector = data['Cost per Unit'].values
wastage_vector = data['Wastage Food Amount'].values
min_food_constraints = data['Actual Guests Present'] * 0.5  # Assuming 0.5 units per guest minimum
print(cost_vector)
print(wastage_vector)
print(min_food_constraints)

# Constraint matrices
A_ub = np.vstack([
    np.eye(len(cost_vector)),        # Budget constraints
    -np.eye(len(cost_vector))        # Minimum food constraints
])
print(A_ub.shape)

# Budget mapping and constraints
#pricing_map = {1: 200, 2: 400, 3: 600}
#data['Budget Per Guest'] = data['Cost per Unit'].map(pricing_map)
budget_limits = data['Cost per Unit'] * data['Number of Guests']
min_quantity_limits = -min_food_constraints.values

b_ub = np.hstack([budget_limits, min_quantity_limits])
c = wastage_vector

# Solve the optimization
result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

# Handle result
if result.success:
    optimal_solution = result.x
    optimal_value = result.fun

    optimization_results = pd.DataFrame({
        'Type of Food': data['Type of Food'],
        'Event Type': data['Event Type'],
        'Optimal Quantity': optimal_solution.round(2),
        'Wastage Per Unit': wastage_vector,
        'Cost Per Unit': cost_vector
    })

    optimization_results.to_csv("Optimal_Food_Preparation_Quantities.csv", index=False)
    print(f"Optimal Wastage Value: {optimal_value}")
    print("Optimal Food Preparation Quantities saved as 'Optimal_Food_Preparation_Quantities.csv'")
else:
    print("Optimization failed.")
    print("Solver message:", result.message)

# Sensitivity Analysis
budget_range = np.arange(300, 1001, 100)
guest_range = np.arange(0.2, 1.2, 0.2)
wastage_multiplier_range = np.arange(0.5, 1.6, 0.2)

sensitivity_results = []

for budget in budget_range:
    for guest_factor in guest_range:
        for wastage_factor in wastage_multiplier_range:
            data['Adjusted Guests'] = (data['Actual Guests Present'] * guest_factor).round().astype(int)
            data['Adjusted Wastage'] = (data['Wastage Food Amount'] * wastage_factor).round().astype(int)
            min_food_constraints = data['Adjusted Guests'] * 0.5

            A_budget = np.eye(len(data))
            A_min_quantity = -np.eye(len(data))
            A_ub = np.vstack([A_budget, A_min_quantity])
            b_ub = np.hstack([
                np.full(len(data), budget),
                -min_food_constraints.values
            ])

            c = data['Adjusted Wastage'].values

            # Validation
            if np.any(np.isnan(c)) or np.any(np.isnan(A_ub)) or np.any(np.isnan(b_ub)):
                print("NaNs in input; skipping scenario.")
                continue
            if np.any(np.isinf(c)) or np.any(np.isinf(A_ub)) or np.any(np.isinf(b_ub)):
                print("Infs in input; skipping scenario.")
                continue

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

            sensitivity_results.append({
                'Budget': budget,
                'Guest Factor': guest_factor,
                'Wastage Factor': wastage_factor,
                'Optimal Wastage': result.fun if result.success else np.nan,
                'Status': 'Optimal' if result.success else 'Infeasible'
            })
print(result.x)

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df.to_csv("Sensitivity_Analysis_Results.csv", index=False)
print("Sensitivity Analysis Results saved as 'Sensitivity_Analysis_Results.csv'")
