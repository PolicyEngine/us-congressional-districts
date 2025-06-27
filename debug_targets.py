import pandas as pd
from us_congressional_districts.utils import get_data_directory
from us_congressional_districts.calibrate import get_dataset
from policyengine_us import Microsimulation

# Compare simulation capacity vs targets
dataset = get_dataset("cps_2023", 2023)
sim = Microsimulation(dataset=dataset)
sim.default_calculation_period = 2023

# Total employment income in simulation (scaled to population)
emp_income = sim.calculate("employment_income").values
total_sim_emp_income = emp_income.sum() * (
    150_000_000 / len(emp_income)
)  # Scale to US population

# Total employment income targets
soi_targets = pd.read_csv(
    get_data_directory() / "input" / "soi" / "agi_district.csv"
)
emp_targets = soi_targets[
    soi_targets["VARIABLE"] == "employment_income_amount"
]
total_target_emp_income = emp_targets["VALUE"].sum()

print("SCALE COMPARISON:")
print(
    f"Total simulation employment income (scaled): ${total_sim_emp_income:,.0f}"
)
print(f"Total target employment income: ${total_target_emp_income:,.0f}")
print(
    f"Ratio (target/simulation): {total_target_emp_income/total_sim_emp_income:.2f}"
)

# Check single district
al01_targets = emp_targets[emp_targets["GEO_NAME"] == "AL-01"]
al01_total_target = al01_targets["VALUE"].sum()

print(f"\nAL-01 DISTRICT:")
print(f"AL-01 total employment income target: ${al01_total_target:,.0f}")
print(f"This is {al01_total_target/1e9:.1f} billion dollars for one district!")

# Check what variables we have
print(f"\nVARIABLE CHECK:")
print(f'SOI variables in targets: {soi_targets["VARIABLE"].unique()}')

# Compare to AGI targets
agi_targets = soi_targets[
    soi_targets["VARIABLE"] == "adjusted_gross_income_amount"
]
total_agi_targets = agi_targets["VALUE"].sum()
print(f"Total AGI targets: ${total_agi_targets:,.0f}")
print(
    f"Employment income is {total_target_emp_income/total_agi_targets:.2f}x larger than AGI!"
)
