import sys
print(sys.executable)

import pandas as pd
import plotly.express as px

from data_handler import df_active_adult

# Append the directory containing your package to sys.path.
sys.path.append('/Users/nirshlomo/Dropbox/frontlife/fraud/euphoria')
# Now import the package
from euphoria.components.graph import GraphComponent
from euphoria import Report

#TODO: get main_adults_flat_ud.py to work
main_adults = pd.read_csv('alert_analysis/data/df_main_active_adult.csv')

#Graph 1
#df_box_plot = px.data.tips()
box_plot = px.box(main_adults, x="date_time_prescribe", y="hosp_days", color="Age_cat", template='plotly')
box_plot.update_traces(quartilemethod="exclusive")

#Graph 2
#df_bar_animation = px.data.gapminder()
bar_animation = px.bar(main_adults, x="Age_cat", y="NumMedAmount",
                       color="Age_cat",animation_frame="DayEN_cat",
                       animation_group="ShiftType_cat",
                       range_y=[0, 100], template='plotly'
                       )


example_report: Report = Report(
    title='Test Report',
    researcher='Nir Shlomo',
)

# Yoy can add markdowns as a text
markdown_text = 'This is an example to **simple** report generated using Euphoria!'
# just add it to your report
example_report.add_object(markdown_text)

bullets = [
    'Step 1: pip install Euphoria',
    'Step 2: Initiate report object',
    'Step 3: add any object you want and Euphoria will create an HTML report'
]

example_report.add_object(bullets)

from euphoria.components.tabs import TabComponent
from euphoria.components.tables import TableComponent

plot_dict = {
    "Box graph": box_plot,
    "More tabs": {"Animation": bar_animation, "Text": ['nested tabs', 'are possible']}, }

example_report.add_object(plot_dict)

example_report.add_object(main_adults)

example_report.render()



#Graph 1
df_box_plot = px.data.tips()
box_plot = px.box(df_box_plot, x="day", y="total_bill", color="smoker", template='plotly')
box_plot.update_traces(quartilemethod="exclusive")

#Graph 2
df_bar_animation = px.data.gapminder()
bar_animation = px.bar(df_bar_animation, x="continent", y="pop", color="continent",animation_frame="year", animation_group="country", range_y=[0, 4000000000], template='plotly')

example_report: Report = Report(
    title='Model Training Report',
    researcher='Nir Shlomo',
)

# Yoy can add markdowns as a text
markdown_text = 'This is an example to **model training** report generated using Euphoria!'
# just add it to your report
example_report.add_object(markdown_text)

# In addition, you can add bullets with simple list of strings
bullets = [
    'Step 1: pip install Euphoria',
    'Step 2: Initiate report object',
    'Step 3: add any object you want and Euphoria will create an HTML report'
]

example_report.add_object(bullets)

from euphoria.components.tabs import TabComponent
from euphoria.components.tables import TableComponent

plot_dict = {
    "Box graph": box_plot,
    "More tabs": {"Animation": bar_animation, "Text": ['nested tabs', 'are possible']}, }

example_report.add_object(plot_dict)

# example_df = pd.read_csv("example_data.csv")
# example_report.add_object(example_df)

example_report.render()



def process_below_exceed_dose(data):
    """
    Creates a new column 'Dose' in the DataFrame based on the following conditions:

    For rows where 'Module_Alert_Rn' is one of:
        - "DRC - Frequency 1"
        - "DRC - Single Dose 1"
        - "DRC - Max Daily Dose 1"

    It checks the 'Alert_Message' column:
        - If the message contains "exceeds", 'Dose' is set to "exceeds".
        - Else if the message contains "below", 'Dose' is set to "below".
        - Otherwise, 'Dose' is set to None.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'Module_Alert_Rn' and 'Alert_Message' columns.

    Returns:
        pd.DataFrame: The modified DataFrame with a new column 'Dose'.
        :param data:
    """
    # Ensure 'Alert_Message' column is of string type for .str.contains to work properly.
    data['Alert_Message'] = data['Alert_Message'].astype(str)

    # Define the modules for which the logic should apply
    modules_to_check = ["DRC - Frequency 1", "DRC - Single Dose 1", "DRC - Max Daily Dose 1"]

    # Build condition masks using np.select.
    conditions = [
        (data['Module_Alert_Rn'].isin(modules_to_check)) & (data['Alert_Message'].str.contains("exceeds", na=False)),
        (data['Module_Alert_Rn'].isin(modules_to_check)) & (data['Alert_Message'].str.contains("below", na=False))
    ]
    choices = ["exceeds", "below"]

    # Create the new column 'Dose'
    data['Dose'] = np.select(conditions, choices, default=None)

    return data



test_df = process_below_exceed_dose(df_active_adult)

test_df_test = test_df.loc[test_df['Dose'] == "below"][['Module_Alert_Rn', 'Alert_Message', 'Dose']]

for col in df_active_adult.columns:
    print(f"Column: {col}")
    print(test_df_test[col].value_counts())
    print("\n")