import sys
import pandas as pd
import plotly.express as px
# Append the directory containing your package to sys.path.
sys.path.append('/Users/nirshlomo/Dropbox/frontlife/fraud/euphoria')
# Now import the package
from euphoria.components.graph import GraphComponent
from euphoria import Report

#TODO: get main_adults_flat_ud.py to work
main_adults = pd.read_csv('alert_analysis/data/main_data_2022/df_main_active_adult_py_version.csv')

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