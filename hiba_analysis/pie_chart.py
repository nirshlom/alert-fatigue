import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Display all available columns
print(f"\n{'='*60}")
print("AVAILABLE COLUMNS IN DATASET:")
print(f"{'='*60}")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
print(f"{'='*60}")

# Check if module_alert_rn column exists
if 'module_alert_rn' in df.columns:
    print(f"\nColumn 'module_alert_rn' found!")
    print(f"Unique values in module_alert_rn: {df['module_alert_rn'].nunique()}")
    print(f"Value counts:\n{df['module_alert_rn'].value_counts()}")
    
    # Create pie chart for alert types distribution
    plt.figure(figsize=(12, 8))
    
    # Get value counts for the pie chart
    alert_counts = df['module_alert_rn'].value_counts()
    
    # Create pie chart
    plt.pie(alert_counts.values, 
            labels=alert_counts.index, 
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05] * len(alert_counts))  # Slight separation between slices
    
    plt.title('Distribution of Alert Types by Module Alert RN Categories', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Add legend
    plt.legend(alert_counts.index, 
              title="Alert Categories",
              loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.show()
    
    # Display summary statistics
    print(f"\nSummary of Alert Types Distribution:")
    print(f"Total alerts: {len(df)}")
    print(f"Number of unique alert categories: {len(alert_counts)}")
    print(f"\nTop 5 most frequent alert types:")
    print(alert_counts.head())
    
else:
    print("Column 'module_alert_rn' not found in the dataset.")

# Check if alert_rn_severity column exists
if 'alert_rn_severity' in df.columns:
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR ALERT RN SEVERITY")
    print(f"{'='*60}")
    print(f"Column 'alert_rn_severity' found!")
    print(f"Unique values in alert_rn_severity: {df['alert_rn_severity'].nunique()}")
    print(f"Value counts:\n{df['alert_rn_severity'].value_counts()}")
    
    # Create pie chart for alert severity distribution
    plt.figure(figsize=(12, 8))
    
    # Get value counts for the pie chart
    severity_counts = df['alert_rn_severity'].value_counts()
    
    # Create pie chart
    plt.pie(severity_counts.values, 
            labels=severity_counts.index, 
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05] * len(severity_counts))  # Slight separation between slices
    
    plt.title('Distribution of Alert Types by Alert RN Severity Categories', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Add legend
    plt.legend(severity_counts.index, 
              title="Severity Categories",
              loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.show()
    
    # Display summary statistics
    print(f"\nSummary of Alert Severity Distribution:")
    print(f"Total alerts: {len(df)}")
    print(f"Number of unique severity categories: {len(severity_counts)}")
    print(f"\nSeverity distribution:")
    print(severity_counts)
    
else:
    print("Column 'alert_rn_severity' not found in the dataset.")
    print("Available columns:")
    print(df.columns.tolist())

# Analysis for DDI-1 alerts and their severity distribution
print(f"\n{'='*60}")
print(f"ANALYSIS FOR DDI-1 ALERTS SEVERITY DISTRIBUTION")
print(f"{'='*60}")

# Check if both required columns exist
if 'module_alert_rn' in df.columns and 'ALERT_SEVERITY' in df.columns:
    # Filter for DDI-1 alerts
    ddi1_alerts = df[df['module_alert_rn'] == 'DDI-1']
    
    print(f"Total DDI-1 alerts found: {len(ddi1_alerts)}")
    print(f"Percentage of total alerts that are DDI-1: {(len(ddi1_alerts)/len(df)*100):.2f}%")
    
    if len(ddi1_alerts) > 0:
        # Get severity distribution for DDI-1 alerts
        ddi1_severity_counts = ddi1_alerts['ALERT_SEVERITY'].value_counts()
        
        print(f"\nALERT_SEVERITY distribution for DDI-1 alerts:")
        print(ddi1_severity_counts)
        
        # Create pie chart for DDI-1 severity distribution
        plt.figure(figsize=(12, 8))
        
        # Create pie chart
        plt.pie(ddi1_severity_counts.values, 
                labels=ddi1_severity_counts.index, 
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                explode=[0.05] * len(ddi1_severity_counts))  # Slight separation between slices
        
        plt.title('Distribution of ALERT_SEVERITY for DDI-1 Alerts', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add legend
        plt.legend(ddi1_severity_counts.index, 
                  title="Severity Levels",
                  loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.show()
        
        # Display summary statistics
        print(f"\nSummary of DDI-1 Alert Severity Distribution:")
        print(f"Total DDI-1 alerts: {len(ddi1_alerts)}")
        print(f"Number of unique severity levels: {len(ddi1_severity_counts)}")
        print(f"\nSeverity distribution for DDI-1 alerts:")
        print(ddi1_severity_counts)
        
    else:
        print("No DDI-1 alerts found in the dataset.")
        
elif 'module_alert_rn' not in df.columns:
    print("Column 'module_alert_rn' not found in the dataset.")
elif 'ALERT_SEVERITY' not in df.columns:
    print("Column 'ALERT_SEVERITY' not found in the dataset.")
    print("Available columns:")
    print(df.columns.tolist())


    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIG ====
UNIT_COL = "unit_category"   # replace with the exact column name in your data
COLS = [
    "dosing_frequency_direction",
    "dosing_single_dose_direction",
    "dosing_max_daily_dose_direction",
]
LABELS = {
    "dosing_frequency_direction": "FREQUENCY",
    "dosing_single_dose_direction": "SINGLE DOSE",
    "dosing_max_daily_dose_direction": "MAX DAILY DOSE",
}
VALID_DIRECTIONS = {"EXCEEDS", "BELOW"}

# ==== 1) TIDY THE DATA ====
def tidy_dosing_direction(df):
    use_cols = [c for c in COLS if c in df.columns]
    if UNIT_COL in df.columns:
        use_cols = [UNIT_COL] + use_cols
    t = df[use_cols].copy()

    t_long = t.melt(
        id_vars=[UNIT_COL] if UNIT_COL in t.columns else None,
        value_vars=[c for c in COLS if c in t.columns],
        var_name="error_type",
        value_name="direction"
    )

    # Normalize labels
    t_long["error_type"] = t_long["error_type"].map(LABELS).fillna(t_long["error_type"])
    t_long["direction"] = (
        t_long["direction"]
        .astype(str)
        .str.strip()
        .str.upper()
        .where(lambda s: s.isin(VALID_DIRECTIONS), "OTHER/NA")
    )
    return t_long

t_long = tidy_dosing_direction(df)

# ==== 2) OVERALL TABLE ====
def table_overall(t_long):
    counts = pd.crosstab(t_long["error_type"], t_long["direction"]).reindex(
        ["FREQUENCY", "SINGLE DOSE", "MAX DAILY DOSE"]
    )
    counts = counts.fillna(0).astype(int)
    totals = counts.sum(axis=1)
    pct = (counts.T / totals).T * 100
    combined = pd.concat({"Count": counts, "Percent": pct.round(1)}, axis=1)
    combined[("Count", "TOTAL")] = totals
    combined[("Percent", "TOTAL")] = 100.0
    return combined

overall_tbl = table_overall(t_long)
print("=== Overall ===")
display(overall_tbl)

# ==== 3) BY UNIT CATEGORY ====
def table_by_unit(t_long):
    if UNIT_COL not in t_long.columns:
        raise ValueError(f"Column '{UNIT_COL}' not found.")
    counts = pd.crosstab(
        [t_long[UNIT_COL], t_long["error_type"]],
        t_long["direction"]
    )
    counts = counts.fillna(0).astype(int)
    totals = counts.sum(axis=1).replace(0, np.nan)
    pct = (counts.T / totals).T * 100
    combined = pd.concat({"Count": counts, "Percent": pct.round(1)}, axis=1)
    return combined

by_unit_tbl = table_by_unit(t_long)
print("\n=== By unit_category ===")
display(by_unit_tbl)

# ==== 4) PLOTS ====
# Overall stacked bars
overall_counts = pd.crosstab(t_long["error_type"], t_long["direction"])[["EXCEEDS","BELOW"]].fillna(0)
overall_counts.plot(kind="bar", stacked=True, figsize=(7,4))
plt.title("Dosing-error directions by category (overall)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# By unit_category
for unit, g in t_long.groupby(UNIT_COL):
    c = pd.crosstab(g["error_type"], g["direction"])[["EXCEEDS","BELOW"]].fillna(0)
    if c.sum().sum() == 0:
        continue
    c.plot(kind="bar", stacked=True, figsize=(7,4))
    plt.title(f"Dosing-error directions by category – {unit}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np

# ==== CONFIG (matches your schema) ====
PATIENT_COL = "id1"
ORDER_ID_COL = "num_med_order"
ALERT_TYPE_COL = "alert_type"
ERROR_VALUE = "ERROR_ALERT"   # rows with this value indicate an error alert

# ==== 1) Orders per patient ====
orders_per_patient = (
    df.groupby(PATIENT_COL)[ORDER_ID_COL]
      .nunique()
      .rename("n_orders_per_patient")
      .reset_index()
)

# ==== 2) Did the patient have any medication errors? ====
errors_per_patient = (
    df.assign(is_error = df[ALERT_TYPE_COL].eq(ERROR_VALUE))
      .groupby(PATIENT_COL)["is_error"]
      .any()
      .rename("has_error")
      .reset_index()
)

# Merge to patient-level table
pt = orders_per_patient.merge(errors_per_patient, on=PATIENT_COL, how="left")
pt["has_error"] = pt["has_error"].fillna(False)

# ==== 3) Bin patients by number of orders ====
bins = [0, 1, 4, 9, 15, np.inf]
labels = ["<2", "2-4", "5-9", "10-15", ">15"]
pt["orders_bin"] = pd.cut(pt["n_orders_per_patient"], bins=bins, labels=labels, right=True)

# ==== 4) Build summary table ====
err_counts   = pt.groupby("orders_bin")["has_error"].sum().astype(int)
total_counts = pt.groupby("orders_bin")[PATIENT_COL].nunique().astype(int)
err_percent  = (err_counts / total_counts.replace(0, np.nan) * 100).round(1)

summary = pd.DataFrame({
    "Number of patients (all)": total_counts,
    "Number of patients with medication errors": err_counts,
    "Percent with errors (%)": err_percent
}).reindex(labels)

# n = total patients with errors
n_errors_total = int(pt["has_error"].sum())

print(
    "Number of medications orders per patient and number of patients with medication errors "
    f"(n = {n_errors_total})"
)
display(summary)

# ==== 5) If you need the compact table exactly as in your spec ====
final_table = summary[["Number of patients with medication errors"]].rename(
    columns={
        "Number of patients with medication errors":
        f"Number of patients with medication errors (n = {n_errors_total})"
    }
)
display(final_table)

#to do list - change the columns to the correct ones as in the data set 
import pandas as pd
import numpy as np

# ===================== CONFIGURE TO YOUR SCHEMA =====================
DF = df  # your source DataFrame

# Core columns
PATIENT_COL    = "id1"             # patient id (not strictly needed for this table)
ORDER_ID_COL   = "num_med_order"   # medication order id
PHYSICIAN_COL  = "physician_id"    # prescriber id (unique per physician)
ALERT_TYPE_COL = "alert_type"      # alert type string
SHIFT_COL      = "shift"           # expected values like 'Morning'/'Evening'/'Night'

# Optional time columns – used to count total shifts across the 10-day period
# Use either DATE_COL (date only) or a TIMESTAMP_COL; leave both None if not available.
DATE_COL       = "order_date"      # e.g., 'YYYY-MM-DD' (set to None if you don't have it)
TIMESTAMP_COL  = None              # e.g., 'order_ts' with full datetime (set to None if you don't have it)

# Error flags / override info (pick what you have)
ERROR_VALUE            = "ERROR_ALERT"  # rows with this alert_type are considered prescription errors
OVERRIDE_FLAG_COL      = "is_overridden"    # boolean True/False if present; else None
RESPONSE_COL           = None               # alternative: a text column with values like 'OVERRIDE', 'ACCEPT', etc.
RESPONSE_OVERRIDE_VALS = {"OVERRIDE", "OVERRIDDEN", "OVERRIDED"}  # normalized uppercase values if using RESPONSE_COL

# Reason text column (optional) + mapping to your 4 buckets
REASON_TEXT_COL = "ANSWER_TEXT_EN"  # set to None if not available
REASON_MAP = {
    "THERE IS INFORMATION IN THE CLINICAL LITERATURE": "Reason 1 (ANSWER_TEXT_EN): There Is Information In The Clinical Literature",
    "THE DRUG IS NOT MAPPED": "Reason 2: The Drug Is Not Mapped",
    "TECHNICAL ALERT": "Reason 3: Technical Alert",
    # everything else → Reason 4: Other
}

# Standardize shift names (optional)
SHIFT_ORDER = ["Morning", "Evening", "Night"]
SHIFT_NORMALIZE = {
    "MORNING": "Morning",
    "EVENING": "Evening",
    "NIGHT": "Night",
    "AM": "Morning",
    "PM": "Evening",
    # add any local variants if needed
}
# ===================== END CONFIG =====================

# --- Helper: safe uppercase strip
def _norm_str(x):
    return str(x).strip().upper() if pd.notna(x) else x

d = DF.copy()

# Normalize alert_type and shift for consistent matching
if ALERT_TYPE_COL in d.columns:
    d[ALERT_TYPE_COL] = d[ALERT_TYPE_COL].astype(str).str.strip()

if SHIFT_COL in d.columns:
    d[SHIFT_COL] = d[SHIFT_COL].map(lambda x: SHIFT_NORMALIZE.get(_norm_str(x), x))

# Identify error-alert rows
is_error = d[ALERT_TYPE_COL].eq(ERROR_VALUE)
d["__is_error__"] = is_error

# ---- Primary counts ----
# Number Of Medication Orders Prescribed (unique orders)
n_orders = d[ORDER_ID_COL].nunique()

# Number Of Participating Physicians (unique prescribers who issued at least one order)
if PHYSICIAN_COL in d.columns:
    n_physicians = d.loc[d[ORDER_ID_COL].notna(), PHYSICIAN_COL].nunique()
else:
    n_physicians = np.nan

# ---- Total Number Of Shifts (over the 10 days) ----
# Best: count unique (date, shift) pairs; fallback: estimate using timestamp → date; fallback: NaN
if DATE_COL and DATE_COL in d.columns:
    # Ensure date type
    d["_date_"] = pd.to_datetime(d[DATE_COL]).dt.date
    total_shifts = d[[ "_date_", SHIFT_COL ]].dropna().drop_duplicates().shape[0]
elif TIMESTAMP_COL and TIMESTAMP_COL in d.columns:
    d["_date_"] = pd.to_datetime(d[TIMESTAMP_COL]).dt.date
    total_shifts = d[[ "_date_", SHIFT_COL ]].dropna().drop_duplicates().shape[0]
else:
    total_shifts = np.nan  # not enough info to compute distinct shifts across days

# ---- Alerts per shift (ERROR_ALERT only) ----
alerts_shift_series = (
    d.loc[d["__is_error__"] & d[SHIFT_COL].notna(), SHIFT_COL]
      .value_counts()
      .reindex(SHIFT_ORDER, fill_value=0)
)

morning_alerts = int(alerts_shift_series.get("Morning", 0))
evening_alerts = int(alerts_shift_series.get("Evening", 0))
night_alerts   = int(alerts_shift_series.get("Night",   0))

# ---- Prescription Error Rate (ERROR_ALERT) ----
# Define as: (# unique orders that had ≥1 ERROR_ALERT) / (total unique orders)
orders_with_error = (
    d.loc[d["__is_error__"], ORDER_ID_COL].dropna().drop_duplicates().shape[0]
)
error_rate = (orders_with_error / n_orders * 100) if n_orders else np.nan

# ---- Override Rate ----
# Numerator: count of error alerts that were overridden
# Denominator: total error alerts with an override decision (or simply total error alerts if you don't track "decision recorded")
error_rows = d.loc[d["__is_error__"]].copy()

if OVERRIDE_FLAG_COL and OVERRIDE_FLAG_COL in d.columns:
    # boolean column (True/False)
    num_overridden = int(error_rows[OVERRIDE_FLAG_COL].fillna(False).sum())
    denom_overridables = int(error_rows[OVERRIDE_FLAG_COL].notna().sum())  # if all rows have flag, equals len(error_rows)
elif RESPONSE_COL and RESPONSE_COL in d.columns:
    rr = error_rows[RESPONSE_COL].astype(str).str.strip().str.upper()
    num_overridden = int(rr.isin(RESPONSE_OVERRIDE_VALS).sum())
    denom_overridables = int(rr.notna().sum())
else:
    # Fallback: cannot detect overrides → set NaNs
    num_overridden = np.nan
    denom_overridables = error_rows.shape[0] if error_rows.shape[0] > 0 else np.nan

override_pct = (num_overridden / denom_overridables * 100) if (denom_overridables and denom_overridables == denom_overridables) else np.nan

# ---- Reasons breakdown (from ANSWER_TEXT_EN), bucketed to 4 reasons ----
def map_reason(text):
    if pd.isna(text):
        return "Reason 4: Other"
    key = _norm_str(text)
    for k, label in REASON_MAP.items():
        if k in key:
            return label
    return "Reason 4: Other"

reasons_table = None
if REASON_TEXT_COL and REASON_TEXT_COL in d.columns:
    r = error_rows[REASON_TEXT_COL].map(map_reason).value_counts(dropna=False)
    # Ensure all 4 rows exist (even if 0)
    desired_rows = [
        "Reason 1 (ANSWER_TEXT_EN): There Is Information In The Clinical Literature",
        "Reason 2: The Drug Is Not Mapped",
        "Reason 3: Technical Alert",
        "Reason 4: Other",
    ]
    r = r.reindex(desired_rows, fill_value=0)
    reasons_total = int(r.sum())
    reasons_pct = (r / max(reasons_total, 1) * 100).round(2)
    reasons_table = pd.DataFrame({
        "count": r.astype(int),
        "percent": reasons_pct
    })
else:
    reasons_total = 0

# ---- Build the final presentation table ----
rows = [
    ("Number Of Medication Order Prescribed", f"{n_orders:,}"),
    ("Number Of Participating Physicians",    f"{int(n_physicians):,}" if pd.notna(n_physicians) else "N/A"),
    ("Total Number Of Shifts",                f"{int(total_shifts):,}" if pd.notna(total_shifts) else "N/A"),
    ("Morning Shifts ALERTS",                 f"{morning_alerts:,}"),
    ("Evening Shifts ALERTS",                 f"{evening_alerts:,}"),
    ("Night Shifts ALERTS",                   f"{night_alerts:,}"),
    ("Prescription Error Rate – ALERT TYPE (ERROR ALERT)",
     f"{error_rate:.2f}%"
     if pd.notna(error_rate) else "N/A"),
]

# Override rate row
if pd.notna(num_overridden) and pd.notna(denom_overridables) and denom_overridables > 0:
    rows.append(("Override Rate", f"{num_overridden:,} / {denom_overridables:,} ({override_pct:.2f}%)"))
else:
    rows.append(("Override Rate", "N/A"))

# Reason rows (if available)
if reasons_table is not None:
    rows.extend([
        ("Reason 1 (ANSWER_TEXT_EN): There Is Information In The Clinical Literature",
         f"{int(reasons_table.loc['Reason 1 (ANSWER_TEXT_EN): There Is Information In The Clinical Literature','count'])} "
         f"({reasons_table.loc['Reason 1 (ANSWER_TEXT_EN): There Is Information In The Clinical Literature','percent']:.2f}%)"),
        ("Reason 2: The Drug Is Not Mapped",
         f"{int(reasons_table.loc['Reason 2: The Drug Is Not Mapped','count'])} "
         f"({reasons_table.loc['Reason 2: The Drug Is Not Mapped','percent']:.2f}%)"),
        ("Reason 3: Technical Alert",
         f"{int(reasons_table.loc['Reason 3: Technical Alert','count'])} "
         f"({reasons_table.loc['Reason 3: Technical Alert','percent']:.2f}%)"),
        ("Reason 4: Other",
         f"{int(reasons_table.loc['Reason 4: Other','count'])} "
         f"({reasons_table.loc['Reason 4: Other','percent']:.2f}%)"),
    ])

final_table = pd.DataFrame(rows, columns=["Parameter", "Value"])

# Show
print("Physicians, prescription, and workload characteristics during the study period (10 days).")
display(final_table)

# Optional: export
# final_table.to_csv("physicians_prescription_workload_10days.csv", index=False)


