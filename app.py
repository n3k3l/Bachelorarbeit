import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.optimize import curve_fit

# ───────────────────────────────────────────
# BASIC SETTINGS & LAYOUT
# ───────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Circadian Rhythm Analysis")
sns.set_style("whitegrid")  # Apply clean style to plots

st.markdown("<h2 style='text-align:center; margin-bottom: 25px;'>Visualization & Analysis of Diurnal Fluctuations</h2>", unsafe_allow_html=True)

# ───────────────────────────────────────────
# 1) LITERATURE/DEFAULT VALUES & CONSTANTS
# ───────────────────────────────────────────
# Default parameters based on literature for different analytes and genders
default_params = {
    ("Glucose", "Male"):    {"t0": 8.5, "A": 15, "MU": 12, "M": 100},
    ("Glucose", "Female"):  {"t0": 9.0, "A": 13, "MU": 11, "M":  95},
    ("Cortisol", "Male"):   {"t0": 7.0, "A": 20, "MU": 10, "M": 180},
    ("Cortisol", "Female"): {"t0": 7.5, "A": 18, "MU": 10, "M": 170},
    ("Other", "Male"):      {"t0": 4.0, "A": 10, "MU": 15, "M": 100},
    ("Other", "Female"):    {"t0": 4.5, "A": 10, "MU": 15, "M":  95},
}
# Conversion factor for glucose: mg/dL <-> mmol/L
GLUCOSE_CONVERSION_FACTOR = 18.016

# ───────────────────────────────────────────
# 2) HELPER FUNCTIONS (GLOBAL SCOPE)
# ───────────────────────────────────────────
def circadian(t, M, A, t0):
    """Cosine function to model circadian rhythm: y = M + A * cos(2*pi*(t-t0)/24)."""
    return M + A * np.cos(2 * np.pi * (t - t0) / 24)

def format_time_string(decimal_hour):
    """Formats a decimal hour (e.g., 8.5) into a time string (e.g., '08:30')."""
    hours = int(decimal_hour)
    minutes = int(round((decimal_hour - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours % 24:02d}:{minutes:02d}"

def chronomap_delta(A, M, t0, steps=100):
    """Calculates the absolute difference between circadian values at all pairs of timepoints."""
    t_vals = np.linspace(0, 24, steps)
    T1, T2 = np.meshgrid(t_vals, t_vals)
    Y1 = circadian(T1, M, A, t0)
    Y2 = circadian(T2, M, A, t0)
    return T1, T2, np.abs(Y1 - Y2)

@st.cache_data
def load_and_process_data(_file, file_identifier):
    """
    Loads and preprocesses the uploaded CSV data.
    Expects columns roughly matching: Age, Gender, Analyse_date, Value.
    """
    try:
        # Read CSV with auto-separator detection
        df = pd.read_csv(_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        
        # Map flexible column names to standard internal names
        # 'Value' is assumed to be the analyte concentration
        COLUMN_MAP = {
            'Age': 'age', 
            'Gender': 'gender', 
            'Analyse_date': 'timestamp', 
            'Value': 'glucose_mmol_l'
        }
        
        # Check for missing columns
        required_cols = set(COLUMN_MAP.keys())
        found_cols = set(df.columns)
        if not required_cols.issubset(found_cols):
            st.error(f"Error in '{file_identifier}': Missing columns: {list(required_cols - found_cols)}")
            st.warning(f"Found columns: {list(found_cols)}")
            return None
            
        df = df.rename(columns=COLUMN_MAP)
        
        # Clean Data: Handle decimal commas if present (e.g. "5,5" -> "5.5")
        if df['glucose_mmol_l'].dtype == object:
            df['glucose_mmol_l'] = df['glucose_mmol_l'].astype(str).str.replace(',', '.', regex=False)
            
        # Convert to numeric, coercing errors to NaN
        df['glucose_mmol_l'] = pd.to_numeric(df['glucose_mmol_l'], errors='coerce')
        
        # Determine health status (simple threshold logic for demo)
        df['health_status'] = np.where(df['glucose_mmol_l'] >= 11.1, 'Sick', 'Healthy')
        
        # Remove invalid rows
        df.dropna(subset=['glucose_mmol_l', 'age'], inplace=True)
        
        df['age'] = df['age'].astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        
        # Extract hour and normalize gender
        df['hour_int'] = df['timestamp'].dt.hour
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.upper().map({'M': 'Male', 'F': 'Female'}).fillna('Other')
            
        # Create Age Groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 120], labels=["< 30 years", "30-50 years", "> 50 years"], right=False)
        df['source_file'] = file_identifier
        
        return df
    except Exception as e:
        st.error(f"Error processing '{file_identifier}': {e}")
        return None

def get_fitted_parameters(df_group, value_column='glucose_mmol_l'):
    """Fits the circadian model to a group of data and returns (M, A, t0)."""
    # Need enough data points to fit
    if len(df_group) < 10: 
        return np.nan, np.nan, np.nan
        
    # Group by hour to get median trajectory
    medians = df_group.groupby('hour_int')[value_column].median()
    if len(medians) < 3: 
        return np.nan, np.nan, np.nan
        
    x_data, y_data = medians.index.values, medians.values
    
    # Initial Guesses
    M_guess = np.mean(y_data)
    A_guess = (np.max(y_data) - np.min(y_data)) / 2
    # Guess peak time (t0) based on max value index
    t0_guess = x_data[np.argmax(y_data)]
    
    try:
        # Fit the cosine function
        popt, _ = curve_fit(circadian, x_data, y_data, p0=[M_guess, A_guess, t0_guess], maxfev=5000)
        return popt[0], popt[1], popt[2]
    except RuntimeError:
        return np.nan, np.nan, np.nan

# ───────────────────────────────────────────
# 3) STREAMLIT APPLICATION
# ───────────────────────────────────────────
tab1, tab2 = st.tabs(["Circadian Model (Simulation)", "Data Analysis & Parameter Estimation"])

# ==========================================
# TAB 1: SIMULATION
# ==========================================
with tab1:
    left, right = st.columns([1, 2], gap="large")
    
    with left:
        st.markdown("#### Input & Parameters")
        c1, c2 = st.columns(2)
        analyte = c1.selectbox("Analyte", ["Glucose", "Cortisol", "Other"], key="analyte_sim")
        gender = c1.selectbox("Gender", ["Male", "Female"], key="gender_sim")
        age = c2.slider("Age", 0, 100, 35, key="age_sim")
        t1_time = c2.time_input("Time t₁", value=datetime.time(8, 0), key="t1_sim")
        t1_hour  = t1_time.hour + t1_time.minute / 60
        
        # Conditional Unit selection
        unit = st.radio("Unit for Glucose", ["mg/dL", "mmol/L"], horizontal=True) if analyte == "Glucose" else "mg/dL"
        
        # Get default parameters
        p = default_params.get((analyte, gender), default_params[("Other", "Male")])
        t0_lit, A_lit, MU_perc_lit, M_literature = p["t0"], p["A"], p["MU"], p["M"]
        
        st.markdown("**1. Adjust Model to Measured Value**")
        personalize_mode = st.checkbox("Adjust Mean (M) to measured value at t₁", value=True)
        
        M = M_literature
        
        if personalize_mode:
            # Display default value converted to selected unit
            val_default = M_literature / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else float(M_literature)
            step_val = 0.1 if unit == 'mmol/L' else 1.0
            
            y_measured_t1_display = st.number_input(
                f"Measured value at t₁ ({format_time_string(t1_hour)}) in {unit}", 
                value=val_default, step=step_val, format="%.1f"
            )
            
            # Convert input back to mg/dL for internal calculation
            y_measured_t1_mgdl = y_measured_t1_display * GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_measured_t1_display
            
            # Recalculate Mesor (M) so the curve hits the point exactly
            M = y_measured_t1_mgdl - A_lit * np.cos(2 * np.pi * (t1_hour - t0_lit) / 24)
            st.info(f"Adjusted Mean (M): **{M:.2f} mg/dL**")
            
        st.markdown("**2. Manually Adjust Parameters (Editor)**")
        editor_mode = st.checkbox("Enable Editor Mode")
        
        # Initialize params
        A, t0, MU_perc = A_lit, t0_lit, MU_perc_lit
        
        if editor_mode:
            A = st.slider("Amplitude A (mg/dL)", 1.0, 50.0, float(A_lit), 0.5)
            # Allow M editing only if not calculating it automatically
            M = st.slider("Mean M (mg/dL)", 50.0, 250.0, float(M), 1.0, disabled=personalize_mode)
            t0 = st.slider("Acrophase t₀ (h)", 0.0, 24.0, t0_lit, 0.1)
            MU_perc = st.slider("Measurement Uncertainty MU %", 1.0, 50.0, float(MU_perc_lit), 0.5)
        
        mu_abs = M * MU_perc / 100

    with right:
        # --- Calculations for Plotting ---
        t_arr = np.linspace(0, 24, 500)
        y_arr_mgdl = circadian(t_arr, M, A, t0)
        
        # Convert arrays for display
        y_arr_display = y_arr_mgdl / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_arr_mgdl
        mu_abs_display = mu_abs / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else mu_abs
        
        # --- Main Diurnal Curve Plot ---
        fig_sin, ax_sin = plt.subplots(figsize=(10, 3))
        ax_sin.set_title(f"Simulated Diurnal Fluctuation for {analyte}", fontsize=12)
        ax_sin.plot(t_arr, y_arr_display, color="blue", label="Expected Daily Profile")
        ax_sin.axvline(t1_hour, color="red", ls="--", label=f"t₁ = {format_time_string(t1_hour)}")
        ax_sin.fill_between(t_arr, y_arr_display - mu_abs_display, y_arr_display + mu_abs_display, color="lightblue", alpha=0.5, label=f"Tolerance (±{MU_perc}%)")
        
        if personalize_mode:
            y_t1_display = circadian(t1_hour, M, A, t0) / (GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else 1.0)
            ax_sin.plot(t1_hour, y_t1_display, 'ro', markersize=6, label='Measured Value')
            
        ax_sin.set_xlabel("Time of Day (h)")
        ax_sin.set_ylabel(f"Concentration ({unit})")
        ax_sin.legend(fontsize='small')
        ax_sin.set_xlim(0, 24)
        st.pyplot(fig_sin)
        plt.close(fig_sin)
        
        # --- Sub-plots (Chronomap & Clock) ---
        c1, c2 = st.columns(2, gap="medium")
        y_t1 = circadian(t1_hour, M, A, t0)
        
        with c1:
            st.markdown("##### Chronomap: Comparability")
            # Calculate delta map
            T1, T2, delta_values = chronomap_delta(A, M, t0)
            
            # Convert delta values if necessary
            if unit == 'mmol/L':
                delta_values_display = delta_values / GLUCOSE_CONVERSION_FACTOR
            else:
                delta_values_display = delta_values

            fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
            pcm = ax_cm.pcolormesh(T2, T1, delta_values_display, cmap="coolwarm", shading='gouraud')
            fig_cm.colorbar(pcm, ax=ax_cm, label=f"Abs. Difference ({unit})")
            
            # Contour line for Measurement Uncertainty
            ax_cm.contour(T2, T1, delta_values, levels=[mu_abs], colors='black', linestyles='dashed')
            
            delta_h = st.slider("Time Difference t₂ ↔ t₁ (h)", 0.0, 24.0, 6.0, 0.25, key="delta_h_slider")
            t2_hour = (t1_hour + delta_h) % 24
            
            ax_cm.axhline(t1_hour, color='white', ls='-', lw=2, label=f"t₁")
            ax_cm.axvline(t2_hour, color='lime', ls='-', lw=2, label=f"t₂")
            ax_cm.plot(t2_hour, t1_hour, 'ko', markersize=8, mfc='white')
            
            ax_cm.set_xlabel("Timepoint t₂ (h)")
            ax_cm.set_ylabel("Timepoint t₁ (h)")
            ax_cm.set_xlim(0, 24); ax_cm.set_ylim(0, 24)
            ax_cm.set_aspect('equal')
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            
        with c2:
            st.markdown("##### Concentration on the 24h Clock")
            y_t2 = circadian(t2_hour, M, A, t0)
            
            # Comparison Check
            diff = abs(y_t1 - y_t2)
            conv = GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else 1.0
            
            if diff <= mu_abs: 
                st.success(f"**Comparable:** Δ = {diff/conv:.2f} (≤ {mu_abs/conv:.2f})")
            else: 
                st.error(f"**Not comparable:** Δ = {diff/conv:.2f} (> {mu_abs/conv:.2f})")
            
            # Polar Plot Logic
            min_val, max_val = M - A, M + A
            range_val = max_val - min_val
            # Normalize values 0.1 to 1.0 radius
            norm = lambda y: 0.1 + 0.9 * np.clip((y - min_val) / range_val, 0, 1) if range_val > 0 else 0.5
            
            r1, r2 = norm(y_t1), norm(y_t2)
            angle = lambda h: (h / 24.0) * 2 * np.pi
            theta1, theta2 = angle(t1_hour), angle(t2_hour)
            
            fig_clk, ax_clk = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(4.5, 4.5))
            ax_clk.set_theta_offset(np.pi/2)
            ax_clk.set_theta_direction(-1)
            ax_clk.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
            ax_clk.set_xticklabels([f"{h*2}" for h in range(12)])
            ax_clk.set_yticklabels([])
            ax_clk.set_rlim(0, 1)
            
            y_t1_disp = y_t1 / conv
            y_t2_disp = y_t2 / conv
            
            label_t1 = f"t₁: {format_time_string(t1_hour)} ({y_t1_disp:.1f})"
            label_t2 = f"t₂: {format_time_string(t2_hour)} ({y_t2_disp:.1f})"
            
            ax_clk.plot([theta1, theta1], [0, r1], color='red', lw=2.5, ls='--', label=label_t1)
            ax_clk.plot([theta2, theta2], [0, r2], color='black', lw=2.5, label=label_t2)
            
            ax_clk.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize='small')
            st.pyplot(fig_clk)
            plt.close(fig_clk)

# ==========================================
# TAB 2: DATA ANALYSIS
# ==========================================
with tab2:
    st.markdown("This section analyzes patient data to identify circadian patterns and enable comparisons.")
    
    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        uploaded_file_1 = st.file_uploader("Upload File 1 (e.g., Control Group)", type=["csv"], key="file1")
    with upload_col2:
        uploaded_file_2 = st.file_uploader("Upload File 2 (e.g., Test Group)", type=["csv"], key="file2")

    df1, df2 = None, None
    if uploaded_file_1:
        df1 = load_and_process_data(uploaded_file_1, "File 1")
    if uploaded_file_2:
        df2 = load_and_process_data(uploaded_file_2, "File 2")
    
    valid_dfs = [df for df in [df1, df2] if df is not None]

    if valid_dfs:
        df_combined = pd.concat(valid_dfs, ignore_index=True)
        
        st.success(f"Data loaded successfully. Total: {len(df_combined)} measurements.")
        st.markdown("---")
        st.markdown("### 1. Exploratory Data Analysis")
        
        # --- Filtering Controls ---
        c1, c2, c3 = st.columns(3)
        gender_options = ["All Genders"] + sorted(list(df_combined['gender'].unique()))
        # Safely get unique age groups that actually exist in data
        age_options = ["All Age Groups"] + list(df_combined['age_group'].dropna().unique())
        health_options = ["All (Healthy & Sick)"] + list(sorted(df_combined['health_status'].unique()))

        gender_filter = c1.selectbox("Gender", gender_options, key="gender_data")
        age_group_filter = c2.selectbox("Age Group", age_options, key="age_data")
        health_status_filter = c3.selectbox("Health Status", health_options, key="health_data")

        # --- Apply Filters ---
        plot_df_base = df_combined.copy()
        if gender_filter != "All Genders":
            plot_df_base = plot_df_base[plot_df_base['gender'] == gender_filter]
        if age_group_filter != "All Age Groups":
            plot_df_base = plot_df_base[plot_df_base['age_group'] == age_group_filter]
        if health_status_filter != "All (Healthy & Sick)":
            plot_df_base = plot_df_base[plot_df_base['health_status'] == health_status_filter]

        # --- Source Selection for Display ---
        source_files = sorted(list(df_combined['source_file'].unique()))
        if len(source_files) > 1:
            display_mode = st.radio("Select data to display:", ["Combined"] + source_files, horizontal=True, key="display_mode_radio")
        else:
            display_mode = source_files[0]

        if display_mode == "Combined":
            df_to_plot = plot_df_base
            plot_color, line_color = 'mediumseagreen', 'darkgreen'
        elif display_mode == "File 1":
            df_to_plot = plot_df_base[plot_df_base['source_file'] == 'File 1']
            plot_color, line_color = 'cornflowerblue', 'darkblue'
        else: # File 2
            df_to_plot = plot_df_base[plot_df_base['source_file'] == 'File 2']
            plot_color, line_color = 'sandybrown', 'darkred'
            
        title = f"Data for: {display_mode} (Filters: {gender_filter}, {age_group_filter}, {health_status_filter})"

        # --- Boxplot Visualization ---
        fig_box, ax_box = plt.subplots(figsize=(15, 7))
        ax_box.set_title(title)
        ax_box.set_xlabel("Time of Day (h)")
        ax_box.set_ylabel("Glucose (mmol/L)")
        ax_box.set_xlim(-0.5, 23.5)
        ax_box.set_xticks(np.arange(0, 24, 1))
        
        if not df_to_plot.empty:
            all_hours = range(24)
            # Prepare list of arrays for boxplot
            box_data = [df_to_plot[df_to_plot['hour_int'] == h]['glucose_mmol_l'].values for h in all_hours]
            
            # Calculate variable widths based on sample count
            sample_counts = [len(data) for data in box_data]
            max_count = max(sample_counts) if any(sample_counts) else 1
            widths = [0.2 + 0.6 * (count / max_count) if max_count > 0 else 0.2 for count in sample_counts]
            
            boxes = ax_box.boxplot(box_data, positions=list(all_hours), widths=widths, patch_artist=True, showfliers=False, 
                                   boxprops=dict(facecolor=plot_color, alpha=0.8), medianprops=dict(color=line_color, linewidth=2))
            
            # Add sample count (n) below boxes
            for hour, count, box in zip(all_hours, sample_counts, boxes['boxes']):
                if count > 0:
                    path = box.get_path()
                    # Safe check for vertices
                    if path.vertices.size > 0:
                        box_ymin = np.min(path.vertices[:,1])
                        ax_box.text(hour, box_ymin - 0.2, str(count), ha='center', va='top', fontsize=8, color='gray')
            
            # Add Median Line
            medians = df_to_plot.groupby('hour_int')['glucose_mmol_l'].median().reindex(all_hours)
            ax_box.plot(all_hours, medians, 'o-', color=line_color, label=f'Median ({display_mode})', zorder=3, ms=5)
            ax_box.legend(loc='upper left')
        else:
            ax_box.text(11.5, 5, "No data available for this selection.", ha='center', va='center', fontsize=12)
        
        st.pyplot(fig_box)
        plt.close(fig_box)

        st.markdown("---")
        st.markdown("### 2. Parameter Estimation")
        
        # --- Parameter Fitting Section ---
        analysis_options = sorted(list(df_combined['source_file'].unique()))
        if len(analysis_options) > 1: 
            analysis_options.append("Combined")
            
        source_to_analyze = st.selectbox("Select dataset for modeling:", analysis_options, key="analysis_source")

        if source_to_analyze == "Combined":
            df_for_fitting = df_combined
        else:
            df_for_fitting = df_combined[df_combined['source_file'] == source_to_analyze]
            
        results = []
        unique_genders = sorted(df_for_fitting['gender'].unique())
        unique_age_groups = df_for_fitting['age_group'].cat.categories
        
        if unique_genders and not unique_age_groups.empty:
            # Create grid plot
            fig_grid, axes = plt.subplots(len(unique_age_groups), len(unique_genders), 
                                          figsize=(12, 4 * len(unique_age_groups)), 
                                          sharey=True, sharex=True, squeeze=False)
            
            fig_grid.suptitle(f"Daily Profiles & Models for '{source_to_analyze}' (Values in mmol/L)", fontsize=16)

            for i, age_group in enumerate(unique_age_groups):
                for j, gender in enumerate(unique_genders):
                    ax = axes[i, j]
                    ax.set_title(f"{gender}, {age_group}", fontsize=10)
                    
                    # Labels only on outer edges
                    if i == len(unique_age_groups) - 1: 
                        ax.set_xlabel("Time of Day (h)")
                    if j == 0: 
                        ax.set_ylabel("Glucose (mmol/L)")
                        
                    sub_df = df_for_fitting[(df_for_fitting['gender'] == gender) & (df_for_fitting['age_group'] == age_group)]
                    
                    if not sub_df.empty:
                        medians = sub_df.groupby('hour_int')['glucose_mmol_l'].median()
                        if not medians.empty:
                            ax.plot(medians.index, medians.values, 'o-', ms=3, color='blue', alpha=0.6, label='Median')
                            
                            # Fit Model
                            M_fit, A_fit, t0_fit = get_fitted_parameters(sub_df)
                            
                            results.append({
                                "Gender": gender, 
                                "Age Group": age_group, 
                                "Mesor (M)": M_fit, 
                                "Amplitude (A)": A_fit, 
                                "Acrophase (t0)": t0_fit % 24 if not np.isnan(t0_fit) else np.nan
                            })
                            
                            if not np.isnan(M_fit):
                                t_fit = np.linspace(0, 24, 100)
                                y_fit = circadian(t_fit, M_fit, A_fit, t0_fit)
                                ax.plot(t_fit, y_fit, '--', color='red', lw=2, label='Model')
                    
                    ax.legend(fontsize='x-small')
                    ax.set_xlim(0, 24)

            fig_grid.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig_grid)
            plt.close(fig_grid)
            
            st.markdown("#### Estimated Parameters (in mmol/L)")
            if results:
                res_df = pd.DataFrame(results).dropna()
                if not res_df.empty:
                    st.dataframe(res_df.round(2), use_container_width=True)
                else:
                    st.warning("Not enough data to fit models for the subgroups.")
    else:
        st.info("Please upload at least one CSV file to start the analysis. Expected columns: Age, Gender, Analyse_date, Value.")