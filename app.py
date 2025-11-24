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
sns.set_style("whitegrid")

# CSS für grauen Hintergrund
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; margin-bottom: 25px; color: #333;'>Visualization & Analysis of Diurnal Fluctuations</h2>", unsafe_allow_html=True)

# ───────────────────────────────────────────
# 1) LITERATURE/DEFAULT VALUES & CONSTANTS
# ───────────────────────────────────────────
default_params = {
    ("Glucose", "Male"):    {"t0": 8.5, "A": 15, "MU": 12, "M": 100},
    ("Glucose", "Female"):  {"t0": 9.0, "A": 13, "MU": 11, "M":  95},
    ("Cortisol", "Male"):   {"t0": 7.0, "A": 5, "MU": 10, "M": 15}, # Adjusted for ug/dl scale example
    ("Cortisol", "Female"): {"t0": 7.5, "A": 4, "MU": 10, "M": 14},
    ("Other", "Male"):      {"t0": 4.0, "A": 10, "MU": 15, "M": 100},
    ("Other", "Female"):    {"t0": 4.5, "A": 10, "MU": 15, "M":  95},
}
GLUCOSE_CONVERSION_FACTOR = 18.016

# ───────────────────────────────────────────
# 2) HELPER FUNCTIONS (GLOBAL SCOPE)
# ───────────────────────────────────────────
def circadian(t, M, A, t0):
    return M + A * np.cos(2 * np.pi * (t - t0) / 24)

def format_time_string(decimal_hour):
    hours = int(decimal_hour)
    minutes = int(round((decimal_hour - hours) * 60))
    if minutes == 60: hours += 1; minutes = 0
    return f"{hours % 24:02d}:{minutes:02d}"

def chronomap_delta(A, M, t0, steps=100):
    t_vals = np.linspace(0, 24, steps)
    T1, T2 = np.meshgrid(t_vals, t_vals)
    Y1 = circadian(T1, M, A, t0)
    Y2 = circadian(T2, M, A, t0)
    return T1, T2, np.abs(Y1 - Y2)

@st.cache_data
def get_template_csv():
    """Generates a sample CSV matching the user's screenshot."""
    data = {
        "ANALYT": ["Cholesterin", "Glucose", "Cortisol", "Glucose", "Cortisol"],
        "VALUE": [167.0, 95.0, 14.5, 110.0, 8.2],
        "DIM": ["mg/dl", "mg/dl", "ug/dl", "mg/dl", "ug/dl"],
        "TIME": ["27.11.2013 16:06", "27.11.2013 08:30", "27.11.2013 20:00", "27.11.2013 14:00", "28.11.2013 02:00"],
        "SEX": ["M", "F", "M", "M", "F"],
        "AGE": [47, 32, 55, 45, 29]
    }
    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')

@st.cache_data
def load_and_process_data(_file, file_identifier):
    try:
        df = pd.read_csv(_file, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.upper() # Normalize to uppercase
        
        # Flexible Mapping based on Screenshot
        # Expected: ANALYT, VALUE, DIM, TIME, SEX, AGE
        COLUMN_MAP = {
            'ANALYT': 'analyte',
            'VALUE': 'value',
            'DIM': 'unit',
            'TIME': 'timestamp',
            'SEX': 'gender',
            'AGE': 'age'
        }
        
        # Rename available columns
        df = df.rename(columns=COLUMN_MAP)
        
        # Check requirements (Time and Value are minimum)
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            st.error(f"Error in '{file_identifier}': Missing crucial columns (TIME or VALUE). Found: {df.columns.tolist()}")
            return None
            
        # 1. Handle Value
        if df['value'].dtype == object:
            df['value'] = df['value'].astype(str).str.replace(',', '.', regex=False)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # 2. Handle Time
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df.dropna(subset=['timestamp', 'value'], inplace=True)
        df['hour_int'] = df['timestamp'].dt.hour
        
        # 3. Handle Missing Gender -> Group all
        if 'gender' not in df.columns:
            df['gender'] = 'All'
        else:
            df['gender'] = df['gender'].astype(str).str.upper().map({'M': 'Male', 'F': 'Female'}).fillna('All')

        # 4. Handle Missing Age -> Group all
        if 'age' not in df.columns:
            df['age_group'] = 'All Ages'
        else:
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 120], labels=["< 30 years", "30-50 years", "> 50 years"], right=False)
            df['age_group'] = df['age_group'].astype(str).replace('nan', 'All Ages')

        # 5. Handle Missing Analyte -> Default
        if 'analyte' not in df.columns:
            df['analyte'] = 'Unknown'
            
        df['source_file'] = file_identifier
        return df
    except Exception as e:
        st.error(f"Error processing '{file_identifier}': {e}")
        return None

def get_fitted_parameters(df_group, value_column='value'):
    if len(df_group) < 5: return np.nan, np.nan, np.nan
    medians = df_group.groupby('hour_int')[value_column].median()
    if len(medians) < 3: return np.nan, np.nan, np.nan
    x_data, y_data = medians.index.values, medians.values
    M_guess, A_guess = np.mean(y_data), (np.max(y_data) - np.min(y_data)) / 2
    t0_guess = x_data[np.argmax(y_data)]
    try:
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
        
        # Determine Unit
        if analyte == "Glucose":
            unit = st.radio("Unit", ["mg/dL", "mmol/L"], horizontal=True)
        elif analyte == "Cortisol":
            unit = "ug/dL"
            st.caption(f"Unit: {unit}")
        else:
            unit = "mg/dL"
        
        t1_time = c2.time_input("Time t₁", value=datetime.time(8, 0), key="t1_sim")
        t1_hour  = t1_time.hour + t1_time.minute / 60
        
        # Get defaults
        p = default_params.get((analyte, gender), default_params[("Other", "Male")])
        t0_lit, A_lit, MU_perc_lit, M_literature = p["t0"], p["A"], p["MU"], p["M"]
        
        st.markdown("**1. Adjust Model to Measured Value**")
        personalize_mode = st.checkbox("Adjust Mean (M) to t₁ value", value=True)
        M = M_literature
        
        conv_factor = 1.0
        if analyte == "Glucose" and unit == "mmol/L":
            conv_factor = GLUCOSE_CONVERSION_FACTOR

        if personalize_mode:
            val_default = M_literature / conv_factor
            step_val = 0.1 if unit == 'mmol/L' else 1.0
            
            y_measured_t1_display = st.number_input(f"Value at t₁ ({format_time_string(t1_hour)})", value=val_default, step=step_val, format="%.1f")
            
            y_measured_internal = y_measured_t1_display * conv_factor
            M = y_measured_internal - A_lit * np.cos(2 * np.pi * (t1_hour - t0_lit) / 24)
            st.info(f"Adjusted Mean (M): **{M:.2f} (Internal Scale)**")
            
        st.markdown("**2. Manual Adjustments**")
        editor_mode = st.checkbox("Enable Editor")
        A, t0, MU_perc = A_lit, t0_lit, MU_perc_lit
        
        if editor_mode:
            A = st.slider("Amplitude A", 0.5, 100.0, float(A_lit), 0.5)
            M = st.slider("Mean M", 0.0, 500.0, float(M), 1.0, disabled=personalize_mode)
            t0 = st.slider("Acrophase t₀ (h)", 0.0, 24.0, t0_lit, 0.1)
            MU_perc = st.slider("Uncertainty MU %", 1.0, 50.0, float(MU_perc_lit), 0.5)
        
        mu_abs = M * MU_perc / 100

    with right:
        # Arrays
        t_arr = np.linspace(0, 24, 500)
        y_arr = circadian(t_arr, M, A, t0)
        
        # Display Conversion
        y_disp = y_arr / conv_factor
        mu_disp = mu_abs / conv_factor
        
        # --- PLOT 1: Daily Profile ---
        fig_sin, ax_sin = plt.subplots(figsize=(10, 3.5))
        ax_sin.set_title(f"Simulated Profile: {analyte}", fontsize=12)
        
        # Main Curve
        ax_sin.plot(t_arr, y_disp, color="cornflowerblue", label="Profile")
        ax_sin.fill_between(t_arr, y_disp - mu_disp, y_disp + mu_disp, color="lightblue", alpha=0.3)
        
        # t1 Line (Black Solid)
        ax_sin.axvline(t1_hour, color="black", ls="-", lw=2, label=f"t₁ = {format_time_string(t1_hour)}")
        if personalize_mode:
            y_t1_disp = circadian(t1_hour, M, A, t0) / conv_factor
            ax_sin.plot(t1_hour, y_t1_disp, 'o', color='black', markersize=6)

        ax_sin.set_xlim(0, 24)
        ax_sin.set_ylabel(f"Concentration ({unit})")
        ax_sin.set_xlabel("Time (h)")
        ax_sin.legend(loc="upper right", fontsize='small')
        st.pyplot(fig_sin)
        plt.close(fig_sin)
        
        # --- PLOT 2 & 3: Chronomap & Clock ---
        c1, c2 = st.columns(2, gap="medium")
        y_t1 = circadian(t1_hour, M, A, t0)
        
        with c1:
            st.markdown("##### Chronomap")
            T1, T2, delta_values = chronomap_delta(A, M, t0)
            delta_disp = delta_values / conv_factor
            
            # Square figure for alignment
            fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
            pcm = ax_cm.pcolormesh(T2, T1, delta_disp, cmap="coolwarm", shading='gouraud')
            
            # t1 (Black Solid)
            ax_cm.axhline(t1_hour, color='black', ls='-', lw=2.5, label=f"t₁")
            
            # Slider for t2
            delta_h = st.slider("Diff t₂ ↔ t₁ (h)", 0.0, 24.0, 6.0, 0.5)
            t2_hour = (t1_hour + delta_h) % 24
            
            # t2 (Orange Dashed)
            ax_cm.axvline(t2_hour, color='orange', ls='--', lw=2.5, label=f"t₂")
            
            # Intercept
            ax_cm.plot(t2_hour, t1_hour, 'o', markerfacecolor='white', markeredgecolor='black', markersize=8)
            
            # Contour
            ax_cm.contour(T2, T1, delta_values, levels=[mu_abs], colors='gray', linestyles='dotted', linewidths=1)
            
            ax_cm.set_xlabel("Timepoint t₂")
            ax_cm.set_ylabel("Timepoint t₁")
            ax_cm.set_aspect('equal')
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            
        with c2:
            st.markdown("##### 24h Clock")
            y_t2 = circadian(t2_hour, M, A, t0)
            diff = abs(y_t1 - y_t2)
            
            # Status Box
            if diff <= mu_abs: 
                st.success(f"**Comparable** (Δ ≤ Limit)")
            else: 
                st.error(f"**Not Comparable** (Δ > Limit)")

            # Polar Plot
            fig_clk, ax_clk = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(5, 5))
            ax_clk.set_theta_offset(np.pi/2)
            ax_clk.set_theta_direction(-1)
            ax_clk.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
            ax_clk.set_xticklabels([f"{h*2}" for h in range(12)])
            ax_clk.set_yticks([]) 
            
            # Normalization
            mn, mx = M - A*1.2, M + A*1.2
            rng = mx - mn
            norm = lambda v: 0.2 + 0.8 * ((v - mn)/rng)
            
            r1, r2 = norm(y_t1), norm(y_t2)
            th1, th2 = (t1_hour/24)*2*np.pi, (t2_hour/24)*2*np.pi
            
            # t1 (Black Solid)
            ax_clk.plot([th1, th1], [0, r1], color='black', lw=3, ls='-', label=f"t₁ ({y_t1/conv_factor:.1f})")
            
            # t2 (Orange Dashed)
            ax_clk.plot([th2, th2], [0, r2], color='orange', lw=3, ls='--', label=f"t₂ ({y_t2/conv_factor:.1f})")
            
            ax_clk.set_ylim(0, 1)
            ax_clk.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), frameon=False)
            st.pyplot(fig_clk)
            plt.close(fig_clk)

# ==========================================
# TAB 2: DATA ANALYSIS
# ==========================================
with tab2:
    st.markdown("### Data Upload & Analysis")
    
    # Template Download
    st.download_button(
        label="📥 Download CSV Template",
        data=get_template_csv(),
        file_name="circadian_template.csv",
        mime="text/csv",
        help="Use this structure: ANALYT, VALUE, DIM, TIME, SEX, AGE"
    )
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        f1 = st.file_uploader("Upload Control Group (File 1)", type=["csv"], key="f1")
    with col_u2:
        f2 = st.file_uploader("Upload Test Group (File 2)", type=["csv"], key="f2")
        
    df1 = load_and_process_data(f1, "File 1") if f1 else None
    df2 = load_and_process_data(f2, "File 2") if f2 else None
    
    valid_dfs = [d for d in [df1, df2] if d is not None]
    
    if valid_dfs:
        df_all = pd.concat(valid_dfs, ignore_index=True)
        st.success(f"Loaded {len(df_all)} records.")
        
        # --- 1. FILTERING (Crucial Step: Analyte) ---
        st.markdown("---")
        st.subheader("1. Filter Data")
        
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        # Analyte Filter (Mandatory)
        avail_analytes = sorted(list(df_all['analyte'].unique()))
        sel_analyte = f_col1.selectbox("Analyte", avail_analytes)
        
        # Apply Analyte Filter first
        df_sub = df_all[df_all['analyte'] == sel_analyte].copy()
        
        # Determine Unit from Data
        if 'unit' in df_sub.columns:
            detected_units = df_sub['unit'].unique()
            current_unit = detected_units[0] if len(detected_units) > 0 else "Unknown"
        else:
            current_unit = "Unknown"
            
        st.caption(f"Showing data for **{sel_analyte}** (Unit: {current_unit})")

        # Other Filters
        avail_genders = ["All"] + sorted([g for g in df_sub['gender'].unique() if g != 'All'])
        avail_ages = ["All"] + sorted([a for a in df_sub['age_group'].unique() if a != 'All Ages'])
        avail_src = ["Combined"] + sorted(list(df_sub['source_file'].unique()))
        
        sel_gender = f_col2.selectbox("Gender", avail_genders)
        sel_age = f_col3.selectbox("Age Group", avail_ages)
        sel_source = f_col4.selectbox("Data Source", avail_src)
        
        # Apply remaining filters
        if sel_gender != "All": df_sub = df_sub[df_sub['gender'] == sel_gender]
        if sel_age != "All": df_sub = df_sub[df_sub['age_group'] == sel_age]
        
        df_plot = df_sub if sel_source == "Combined" else df_sub[df_sub['source_file'] == sel_source]
        
        # --- 2. VISUALIZATION ---
        st.subheader("2. Visualization")
        
        if not df_plot.empty:
            fig_box, ax_box = plt.subplots(figsize=(12, 5))
            colors = {'File 1': 'cornflowerblue', 'File 2': 'orange', 'Combined': 'mediumseagreen'}
            curr_color = colors.get(sel_source, 'gray')
            
            hours = np.arange(24)
            data_by_hour = [df_plot[df_plot['hour_int'] == h]['value'].values for h in hours]
            
            # Boxplot
            bp = ax_box.boxplot(data_by_hour, positions=hours, patch_artist=True, 
                                boxprops=dict(facecolor=curr_color, alpha=0.6),
                                medianprops=dict(color='black'))
            
            # Count labels
            for h, data in enumerate(data_by_hour):
                if len(data) > 0:
                    ax_box.text(h, min(data)*0.95, f"n={len(data)}", ha='center', fontsize=7, color='#555')
            
            # Median Line
            medians = df_plot.groupby('hour_int')['value'].median().reindex(hours)
            ax_box.plot(hours, medians, 'o-', color='darkblue', lw=1.5, label='Median')
            
            ax_box.set_xlim(-0.5, 23.5)
            ax_box.set_xlabel("Time (h)")
            ax_box.set_ylabel(f"{sel_analyte} ({current_unit})")
            ax_box.set_title(f"Distribution: {sel_analyte} - {sel_source}")
            st.pyplot(fig_box)
            plt.close(fig_box)
            
            # --- 3. MODELING ---
            st.subheader("3. Model Parameters")
            
            # Grid Plot of Subgroups (if any)
            groups = df_sub.groupby(['gender', 'age_group'])
            if len(groups) > 0:
                res_list = []
                st.markdown(f"**Fitted Curves by Subgroup ({sel_source if sel_source != 'Combined' else 'All Data'})**")
                
                # Check how many plots we need
                n_groups = len(groups)
                cols = 3
                rows = (n_groups // cols) + 1
                
                fig_grid, axes = plt.subplots(rows, cols, figsize=(12, 3*rows), sharex=True)
                axes = axes.flatten()
                
                for i, ((g, a), sub_data) in enumerate(groups):
                    if sel_source != "Combined":
                         sub_data = sub_data[sub_data['source_file'] == sel_source]
                         
                    ax = axes[i]
                    if not sub_data.empty:
                        # Plot Data points
                        medians = sub_data.groupby('hour_int')['value'].median()
                        ax.plot(medians.index, medians.values, '.', color='gray', alpha=0.5)
                        
                        # Fit
                        M_fit, A_fit, t0_fit = get_fitted_parameters(sub_data)
                        if not np.isnan(M_fit):
                            t_f = np.linspace(0, 24, 100)
                            y_f = circadian(t_f, M_fit, A_fit, t0_fit)
                            ax.plot(t_f, y_f, 'r-', label='Fit')
                            res_list.append({'Group': f"{g} | {a}", 'M': M_fit, 'A': A_fit, 't0': t0_fit%24})
                        
                        ax.set_title(f"{g}\n{a}", fontsize=9)
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, "No Data", ha='center')
                        
                for k in range(i+1, len(axes)): axes[k].axis('off') # Hide empty plots
                plt.tight_layout()
                st.pyplot(fig_grid)
                plt.close(fig_grid)
                
                if res_list:
                    st.dataframe(pd.DataFrame(res_list).set_index('Group').style.format("{:.2f}"))
                    
        else:
            st.warning("No data found for the selected filters.")
            
    else:
        st.info("Please upload a CSV file (use template above).")