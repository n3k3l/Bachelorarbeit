import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.optimize import curve_fit
from io import BytesIO

# ───────────────────────────────────────────
# BASIC SETTINGS & CSS STYLING
# ───────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Circadian Analysis")

# OPTIMIZED CSS FOR GRAY BACKGROUND & HIGH CONTRAST TEXT
st.markdown("""
    <style>
    /* 1. Main Background color (Light Professional Gray) */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* 2. Force Text Color to Dark Gray/Black for contrast */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, label, div {
        color: #1f1f1f !important;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    
    /* 3. Style the Tabs to look like cards */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 10px 20px;
        color: #1f1f1f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-top: 3px solid #ff4b4b;
        font-weight: bold;
    }
    
    /* 4. Make Dataframes and Inputs pop on gray background */
    .stDataFrame, .stTable {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Set global plot style
sns.set_style("whitegrid")
# Ensure text in plots is dark
plt.rcParams['text.color'] = '#1f1f1f'
plt.rcParams['axes.labelcolor'] = '#1f1f1f'
plt.rcParams['xtick.color'] = '#1f1f1f'
plt.rcParams['ytick.color'] = '#1f1f1f'

st.markdown("<h2 style='text-align:center; margin-bottom: 25px;'>Visualization & Analysis of Diurnal Fluctuations</h2>", unsafe_allow_html=True)

# ───────────────────────────────────────────
# 1) CONSTANTS
# ───────────────────────────────────────────
default_params = {
    ("Glucose", "Male"):    {"t0": 8.5, "A": 15, "MU": 12, "M": 100},
    ("Glucose", "Female"):  {"t0": 9.0, "A": 13, "MU": 11, "M":  95},
    ("Cortisol", "Male"):   {"t0": 7.0, "A": 20, "MU": 10, "M": 180},
    ("Cortisol", "Female"): {"t0": 7.5, "A": 18, "MU": 10, "M": 170},
    ("Other", "Male"):      {"t0": 4.0, "A": 10, "MU": 15, "M": 100},
    ("Other", "Female"):    {"t0": 4.5, "A": 10, "MU": 15, "M":  95},
}
GLUCOSE_CONVERSION_FACTOR = 18.016

# ───────────────────────────────────────────
# 2) HELPER FUNCTIONS
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
def generate_template_csv():
    data = {
        'ANALYT': ['Cholesterin', 'Glucose', 'Cortisol', 'Glucose', 'Glucose'],
        'VALUE': [167.0, 95.0, 14.5, 5.2, 105.0],
        'DIM': ['mg/dl', 'mg/dl', 'ug/dl', 'mmol/l', 'mg/dl'],
        'TIME': ['27.11.2013 16:06', '27.11.2013 08:30', '27.11.2013 20:00', '28.11.2013 09:15', '28.11.2013 14:00'],
        'SEX': ['M', 'F', 'M', 'F', 'M'],
        'AGE': [47, 32, 55, 29, 60]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def load_and_process_data(_file, file_identifier):
    try:
        df = pd.read_csv(_file, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.upper()
        
        COLUMN_MAP = {'ANALYT': 'analyte', 'VALUE': 'value', 'DIM': 'unit', 'TIME': 'timestamp', 'SEX': 'gender', 'AGE': 'age'}
        
        required_cols = set(COLUMN_MAP.keys())
        found_cols = set(df.columns)
        
        if not required_cols.issubset(found_cols):
            fallback_map = {'GENDER': 'gender', 'AGE': 'age', 'VALUE': 'value', 'ANALYSE_DATE': 'timestamp'}
            renamed = False
            for k, v in fallback_map.items():
                if k in found_cols:
                    df.rename(columns={k: v}, inplace=True); renamed = True
            if not renamed:
                st.error(f"Error in '{file_identifier}': Missing columns. Expected: {list(required_cols)}")
                return None
        else:
            df = df.rename(columns=COLUMN_MAP)

        if df['value'].dtype == object:
            df['value'] = df['value'].astype(str).str.replace(',', '.', regex=False)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        if 'analyte' in df.columns:
            df['analyte'] = df['analyte'].astype(str).str.title()
        else:
            df['analyte'] = "Unknown"
            
        df.dropna(subset=['value', 'age'], inplace=True)
        df['age'] = df['age'].astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['hour_int'] = df['timestamp'].dt.hour
        
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.upper().map({'M': 'Male', 'F': 'Female'}).fillna('Other')
            
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 120], labels=["< 30 years", "30-50 years", "> 50 years"], right=False)
        df['source_file'] = file_identifier
        return df
    except Exception as e:
        st.error(f"Error processing '{file_identifier}': {e}"); return None

def get_fitted_parameters(df_group, value_column='value'):
    if len(df_group) < 5: return np.nan, np.nan, np.nan
    medians = df_group.groupby('hour_int')[value_column].median()
    if len(medians) < 3: return np.nan, np.nan, np.nan
    x, y = medians.index.values, medians.values
    M_guess, A_guess = np.mean(y), (np.max(y) - np.min(y)) / 2
    t0_guess = x[np.argmax(y)]
    try:
        popt, _ = curve_fit(circadian, x, y, p0=[M_guess, A_guess, t0_guess], maxfev=5000)
        return popt[0], popt[1], popt[2]
    except: return np.nan, np.nan, np.nan

# ───────────────────────────────────────────
# 3) STREAMLIT APP
# ───────────────────────────────────────────
tab1, tab2 = st.tabs(["Circadian Model (Simulation)", "Data Analysis & Parameter Estimation"])

# --- TAB 1: SIMULATION ---
with tab1:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("#### Input & Parameters")
        c1, c2 = st.columns(2)
        analyte = c1.selectbox("Analyte", ["Glucose", "Cortisol", "Other"], key="analyte_sim")
        gender = c1.selectbox("Gender", ["Male", "Female"], key="gender_sim")
        age = c2.slider("Age", 0, 100, 35, key="age_sim")
        t1_time = c2.time_input("Time t₁", value=datetime.time(8, 0), key="t1_sim")
        t1_hour = t1_time.hour + t1_time.minute / 60
        
        unit = st.radio("Unit for Glucose", ["mg/dL", "mmol/L"], horizontal=True) if analyte == "Glucose" else "mg/dL"
        
        p = default_params.get((analyte, gender), default_params[("Other", "Male")])
        t0_lit, A_lit, MU_perc_lit, M_literature = p["t0"], p["A"], p["MU"], p["M"]
        
        st.markdown("**1. Adjust Model**")
        personalize_mode = st.checkbox("Adjust Mean (M) to measured value at t₁", value=True)
        M = M_literature
        if personalize_mode:
            val_default = M_literature / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else float(M_literature)
            step_val = 0.1 if unit == 'mmol/L' else 1.0
            y_measured = st.number_input(f"Value at t₁ ({format_time_string(t1_hour)})", value=val_default, step=step_val, format="%.1f")
            y_measured_mgdl = y_measured * GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_measured
            M = y_measured_mgdl - A_lit * np.cos(2 * np.pi * (t1_hour - t0_lit) / 24)
            
        st.markdown("**2. Manual Adjustments**")
        if st.checkbox("Enable Editor Mode"):
            A = st.slider("Amplitude A", 1.0, 50.0, float(A_lit), 0.5)
            M = st.slider("Mean M", 0.0, 300.0, float(M), 1.0, disabled=personalize_mode)
            t0 = st.slider("Acrophase t₀ (h)", 0.0, 24.0, t0_lit, 0.1)
            MU_perc = st.slider("Uncertainty MU %", 1.0, 50.0, float(MU_perc_lit), 0.5)
        else:
            A, t0, MU_perc = A_lit, t0_lit, MU_perc_lit
            
        mu_abs = M * MU_perc / 100

    with right:
        t_arr = np.linspace(0, 24, 500)
        y_arr = circadian(t_arr, M, A, t0)
        y_disp = y_arr / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_arr
        mu_disp = mu_abs / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else mu_abs

        # Main Plot with White Background (Card Style)
        fig_sin, ax_sin = plt.subplots(figsize=(10, 3.5), facecolor='white')
        ax_sin.set_title(f"Simulated Diurnal Fluctuation for {analyte}", fontsize=12)
        ax_sin.plot(t_arr, y_disp, color="cornflowerblue", alpha=0.9, lw=2, label="Expected Profile")
        
        # t1: Black Solid
        ax_sin.axvline(t1_hour, color="black", ls="-", lw=2, label=f"t₁ = {format_time_string(t1_hour)}")
        
        ax_sin.fill_between(t_arr, y_disp - mu_disp, y_disp + mu_disp, color="gray", alpha=0.15, label=f"Tolerance (±{MU_perc}%)")
        if personalize_mode:
             ax_sin.plot(t1_hour, y_measured, 'o', color='black', markersize=6)

        ax_sin.set_xlabel("Time of Day (h)")
        ax_sin.set_ylabel(f"Concentration ({unit})")
        ax_sin.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1)
        ax_sin.set_xlim(0, 24)
        st.pyplot(fig_sin)
        plt.close(fig_sin)
        
        c1, c2 = st.columns(2, gap="medium")
        y_t1 = circadian(t1_hour, M, A, t0)
        
        with c1:
            st.markdown("##### Chronomap")
            T1, T2, delta = chronomap_delta(A, M, t0)
            delta_disp = delta / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else delta
            
            fig_cm, ax_cm = plt.subplots(figsize=(5, 5), facecolor='white')
            pcm = ax_cm.pcolormesh(T2, T1, delta_disp, cmap="coolwarm", shading='gouraud')
            fig_cm.colorbar(pcm, ax=ax_cm, label=f"Diff ({unit})", fraction=0.046, pad=0.04)
            ax_cm.contour(T2, T1, delta, levels=[mu_abs], colors='black', linestyles='dotted')
            
            delta_h = st.slider("Δ Time t₂ (h)", 0.0, 24.0, 6.0, 0.25)
            t2_hour = (t1_hour + delta_h) % 24
            
            # t1: Black Solid, t2: Green Dashed
            ax_cm.axhline(t1_hour, color='black', ls='-', lw=2.5, label='t₁')
            ax_cm.axvline(t2_hour, color='#2ca02c', ls='--', lw=2.5, label='t₂') # Stronger Green
            
            ax_cm.plot(t2_hour, t1_hour, 'ko', markersize=7, mfc='white')
            ax_cm.set_xlabel("Timepoint t₂"); ax_cm.set_ylabel("Timepoint t₁")
            ax_cm.legend(fontsize='x-small', loc='upper right', frameon=True, facecolor='white')
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            
        with c2:
            st.markdown("##### 24h Clock Comparison")
            y_t2 = circadian(t2_hour, M, A, t0)
            diff = abs(y_t1 - y_t2)
            conv = GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else 1.0
            
            if diff <= mu_abs: 
                st.success(f"**Comparable**\nΔ = {diff/conv:.1f} (≤ {mu_abs/conv:.1f})")
            else: 
                st.error(f"**Not Comparable**\nΔ = {diff/conv:.1f} (> {mu_abs/conv:.1f})")
            
            norm = lambda y: 0.1 + 0.9 * ((y - (M-A)) / (2*A))
            r1, r2 = np.clip(norm(y_t1), 0, 1), np.clip(norm(y_t2), 0, 1)
            theta1, theta2 = (t1_hour/24)*2*np.pi, (t2_hour/24)*2*np.pi
            
            fig_clk, ax_clk = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(5, 5), facecolor='white')
            ax_clk.set_theta_offset(np.pi/2); ax_clk.set_theta_direction(-1)
            ax_clk.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
            ax_clk.set_xticklabels([f"{h*2}" for h in range(12)])
            ax_clk.set_yticklabels([])
            
            y1_d = y_t1/conv; y2_d = y_t2/conv
            
            # t1: Black Solid, t2: Green Dashed
            ax_clk.plot([theta1, theta1], [0, r1], color='black', lw=3, ls='-', label=f"t₁ ({y1_d:.1f})")
            ax_clk.plot([theta2, theta2], [0, r2], color='#2ca02c', lw=3, ls='--', label=f"t₂ ({y2_d:.1f})")
            
            ax_clk.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=True, facecolor='white')
            st.pyplot(fig_clk)
            plt.close(fig_clk)

# --- TAB 2: DATA ANALYSIS ---
with tab2:
    st.markdown("### 1. Data Import")
    col_dl, col_up1, col_up2 = st.columns([1, 1, 1])
    
    with col_dl:
        st.info("Download the template file to see the required structure.")
        csv_data = generate_template_csv()
        st.download_button("📄 Download CSV Template", csv_data, "circadian_template.csv", "text/csv")
        
    df1, df2 = None, None
    with col_up1:
        f1 = st.file_uploader("Upload Control Group", type=["csv"], key="file1")
        if f1: df1 = load_and_process_data(f1, "File 1")
    with col_up2:
        f2 = st.file_uploader("Upload Test Group", type=["csv"], key="file2")
        if f2: df2 = load_and_process_data(f2, "File 2")

    valid_dfs = [df for df in [df1, df2] if df is not None]

    if valid_dfs:
        df_combined = pd.concat(valid_dfs, ignore_index=True)
        st.success(f"Loaded {len(df_combined)} records.")
        st.markdown("---")
        st.markdown("### 2. Filters & Visualization")
        
        # Analyte Filter Logic
        available_analytes = sorted(df_combined['analyte'].unique())
        default_ix = available_analytes.index("Glucose") if "Glucose" in available_analytes else 0
        
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        analyte_filter = f_col1.selectbox("Analyte", available_analytes, index=default_ix)
        
        df_analyte = df_combined[df_combined['analyte'] == analyte_filter]
        current_unit = df_analyte['unit'].mode()[0] if not df_analyte['unit'].empty else "units"
        
        gender_opts = ["All"] + sorted(df_analyte['gender'].unique())
        age_opts = ["All"] + list(df_analyte['age_group'].dropna().unique())
        source_opts = sorted(df_analyte['source_file'].unique())
        display_opts = ["Combined"] + source_opts if len(source_opts) > 1 else source_opts
        
        gender_filter = f_col2.selectbox("Gender", gender_opts)
        age_filter = f_col3.selectbox("Age Group", age_opts)
        display_mode = f_col4.radio("Display Source", display_opts, horizontal=True)

        df_plot = df_analyte.copy()
        if gender_filter != "All": df_plot = df_plot[df_plot['gender'] == gender_filter]
        if age_filter != "All": df_plot = df_plot[df_plot['age_group'] == age_filter]
        
        if display_mode == "Combined":
            plot_color, line_color = '#d9d9d9', '#1f1f1f' # Light gray fill, dark line
        elif display_mode == "File 1":
            df_plot = df_plot[df_plot['source_file'] == 'File 1']
            plot_color, line_color = '#a0c4ff', '#003366'
        else:
            df_plot = df_plot[df_plot['source_file'] == 'File 2']
            plot_color, line_color = '#ffadad', '#800000'

        # Boxplot with White Card Background
        fig_box, ax_box = plt.subplots(figsize=(12, 6), facecolor='white')
        ax_box.set_title(f"{analyte_filter} ({display_mode}) - {gender_filter}, {age_filter}")
        ax_box.set_xlabel("Time of Day (h)")
        ax_box.set_ylabel(f"Concentration ({current_unit})")
        ax_box.set_xlim(-0.5, 23.5); ax_box.set_xticks(range(24))
        
        if not df_plot.empty:
            box_data = [df_plot[df_plot['hour_int'] == h]['value'].values for h in range(24)]
            ax_box.boxplot(box_data, positions=range(24), patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=plot_color, alpha=0.8, edgecolor='#333'),
                           medianprops=dict(color=line_color, linewidth=2))
            
            medians = df_plot.groupby('hour_int')['value'].median().reindex(range(24))
            ax_box.plot(range(24), medians, 'o-', color=line_color, lw=2, label='Median')
            ax_box.legend(frameon=True, facecolor='white')
        else:
            ax_box.text(12, 0, "No data found", ha='center')
            
        st.pyplot(fig_box)
        plt.close(fig_box)

        st.markdown("### 3. Model Parameters")
        results = []
        df_fit_base = df_analyte if display_mode == "Combined" else df_analyte[df_analyte['source_file'] == display_mode]
        genders = sorted(df_fit_base['gender'].unique())
        ages = df_fit_base['age_group'].cat.categories
        
        if genders and not ages.empty:
            fig_grid, axes = plt.subplots(len(ages), len(genders), figsize=(10, 3*len(ages)), 
                                          sharex=True, sharey=True, squeeze=False, facecolor='white')
            
            for i, ag in enumerate(ages):
                for j, gen in enumerate(genders):
                    ax = axes[i, j]
                    ax.set_title(f"{gen}, {ag}", fontsize=9)
                    sub = df_fit_base[(df_fit_base['gender'] == gen) & (df_fit_base['age_group'] == ag)]
                    if not sub.empty:
                        M_fit, A_fit, t0_fit = get_fitted_parameters(sub)
                        meds = sub.groupby('hour_int')['value'].median()
                        ax.plot(meds.index, meds.values, 'o', color='black', ms=4, alpha=0.6)
                        if not np.isnan(M_fit):
                            t_lin = np.linspace(0, 24, 100)
                            y_lin = circadian(t_lin, M_fit, A_fit, t0_fit)
                            ax.plot(t_lin, y_lin, '-', color='red', alpha=0.8)
                            results.append({"Gender": gen, "Age": ag, "M": M_fit, "A": A_fit, "t0": t0_fit%24})
                    ax.set_xlim(0,24)
            st.pyplot(fig_grid)
            plt.close(fig_grid)
            
            if results:
                st.dataframe(pd.DataFrame(results).set_index(['Gender', 'Age']).style.format("{:.2f}"))
    else:
        st.warning("Please upload a CSV file using the template structure to begin.")