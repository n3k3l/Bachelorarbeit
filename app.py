import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.optimize import curve_fit

# ───────────────────────────────────────────
# 0) PAGE CONFIG & CSS STYLING
# ───────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Circadian Analysis Dashboard")

# Custom CSS for a "Medical Dashboard" Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Containers / Cards style */
    div.css-1r6slb0, div.css-12w0qpk {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px;
        padding-left: 20px;
        padding-right: 20px;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: bold;
        border-bottom: 2px solid #0d47a1;
    }

    /* Info Box Styling */
    .metric-card {
        background-color: white;
        border-left: 5px solid #0d47a1;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────
# 1) CONSTANTS & HELPERS
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

def style_plot(fig, ax):
    """Applies a consistent clean style to plots."""
    sns.despine()
    ax.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    # Text styling
    ax.title.set_color('#2c3e50')
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_color('#555')
    ax.yaxis.label.set_color('#555')
    ax.tick_params(colors='#555')
    return fig, ax

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
    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')

@st.cache_data
def load_and_process_data(_file, file_identifier):
    try:
        df = pd.read_csv(_file, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.upper()
        
        # Mappings
        COLUMN_MAP = {'ANALYT': 'analyte', 'VALUE': 'value', 'DIM': 'unit', 'TIME': 'timestamp', 'SEX': 'gender', 'AGE': 'age'}
        
        # Check required columns
        required_cols = set(COLUMN_MAP.keys())
        found_cols = set(df.columns)
        
        if not required_cols.issubset(found_cols):
            # Fallback mapping
            fallback_map = {'GENDER': 'gender', 'AGE': 'age', 'VALUE': 'value', 'ANALYSE_DATE': 'timestamp'}
            renamed = False
            for k, v in fallback_map.items():
                if k in found_cols:
                    df.rename(columns={k: v}, inplace=True); renamed = True
            if not renamed:
                st.error(f"Missing columns in {file_identifier}. Needed: {list(required_cols)}")
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
            
        df.dropna(subset=['value', 'age', 'timestamp'], inplace=True)
        df['age'] = df['age'].astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['hour_int'] = df['timestamp'].dt.hour
        
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.upper().map({'M': 'Male', 'F': 'Female'}).fillna('Other')
            
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 120], labels=["< 30", "30-50", "> 50"], right=False)
        df['source_file'] = file_identifier
        return df
    except Exception as e:
        st.error(f"Error processing '{file_identifier}': {e}")
        return None

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
    except:
        return np.nan, np.nan, np.nan

# ───────────────────────────────────────────
# MAIN APP HEADER
# ───────────────────────────────────────────
st.markdown("<h1 style='text-align:center; margin-top:0px;'>Diurnal Fluctuation Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#7f8c8d; margin-bottom: 30px;'>Evaluation of circadian rhythms and comparability of measurement time points.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Circadian Model (Simulation)", "📂 Data Analysis & Estimation"])

# ───────────────────────────────────────────
# TAB 1: SIMULATION
# ───────────────────────────────────────────
with tab1:
    col_input, col_viz = st.columns([1, 2.5], gap="large")
    
    # --- LEFT PANEL: INPUTS ---
    with col_input:
        st.markdown("### 🛠 Configuration")
        with st.container():
            st.markdown("##### 1. Patient Profile")
            c_a, c_b = st.columns(2)
            analyte = c_a.selectbox("Analyte", ["Glucose", "Cortisol", "Other"])
            gender = c_b.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 0, 100, 35)
            
            unit = "mg/dL"
            if analyte == "Glucose":
                unit = st.radio("Unit", ["mg/dL", "mmol/L"], horizontal=True)

            st.markdown("---")
            st.markdown("##### 2. Reference Time ($t_1$)")
            t1_time = st.time_input("Measurement Time t₁", value=datetime.time(8, 0))
            t1_hour = t1_time.hour + t1_time.minute / 60
            
            # Load Defaults
            p = default_params.get((analyte, gender), default_params[("Other", "Male")])
            t0_lit, A_lit, MU_perc_lit, M_literature = p["t0"], p["A"], p["MU"], p["M"]
            
            st.markdown("---")
            st.markdown("##### 3. Calibration")
            personalize_mode = st.checkbox("Adjust Mean to measured value", value=True)
            M = M_literature
            if personalize_mode:
                val_default = M_literature / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else float(M_literature)
                step_val = 0.1 if unit == 'mmol/L' else 1.0
                y_measured = st.number_input(f"Value at {format_time_string(t1_hour)}", value=val_default, step=step_val, format="%.1f")
                y_measured_mgdl = y_measured * GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_measured
                M = y_measured_mgdl - A_lit * np.cos(2 * np.pi * (t1_hour - t0_lit) / 24)
            
            with st.expander("Advanced Model Parameters"):
                A = st.slider("Amplitude A", 1.0, 50.0, float(A_lit), 0.5)
                t0 = st.slider("Acrophase t₀ (Peak Hour)", 0.0, 24.0, t0_lit, 0.1)
                MU_perc = st.slider("Uncertainty (MU) %", 1.0, 50.0, float(MU_perc_lit), 0.5)
                if not personalize_mode:
                    M = st.slider("Mean (M)", 0.0, 300.0, float(M), 1.0)
            
            mu_abs = M * MU_perc / 100

    # --- RIGHT PANEL: VISUALIZATION ---
    with col_viz:
        # Pre-calc
        t_arr = np.linspace(0, 24, 500)
        y_arr = circadian(t_arr, M, A, t0)
        
        # Units conversion for display
        conv_factor = GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else 1.0
        y_disp, mu_disp = y_arr / conv_factor, mu_abs / conv_factor
        
        # 1. Main Plot
        st.markdown("#### Expected Daily Profile")
        fig_sin, ax_sin = plt.subplots(figsize=(10, 3.5))
        style_plot(fig_sin, ax_sin)
        
        # Curve
        ax_sin.plot(t_arr, y_disp, color="#3498db", linewidth=2, label="Circadian Model")
        ax_sin.fill_between(t_arr, y_disp - mu_disp, y_disp + mu_disp, color="#3498db", alpha=0.15, label=f"Uncertainty (±{MU_perc}%)")
        
        # t1 Line (Black Solid)
        ax_sin.axvline(t1_hour, color="black", ls="-", lw=2, label=f"t₁: {format_time_string(t1_hour)}")
        if personalize_mode:
            ax_sin.plot(t1_hour, y_measured, 'o', color='black', markeredgecolor='white', markersize=8, zorder=5)

        ax_sin.set_xlabel("Time of Day (h)")
        ax_sin.set_ylabel(f"Concentration ({unit})")
        ax_sin.legend(loc='upper right', frameon=True, framealpha=0.9)
        ax_sin.set_xlim(0, 24)
        st.pyplot(fig_sin)
        plt.close(fig_sin)
        
        # 2. Comparison Section
        st.markdown("---")
        c1, c2 = st.columns([1, 1], gap="medium")
        
        y_t1 = circadian(t1_hour, M, A, t0)
        
        with c1:
            st.markdown("#### Chronomap (Delta Analysis)")
            T1, T2, delta = chronomap_delta(A, M, t0)
            delta_disp = delta / conv_factor
            
            # Interactive Slider for t2
            delta_h = st.slider("Time difference Δt (Hours)", 0.0, 24.0, 6.0, 0.25)
            t2_hour = (t1_hour + delta_h) % 24
            
            fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
            style_plot(fig_cm, ax_cm)
            pcm = ax_cm.pcolormesh(T2, T1, delta_disp, cmap="RdYlBu_r", shading='gouraud') # RdYlBu_r is cleaner for heatmaps
            cbar = fig_cm.colorbar(pcm, ax=ax_cm, fraction=0.046, pad=0.04)
            cbar.set_label(f"Difference ({unit})", color='#555')
            cbar.ax.yaxis.set_tick_params(color='#555')
            
            # Contour
            ax_cm.contour(T2, T1, delta, levels=[mu_abs], colors='#2c3e50', linestyles='dotted', linewidths=1.5)
            
            # Lines: t1 (Black Solid), t2 (Green Dashed)
            ax_cm.axhline(t1_hour, color='black', ls='-', lw=2, label='t₁')
            ax_cm.axvline(t2_hour, color='#27ae60', ls='--', lw=2, label='t₂')
            ax_cm.plot(t2_hour, t1_hour, 'o', color='#2c3e50', markeredgecolor='white', markersize=7)
            
            ax_cm.set_xlabel("Timepoint t₂")
            ax_cm.set_ylabel("Timepoint t₁")
            ax_cm.legend(fontsize='x-small', loc='upper right', facecolor='white', framealpha=0.9)
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            
        with c2:
            st.markdown("#### Comparison Result")
            y_t2 = circadian(t2_hour, M, A, t0)
            diff = abs(y_t1 - y_t2)
            
            # Result Card
            is_comparable = diff <= mu_abs
            bg_color = "#d4edda" if is_comparable else "#f8d7da"
            text_color = "#155724" if is_comparable else "#721c24"
            status_icon = "✅" if is_comparable else "⚠️"
            status_text = "COMPARABLE" if is_comparable else "NOT COMPARABLE"
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; color:{text_color}; padding:15px; border-radius:8px; text-align:center; border:1px solid {text_color}; margin-bottom:15px;">
                <h3 style="margin:0; color:{text_color};">{status_icon} {status_text}</h3>
                <p style="margin:5px 0 0 0;">Difference: <b>{diff/conv_factor:.2f}</b> {unit}</p>
                <p style="margin:0; font-size:0.9em;">(Threshold: {mu_abs/conv_factor:.2f} {unit})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Polar Clock
            norm = lambda y: 0.1 + 0.9 * ((y - (M-A)) / (2*A))
            r1, r2 = np.clip(norm(y_t1), 0, 1), np.clip(norm(y_t2), 0, 1)
            theta1, theta2 = (t1_hour/24)*2*np.pi, (t2_hour/24)*2*np.pi
            
            fig_clk, ax_clk = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(5, 5))
            fig_clk.patch.set_facecolor('white') # Ensure white bg for polar
            ax_clk.set_theta_offset(np.pi/2)
            ax_clk.set_theta_direction(-1)
            ax_clk.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
            ax_clk.set_xticklabels([f"{h*2}" for h in range(12)], color='#555')
            ax_clk.set_yticklabels([])
            ax_clk.grid(color='#bdc3c7', alpha=0.5)
            
            y1_d = y_t1/conv_factor; y2_d = y_t2/conv_factor
            
            # t1 (Black Solid)
            ax_clk.plot([theta1, theta1], [0, r1], color='black', lw=2.5, ls='-', label=f"t₁: {format_time_string(t1_hour)}")
            # t2 (Green Dashed - adjusted color to standard Green)
            ax_clk.plot([theta2, theta2], [0, r2], color='#27ae60', lw=2.5, ls='--', label=f"t₂: {format_time_string(t2_hour)}")
            
            ax_clk.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
            st.pyplot(fig_clk)
            plt.close(fig_clk)

# ───────────────────────────────────────────
# TAB 2: DATA ANALYSIS
# ───────────────────────────────────────────
with tab2:
    # --- Upload Section ---
    st.markdown("### 1. Data Ingestion")
    
    with st.expander("ℹ️ Help & Template", expanded=False):
        st.info("Upload CSV files containing patient data. Use the template below for correct formatting.")
        st.download_button("📄 Download Template CSV", generate_template_csv(), "template.csv", "text/csv")
    
    uc1, uc2 = st.columns(2)
    with uc1:
        f1 = st.file_uploader("📂 Upload Control Group (File 1)", type=["csv"])
    with uc2:
        f2 = st.file_uploader("📂 Upload Test Group (File 2)", type=["csv"])

    df1 = load_and_process_data(f1, "File 1") if f1 else None
    df2 = load_and_process_data(f2, "File 2") if f2 else None
    
    valid_dfs = [d for d in [df1, df2] if d is not None]

    if valid_dfs:
        df_combined = pd.concat(valid_dfs, ignore_index=True)
        st.success(f"Successfully loaded {len(df_combined)} records.")
        st.markdown("---")
        
        # --- Filters Section ---
        st.markdown("### 2. Analysis Filters")
        
        # Prepare filters
        all_analytes = sorted(df_combined['analyte'].unique())
        default_idx = all_analytes.index("Glucose") if "Glucose" in all_analytes else 0
        
        # Filter Row
        fc1, fc2, fc3, fc4 = st.columns(4)
        sel_analyte = fc1.selectbox("🧪 Analyte", all_analytes, index=default_idx)
        
        # Subset data based on analyte first
        df_sub = df_combined[df_combined['analyte'] == sel_analyte]
        current_unit = df_sub['unit'].mode()[0] if not df_sub['unit'].empty else "units"
        
        sel_gender = fc2.selectbox("👤 Gender", ["All"] + sorted(df_sub['gender'].unique()))
        sel_age = fc3.selectbox("🎂 Age Group", ["All"] + list(df_sub['age_group'].dropna().unique()))
        
        sources = sorted(df_sub['source_file'].unique())
        sel_source = fc4.radio("Display Mode", ["Combined"] + sources if len(sources)>1 else sources, horizontal=True)

        # Apply secondary filters
        df_plot = df_sub.copy()
        if sel_gender != "All": df_plot = df_plot[df_plot['gender'] == sel_gender]
        if sel_age != "All": df_plot = df_plot[df_plot['age_group'] == sel_age]
        
        # Colors definition
        if sel_source == "Combined":
            col_main, col_line = '#95a5a6', '#2c3e50' # Grey/DarkBlue
        elif sel_source == "File 1":
            col_main, col_line = '#aed6f1', '#1b4f72' # LightBlue/DarkBlue
        else:
            col_main, col_line = '#f5b7b1', '#922b21' # LightRed/DarkRed

        if sel_source != "Combined":
            df_plot = df_plot[df_plot['source_file'] == sel_source]

        # --- Visualization ---
        st.markdown("### 3. Visualizations")
        
        # Boxplot
        fig_box, ax_box = plt.subplots(figsize=(12, 5))
        style_plot(fig_box, ax_box)
        ax_box.set_title(f"Distribution: {sel_analyte} | {sel_gender} | {sel_age}", pad=15)
        
        if not df_plot.empty:
            box_data = [df_plot[df_plot['hour_int'] == h]['value'].values for h in range(24)]
            bp = ax_box.boxplot(box_data, positions=range(24), patch_artist=True, showfliers=False, widths=0.6)
            
            # Style Boxplot
            for patch in bp['boxes']:
                patch.set_facecolor(col_main)
                patch.set_alpha(0.8)
                patch.set_edgecolor('white')
            for median in bp['medians']:
                median.set_color(col_line)
                median.set_linewidth(2)
            
            # Median Line Overlay
            medians = df_plot.groupby('hour_int')['value'].median().reindex(range(24))
            ax_box.plot(range(24), medians, 'o-', color=col_line, lw=2, label='Hourly Median')
            ax_box.legend(loc='upper left', frameon=False)
        else:
            ax_box.text(12, 0, "No data matching filters.", ha='center', fontsize=12)
            
        ax_box.set_xlim(-0.5, 23.5)
        ax_box.set_xticks(range(0, 24, 2)) # Less crowded ticks
        ax_box.set_xlabel("Time of Day (h)")
        ax_box.set_ylabel(f"Concentration ({current_unit})")
        st.pyplot(fig_box)
        plt.close(fig_box)

        # --- Small Multiples (Fitting) ---
        st.markdown("#### Subgroup Modeling")
        
        # Determine dataset for fitting
        df_fit = df_sub if sel_source == "Combined" else df_sub[df_sub['source_file'] == sel_source]
        
        genders = sorted(df_fit['gender'].unique())
        ages = df_fit['age_group'].cat.categories
        
        if genders and not ages.empty:
            # Create Grid
            fig_grid, axes = plt.subplots(len(ages), len(genders), figsize=(12, 3.5*len(ages)), sharex=True, sharey=True)
            if len(ages) == 1 and len(genders) == 1: axes = np.array([[axes]])
            elif len(ages) == 1: axes = axes.reshape(1, -1)
            elif len(genders) == 1: axes = axes.reshape(-1, 1)

            results_list = []
            
            for i, ag in enumerate(ages):
                for j, gen in enumerate(genders):
                    ax = axes[i, j]
                    style_plot(fig_grid, ax) # Apply clean style
                    ax.set_title(f"{gen}, {ag}", fontsize=10, weight='bold')
                    
                    sub = df_fit[(df_fit['gender'] == gen) & (df_fit['age_group'] == ag)]
                    
                    if not sub.empty:
                        # Raw medians
                        meds = sub.groupby('hour_int')['value'].median()
                        ax.plot(meds.index, meds.values, 'o', color='#7f8c8d', ms=4, alpha=0.6, label='Data')
                        
                        # Fit
                        M_f, A_f, t0_f = get_fitted_parameters(sub)
                        if not np.isnan(M_f):
                            t_l = np.linspace(0, 24, 100)
                            y_l = circadian(t_l, M_f, A_f, t0_f)
                            ax.plot(t_l, y_l, '-', color='#e74c3c', lw=2, label='Fit') # Red for fit line
                            
                            results_list.append({'Gender': gen, 'Age': ag, 'M': M_f, 'A': A_f, 't0': t0_f%24})
                    
                    if i == len(ages)-1: ax.set_xlabel("Hour")
                    if j == 0: ax.set_ylabel(current_unit)
                    ax.set_xlim(0, 24)
            
            st.pyplot(fig_grid)
            plt.close(fig_grid)
            
            if results_list:
                st.markdown("##### Estimated Parameters")
                st.dataframe(pd.DataFrame(results_list).set_index(['Gender', 'Age']).style.format("{:.2f}").background_gradient(cmap="Blues"))
    else:
        st.info("👋 Upload data to begin analysis.")