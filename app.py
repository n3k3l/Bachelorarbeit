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

# OPTIMIZED CSS
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp { background-color: #f0f2f6; }
    
    /* 2. Global Text Color */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, label, li, span {
        color: #1f1f1f;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    
    /* 3. INPUT FIELDS */
    div[data-baseweb="input"] {
        background-color: #4a4a4a !important; 
        border: 1px solid #666 !important;
    }
    div[data-baseweb="input"] input {
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }
    
    /* 4. DROPDOWNS */
    div[data-baseweb="select"] > div {
        background-color: #4a4a4a !important; 
        border: 1px solid #666 !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] div[class*="content"] {
        color: #ffffff !important;
    }
    div[data-baseweb="select"] svg { 
        fill: #ffffff !important; 
    }
    
    /* POPUP MENU */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[data-baseweb="menu"] {
        background-color: #4a4a4a !important;
    }
    li[data-baseweb="option"] {
        background-color: #4a4a4a !important;
        color: #ffffff !important;
    }
    li[data-baseweb="option"] * {
        color: #ffffff !important; 
    }
    li[data-baseweb="option"]:hover, li[aria-selected="true"] {
        background-color: #666666 !important;
    }

    /* 5. THE ULTIMATE BUTTON FIX */
    div[data-testid="stButton"] button, 
    div[data-testid="stDownloadButton"] button {
        background-color: #4a4a4a !important;
        border: 1px solid #666 !important;
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button *, 
    div[data-testid="stDownloadButton"] button * {
        color: #ffffff !important;
        fill: #ffffff !important; 
    }
    div[data-testid="stButton"] button:hover, 
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #555555 !important;
        border-color: #ff4b4b !important;
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button:hover *, 
    div[data-testid="stDownloadButton"] button:hover * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* 6. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
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
    
    /* 7. Slider Labels */
    div[data-testid="stThumbValue"] { color: #1f1f1f !important; }
    
    /* 8. Number Input Buttons */
    button[kind="secondary"] { color: #ffffff !important; }
    div[data-baseweb="input"] button svg { fill: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")
plt.rcParams['text.color'] = '#1f1f1f'
plt.rcParams['axes.labelcolor'] = '#1f1f1f'
plt.rcParams['xtick.color'] = '#1f1f1f'
plt.rcParams['ytick.color'] = '#1f1f1f'

st.markdown("<h2 style='text-align:center; margin-bottom: 25px;'>Visualization & Analysis of Diurnal Fluctuations</h2>", unsafe_allow_html=True)

# ───────────────────────────────────────────
# 1) CONSTANTS & CONVERSION LOGIC
# ───────────────────────────────────────────
# Base Units in default_params:
# Glucose -> mg/dL
# Cortisol -> µg/dL (Micrograms per Deciliter)

default_params = {
    # Glucose (Base: mg/dL)
    ("Glucose", "Male"):    {"t0": 14.0, "A": 10, "MU": 12, "M": 100},
    ("Glucose", "Female"):  {"t0": 14.5, "A": 10, "MU": 11, "M":  95},
    
    # Cortisol (Base: µg/dL) - Corrected to realistic µg/dL values (Mean ~15, Amp ~7)
    ("Cortisol", "Male"):   {"t0": 7.0, "A": 8.0, "MU": 10, "M": 15.0}, 
    ("Cortisol", "Female"): {"t0": 7.5, "A": 7.5, "MU": 10, "M": 14.0},
    
    # Other (Arbitrary)
    ("Other", "Male"):      {"t0": 4.0, "A": 10, "MU": 15, "M": 100},
    ("Other", "Female"):    {"t0": 4.5, "A": 10, "MU": 15, "M":  95},
}

# Factors to convert FROM Base Unit TO SI Unit
FACTORS = {
    "Glucose": 18.016,   # Divide by this (mg/dL -> mmol/L)
    "Cortisol": 27.586   # Multiply by this (µg/dL -> nmol/L)
}

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
        'SEX': ['M', 'F', 'M', 'F', 'D'],
        'AGE': [47, 32, 55, 29, 60]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def load_and_process_data(_file, file_identifier):
    try:
        filename = _file.name.lower()
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df_raw = pd.read_excel(_file, header=None, nrows=20)
        else:
            try:
                df_raw = pd.read_csv(_file, sep=None, engine='python', header=None, nrows=20, encoding='utf-8-sig')
            except:
                _file.seek(0)
                df_raw = pd.read_csv(_file, sep=None, engine='python', header=None, nrows=20, encoding='latin1')

        header_row_idx = 0
        header_keywords = ['VALUE', 'WERT', 'RESULT', 'ANALYT', 'TIME', 'DATUM']
        for idx, row in df_raw.iterrows():
            row_str = " ".join(row.astype(str)).upper()
            matches = sum(1 for kw in header_keywords if kw in row_str)
            if matches >= 2:
                header_row_idx = idx
                break
        
        _file.seek(0)
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(_file, header=header_row_idx)
        else:
            try:
                df = pd.read_csv(_file, sep=None, engine='python', header=header_row_idx, encoding='utf-8-sig')
            except:
                _file.seek(0)
                df = pd.read_csv(_file, sep=None, engine='python', header=header_row_idx, encoding='latin1')

        df.columns = df.columns.astype(str).str.strip().str.replace('"', '').str.replace("'", "").str.upper()

        column_candidates = {
            'value':     ['VALUE', 'WERT', 'MESSWERT', 'CONCENTRATION'],
            'timestamp': ['TIME', 'ZEIT', 'DATE', 'DATUM', 'ANALYSE_DATE', 'TIMESTAMP'],
            'gender':    ['SEX', 'GENDER', 'GESCHLECHT'], 
            'age':       ['AGE', 'ALTER', 'JAHRE'],       
            'analyte':   ['ANALYT', 'ANALYTE', 'PARAMETER', 'STOFF'],
            'unit':      ['DIM', 'UNIT', 'EINHEIT', 'DIMENSION']
        }
        
        found_mapping = {}
        existing_cols = list(df.columns)
        for target, candidates in column_candidates.items():
            for candidate in candidates:
                if candidate in existing_cols: found_mapping[candidate] = target; break 
        for target, candidates in column_candidates.items():
            if target not in found_mapping.values():
                for col in existing_cols:
                    for candidate in candidates:
                        if candidate in col and col not in found_mapping: found_mapping[col] = target; break
                    if target in found_mapping.values(): break

        df.rename(columns=found_mapping, inplace=True)
        if 'value' not in df.columns or 'timestamp' not in df.columns:
            st.error(f"Error in '{file_identifier}': Mandatory columns 'VALUE' or 'TIME' missing.")
            return None

        if df['value'].dtype == object:
            df['value'] = df['value'].astype(str).str.replace(',', '.', regex=False)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value'], inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['hour_int'] = df['timestamp'].dt.hour
        
        if 'analyte' in df.columns:
            df['analyte'] = df['analyte'].astype(str).str.strip().str.title()
            df['analyte'].replace(['', 'Nan', 'None', 'Nan'], 'Unknown Analyte', inplace=True)
            df['analyte'] = df['analyte'].fillna('Unknown Analyte')
        else:
            df['analyte'] = "Unknown Analyte"

        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['has_age'] = True
        else:
            df['age'] = np.nan
            df['has_age'] = False
            
        if 'gender' in df.columns:
            def clean_gender(x):
                s = str(x).strip().upper()
                if not s or s == 'NAN': return 'Not Specified'
                if s.startswith('M') or s.startswith('H'): return 'Male'
                if s.startswith('F') or s.startswith('W'): return 'Female'
                if s.startswith('D'): return 'Divers'
                return 'Not Specified'
            df['gender'] = df['gender'].apply(clean_gender)
            df['has_gender'] = True
        else:
            df['gender'] = 'Not Specified'
            df['has_gender'] = False
            
        if 'unit' in df.columns:
            df['unit'] = df['unit'].astype(str).str.strip()
        else:
            df['unit'] = 'units'
        
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 120], labels=["< 30 years", "30-50 years", "> 50 years"], right=False)
        df['age_group'] = df['age_group'].cat.add_categories(["No Age Data"])
        df['age_group'] = df['age_group'].fillna("No Age Data")
        df['source_file'] = file_identifier
        return df
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten von '{file_identifier}': {e}")
        return None

def calculate_r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / ss_tot)

def get_fitted_parameters(df_group, value_column='value'):
    if len(df_group) < 5: return np.nan, np.nan, np.nan, np.nan
    medians = df_group.groupby('hour_int')[value_column].median()
    if len(medians) < 3: return np.nan, np.nan, np.nan, np.nan
    x, y = medians.index.values, medians.values
    M_guess, A_guess = np.mean(y), (np.max(y) - np.min(y)) / 2
    t0_guess = x[np.argmax(y)]
    try:
        popt, _ = curve_fit(circadian, x, y, p0=[M_guess, A_guess, t0_guess], bounds=([0, 0, -24], [np.inf, np.inf, 48]), maxfev=10000)
        y_pred = circadian(x, *popt)
        r2 = calculate_r_squared(y, y_pred)
        return popt[0], popt[1], popt[2], r2
    except: 
        return np.nan, np.nan, np.nan, np.nan

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
        
        # --- UNIT SELECTION LOGIC ---
        unit_label = "Unit"
        unit_options = ["units"]
        
        if analyte == "Glucose":
            unit_options = ["mg/dL", "mmol/L"]
        elif analyte == "Cortisol":
            unit_options = ["µg/dL", "nmol/L"]
            
        selected_unit = st.radio(f"Unit for {analyte}", unit_options, horizontal=True)
        
        # Load Defaults (Base Units)
        p = default_params.get((analyte, gender), default_params[("Other", "Male")])
        t0_lit, A_lit, MU_perc_lit, M_literature = p["t0"], p["A"], p["MU"], p["M"]
        
        # --- CONVERSION LOGIC FOR DISPLAY/INPUT ---
        # Helper to convert Base -> Display
        def to_display(val, ana, unit):
            if ana == "Glucose" and unit == "mmol/L":
                return val / FACTORS["Glucose"]
            if ana == "Cortisol" and unit == "nmol/L":
                return val * FACTORS["Cortisol"]
            return val

        # Helper to convert Display -> Base (for Input)
        def to_base(val, ana, unit):
            if ana == "Glucose" and unit == "mmol/L":
                return val * FACTORS["Glucose"]
            if ana == "Cortisol" and unit == "nmol/L":
                return val / FACTORS["Cortisol"]
            return val
        
        st.markdown("**1. Adjust Model**")
        personalize_mode = st.checkbox("Adjust Mean (M) to measured value at t₁", value=True)
        M = M_literature
        
        if personalize_mode:
            # Show default value in SELECTED unit
            val_default_disp = to_display(M_literature, analyte, selected_unit)
            
            step_val = 0.1 if (selected_unit in ["mmol/L", "µg/dL"]) else 1.0
            y_measured_disp = st.number_input(f"Value at t₁ ({format_time_string(t1_hour)}) in {selected_unit}", 
                                              value=float(val_default_disp), step=step_val, format="%.1f")
            
            # Convert Input back to Base Unit for Calculation
            y_measured_base = to_base(y_measured_disp, analyte, selected_unit)
            M = y_measured_base - A_lit * np.cos(2 * np.pi * (t1_hour - t0_lit) / 24)
            
        st.markdown("**2. Manual Adjustments**")
        if st.checkbox("Enable Editor Mode"):
            # Sliders operate in Base Units (approximate scaling)
            A = st.slider(f"Amplitude A (Base: {unit_options[0]})", 0.1, 50.0, float(A_lit), 0.1)
            M = st.slider(f"Mean M (Base: {unit_options[0]})", 0.0, 300.0, float(M), 0.5, disabled=personalize_mode)
            t0 = st.slider("Acrophase t₀ (h)", 0.0, 24.0, t0_lit, 0.1)
            MU_perc = st.slider("Uncertainty MU %", 1.0, 50.0, float(MU_perc_lit), 0.5)
        else:
            A, t0, MU_perc = A_lit, t0_lit, MU_perc_lit
            
        mu_abs = M * MU_perc / 100

    with right:
        # Calculate Curve in Base Units
        t_arr = np.linspace(0, 24, 500)
        y_arr = circadian(t_arr, M, A, t0)
        
        # Convert Curve to Display Units
        y_disp = to_display(y_arr, analyte, selected_unit)
        mu_disp = to_display(mu_abs, analyte, selected_unit)

        fig_sin, ax_sin = plt.subplots(figsize=(10, 3.5), facecolor='white')
        ax_sin.set_title(f"Simulated Diurnal Fluctuation for {analyte}", fontsize=12)
        ax_sin.plot(t_arr, y_disp, color="cornflowerblue", alpha=0.9, lw=2, label="Expected Profile")
        ax_sin.axvline(t1_hour, color="black", ls="-", lw=2, label=f"t₁ = {format_time_string(t1_hour)}")
        ax_sin.fill_between(t_arr, y_disp - mu_disp, y_disp + mu_disp, color="gray", alpha=0.15, label=f"Tolerance (±{MU_perc}%)")
        
        if personalize_mode:
             # Plot the measured point (already in display units from input)
             ax_sin.plot(t1_hour, y_measured_disp, 'o', color='black', markersize=6)
             
        ax_sin.set_xlabel("Time of Day (h)"); ax_sin.set_ylabel(f"Concentration ({selected_unit})")
        ax_sin.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1)
        ax_sin.set_xlim(0, 24)
        st.pyplot(fig_sin)
        plt.close(fig_sin)
        
        st.markdown("---")

        row1_c1, row1_c2 = st.columns(2, gap="medium")
        with row1_c1:
            st.markdown("##### Chronomap")
            delta_h = st.slider("Δ Time t₂ (h)", 0.0, 24.0, 6.0, 0.25)
            t2_hour = (t1_hour + delta_h) % 24
            
        with row1_c2:
            st.markdown("##### 24h Clock Comparison")
            # Calculate values in Base units
            y_t1_base = circadian(t1_hour, M, A, t0)
            y_t2_base = circadian(t2_hour, M, A, t0)
            diff_base = abs(y_t1_base - y_t2_base)
            
            # Convert difference to display unit for Text
            diff_disp = to_display(diff_base, analyte, selected_unit)
            mu_abs_disp = to_display(mu_abs, analyte, selected_unit) # Threshold in display units
            
            if diff_base <= mu_abs: 
                st.success(f"**Comparable**\nΔ = {diff_disp:.1f} (≤ {mu_abs_disp:.1f})")
            else: 
                st.error(f"**Not Comparable**\nΔ = {diff_disp:.1f} (> {mu_abs_disp:.1f})")

        row2_c1, row2_c2 = st.columns(2, gap="medium")
        
        with row2_c1:
            T1, T2, delta = chronomap_delta(A, M, t0)
            # Map values need to be converted for colorbar
            delta_disp = to_display(delta, analyte, selected_unit)
            
            fig_cm, ax_cm = plt.subplots(figsize=(5, 5), facecolor='white')
            pcm = ax_cm.pcolormesh(T2, T1, delta_disp, cmap="coolwarm", shading='gouraud')
            fig_cm.colorbar(pcm, ax=ax_cm, label=f"Diff ({selected_unit})", fraction=0.046, pad=0.04)
            
            # Contour lines use Base Units because delta is calculated in base units? 
            # No, we want the line to be at mu_abs.
            # Contour uses the grid 'delta'. We must check against mu_abs (base).
            ax_cm.contour(T2, T1, delta, levels=[mu_abs], colors='black', linestyles='dotted')
            
            ax_cm.axhline(t1_hour, color='black', ls='-', lw=2.5, label='t₁')
            ax_cm.axvline(t2_hour, color='#ff7f0e', ls='--', lw=2.5, label='t₂') 
            ax_cm.plot(t2_hour, t1_hour, 'ko', markersize=7, mfc='white')
            ax_cm.set_xlabel("Timepoint t₂"); ax_cm.set_ylabel("Timepoint t₁")
            ax_cm.legend(fontsize='x-small', loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
            
        with row2_c2:
            norm = lambda y: 0.1 + 0.9 * ((y - (M-A)) / (2*A))
            r1, r2 = np.clip(norm(y_t1_base), 0, 1), np.clip(norm(y_t2_base), 0, 1)
            theta1, theta2 = (t1_hour/24)*2*np.pi, (t2_hour/24)*2*np.pi
            
            # Label values in Display Unit
            y1_d = to_display(y_t1_base, analyte, selected_unit)
            y2_d = to_display(y_t2_base, analyte, selected_unit)
            
            fig_clk, ax_clk = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6, 4), facecolor='white')
            ax_clk.set_theta_offset(np.pi/2); ax_clk.set_theta_direction(-1)
            ax_clk.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
            ax_clk.set_xticklabels([f"{h*2}" for h in range(12)])
            ax_clk.set_yticklabels([])
            ax_clk.plot([theta1, theta1], [0, r1], color='black', lw=3, ls='-', label=f"t₁ ({y1_d:.1f})")
            ax_clk.plot([theta2, theta2], [0, r2], color='#ff7f0e', lw=3, ls='--', label=f"t₂ ({y2_d:.1f})")
            ax_clk.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True, facecolor='white', fontsize='small')
            plt.subplots_adjust(top=0.95, bottom=0.20, left=0.05, right=0.95)
            st.pyplot(fig_clk, use_container_width=True)
            plt.close(fig_clk)

# --- TAB 2: DATA ANALYSIS ---
with tab2:
    st.markdown("### 1. Data Import")
    col_dl, col_up1, col_up2 = st.columns([1, 1, 1])
    
    with col_dl:
        st.info("Download template to see structure.")
        csv_data = generate_template_csv()
        st.download_button("📄 Download Template", csv_data, "circadian_template.csv", "text/csv")
        
    df1, df2 = None, None
    with col_up1:
        f1 = st.file_uploader("Upload Control", type=["csv", "xlsx"], key="file1")
        if f1: df1 = load_and_process_data(f1, "File 1")
    with col_up2:
        f2 = st.file_uploader("Upload Test", type=["csv", "xlsx"], key="file2")
        if f2: df2 = load_and_process_data(f2, "File 2")

    valid_dfs = [df for df in [df1, df2] if df is not None]

    if valid_dfs:
        df_combined = pd.concat(valid_dfs, ignore_index=True)
        # Ensure Analyte formatting
        df_combined['analyte'] = df_combined['analyte'].str.strip().str.title()
        
        st.success(f"Loaded {len(df_combined)} records.")
        st.markdown("---")
        st.markdown("### 2. Visualization")
        
        available_analytes = sorted(df_combined['analyte'].unique())
        default_ix = 0 
        if len(available_analytes) > 1 and "Unknown Analyte" in available_analytes:
             for i, a in enumerate(available_analytes):
                 if a != "Unknown Analyte": default_ix = i; break

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        analyte_filter = f_col1.selectbox("Analyte", available_analytes, index=default_ix)
        
        df_analyte = df_combined[df_combined['analyte'] == analyte_filter]
        
        current_unit = "units"
        if not df_analyte.empty:
            if 'unit' in df_analyte.columns and not df_analyte['unit'].isnull().all():
                current_unit = df_analyte['unit'].mode()[0]
        
        has_gender = df_analyte['has_gender'].any()
        has_age = df_analyte['has_age'].any()

        gender_opts = ["All"] + sorted(df_analyte['gender'].unique())
        age_opts = ["All"] + list(df_analyte['age_group'].unique().dropna())
        source_opts = sorted(df_analyte['source_file'].unique())
        display_opts = ["Combined"] + source_opts if len(source_opts) > 1 else source_opts
        
        gender_filter = f_col2.selectbox("Gender", gender_opts, disabled=not has_gender)
        age_filter = f_col3.selectbox("Age Group", age_opts, disabled=not has_age)
        display_mode = f_col4.radio("Display Source", display_opts, horizontal=True)

        df_plot = df_analyte.copy()
        if gender_filter != "All": df_plot = df_plot[df_plot['gender'] == gender_filter]
        if age_filter != "All": df_plot = df_plot[df_plot['age_group'] == age_filter]
        
        if display_mode == "Combined": plot_color, line_color = '#d9d9d9', '#1f1f1f' 
        elif display_mode == "File 1": df_plot = df_plot[df_plot['source_file'] == 'File 1']; plot_color, line_color = '#a0c4ff', '#003366'
        else: df_plot = df_plot[df_plot['source_file'] == 'File 2']; plot_color, line_color = '#ffadad', '#800000'

        fig_box, ax_box = plt.subplots(figsize=(12, 5), facecolor='white')
        ax_box.set_title(f"{analyte_filter} ({display_mode})")
        ax_box.set_xlabel("Time of Day (h)")
        ax_box.set_ylabel(f"Concentration ({current_unit})")
        ax_box.set_xlim(-0.5, 23.5); ax_box.set_xticks(range(24))
        
        if not df_plot.empty:
            box_data = [df_plot[df_plot['hour_int'] == h]['value'].values for h in range(24)]
            ax_box.boxplot(box_data, positions=range(24), patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=plot_color, alpha=0.8, edgecolor='#333'),
                           medianprops=dict(color=line_color, linewidth=2))
            
            medians = df_plot.groupby('hour_int')['value'].median().reindex(range(24))
            if medians.notna().any():
                ax_box.plot(range(24), medians, 'o-', color=line_color, lw=2, label='Median')
                ax_box.legend(frameon=True, facecolor='white')
        else:
            ax_box.text(12, 0, "No data for selection", ha='center')
            
        st.pyplot(fig_box, use_container_width=True)
        plt.close(fig_box)

        st.markdown("### 3. Estimated Model Parameters")
        results = []
        df_fit_base = df_analyte if display_mode == "Combined" else df_analyte[df_analyte['source_file'] == display_mode]
        
        genders = sorted(df_fit_base['gender'].unique())
        ages = sorted(df_fit_base['age_group'].unique().dropna()) if 'age_group' in df_fit_base.columns else []
        
        if genders and ages:
            rows, cols = len(ages), len(genders)
            fig_grid, axes = plt.subplots(rows, cols, figsize=(max(5, 5*cols), max(3, 3*rows)), 
                                          sharex=True, sharey=True, squeeze=False, facecolor='white')
            
            for i, ag in enumerate(ages):
                for j, gen in enumerate(genders):
                    ax = axes[i, j]
                    ax.set_title(f"{gen}, {ag}", fontsize=10)
                    sub = df_fit_base[(df_fit_base['gender'] == gen) & (df_fit_base['age_group'] == ag)]
                    
                    if not sub.empty:
                        M_fit, A_fit, t0_fit, r2_fit = get_fitted_parameters(sub)
                        meds = sub.groupby('hour_int')['value'].median()
                        
                        ax.plot(meds.index, meds.values, 'o', color='black', ms=4, alpha=0.6)
                        
                        if not np.isnan(M_fit):
                            t_lin = np.linspace(0, 24, 100)
                            y_lin = circadian(t_lin, M_fit, A_fit, t0_fit)
                            ax.plot(t_lin, y_lin, '-', color='red', alpha=0.8, label=f"Fit ($R^2={r2_fit:.2f}$)")
                            ax.legend(fontsize=7, loc='upper right')
                            
                            results.append({
                                "Gender": gen, "Age": ag, 
                                "Mean (M)": M_fit, "Amplitude (A)": A_fit, 
                                "Acrophase (t0)": t0_fit % 24, "R2": r2_fit
                            })
                    else:
                        ax.text(12, 0, "No Data", ha='center', fontsize=8, color='gray')
                    ax.set_xlim(0,24)
            st.pyplot(fig_grid, use_container_width=True)
            plt.close(fig_grid)
            
            if results:
                res_df = pd.DataFrame(results)
                st.dataframe(res_df.style.format({"Mean (M)": "{:.2f}", "Amplitude (A)": "{:.2f}", "Acrophase (t0)": "{:.2f}", "R2": "{:.3f}"}))
                
                csv_res = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Parameters (CSV)", csv_res, f"fit_results_{analyte_filter}.csv", "text/csv")
        else:
            st.info("Insufficient data grouping to display grid fits.")
    else:
        st.warning("Please upload a CSV or Excel file to begin analysis.")