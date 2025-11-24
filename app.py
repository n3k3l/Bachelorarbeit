import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import curve_fit
import io

# ───────────────────────────────────────────
# BASIC SETTINGS & LAYOUT
# ───────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Diurnal Fluctuations Analysis")

# DEFINITIONEN FÜR FARBEN
BG_COLOR = "#f0f2f6"     # Hellgrauer Hintergrund
TEXT_COLOR = "#000000"   # Schwarzer Text für die Seite
WIDGET_TEXT = "#ffffff"  # Weißer Text für Buttons/Inputs

# CSS FÜR STYLING
st.markdown(f"""
    <style>
    /* 1. Haupt-Hintergrund der App */
    .stApp {{
        background-color: {BG_COLOR};
    }}
    
    /* 2. Genereller Text auf der Seite (Überschriften, Labels) -> SCHWARZ */
    h1, h2, h3, h4, h5, h6, p, li, span, div {{
        color: {TEXT_COLOR};
    }}
    
    /* Labels ÜBER den Inputs (z.B. "Analyte", "Upload File") -> SCHWARZ & FETT */
    .stSelectbox label, .stNumberInput label, .stSlider label, 
    .stTimeInput label, .stRadio label, .stFileUploader label, 
    .stCheckbox label {{
        color: {TEXT_COLOR} !important;
        font-weight: 600;
    }}

    /* 3. WIDGETS ANPASSEN (Dunkler Hintergrund -> Weißer Text) */
    
    /* DOWNLOAD BUTTON & Normale Buttons */
    .stDownloadButton button, .stButton button {{
        color: {WIDGET_TEXT} !important;
    }}
    
    /* FILE UPLOADER (Drag & Drop Zone) */
    /* Zwingt allen Text innerhalb der Dropzone auf Weiß */
    [data-testid="stFileUploader"] section, 
    [data-testid="stFileUploader"] div, 
    [data-testid="stFileUploader"] span {{
        color: {WIDGET_TEXT} !important;
    }}
    
    /* Das "Limit 200MB" Kleingedruckte etwas grauer, aber lesbar */
    [data-testid="stFileUploader"] small {{
        color: #e0e0e0 !important;
    }}

    /* Der "Browse files" Button innerhalb des Uploaders */
    [data-testid="stFileUploader"] button {{
        color: {WIDGET_TEXT} !important;
    }}

    /* INPUT FELDER (Zahlen, Zeit) - Text innen weiß, falls Box dunkel ist */
    .stNumberInput input, .stTimeInput input {{
        color: {WIDGET_TEXT} !important;
    }}
    
    /* SELECTBOX Text des ausgewählten Elements */
    .stSelectbox div[data-baseweb="select"] span {{
        color: {WIDGET_TEXT} !important;
    }}
    
    /* RADIO BUTTONS Text */
    .stRadio div[role='radiogroup'] label div {{
        color: {TEXT_COLOR} !important;
    }}

    /* TABS Styling */
    button[data-baseweb="tab"] {{
        color: {TEXT_COLOR} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #d10000 !important; /* Aktiver Tab rot */
    }}

    /* Warnungen/Infos lesbar machen (Hintergrund ist meist hell in st.info) */
    .stAlert div {{
        color: {TEXT_COLOR} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; margin-bottom: 25px; color: black;'>Visualization & Analysis of Diurnal Fluctuations</h2>", unsafe_allow_html=True)

# ───────────────────────────────────────────
# 1) CONSTANTS & DEFAULTS
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
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours % 24:02d}:{minutes:02d}"

def chronomap_delta(A, M, t0, steps=100):
    t_vals = np.linspace(0, 24, steps)
    T1, T2 = np.meshgrid(t_vals, t_vals)
    Y1 = circadian(T1, M, A, t0)
    Y2 = circadian(T2, M, A, t0)
    return T1, T2, np.abs(Y1 - Y2)

def generate_template_csv():
    data = {
        'ANALYT': ['Cholesterin', 'Glucose', 'Cortisol'],
        'VALUE': [167, 95, 14.5],
        'DIM': ['mg/dl', 'mg/dl', 'ug/dl'],
        'TIME': ['27.11.2013 16:06', '27.11.2013 08:30', '27.11.2013 20:00'],
        'SEX': ['M', 'F', 'M'],
        'AGE': [47, 32, 55]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# Helper um Plots an den grauen Hintergrund anzupassen
def style_plot(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    # Axenbeschriftung schwarz machen
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.title.set_color('black')
    # Ränder anpassen
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

# ───────────────────────────────────────────
# 3) STREAMLIT TABS
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
        
        unit = st.radio("Unit for Glucose", ["mg/dL", "mmol/L"], horizontal=True) if analyte == "Glucose" else "mg/dL"
        
        # Default Params
        p = default_params.get((analyte, gender), default_params[("Other", "Male")])
        t0_lit, A_lit, MU_perc_lit, M_literature = p["t0"], p["A"], p["MU"], p["M"]
        
        st.markdown("**1. Adjust Model to Measured Value**")
        personalize_mode = st.checkbox("Adjust Mean (M) to measured value at t₁", value=True)
        M = M_literature
        if personalize_mode:
            val_default = M_literature / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else float(M_literature)
            step_val = 0.1 if unit == 'mmol/L' else 1.0
            y_measured_t1_display = st.number_input(f"Measured value at t₁ ({format_time_string(t1_hour)}) in {unit}", value=val_default, step=step_val, format="%.1f")
            y_measured_t1_mgdl = y_measured_t1_display * GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_measured_t1_display
            M = y_measured_t1_mgdl - A_lit * np.cos(2 * np.pi * (t1_hour - t0_lit) / 24)
            st.info(f"Adjusted Mean (M): **{M:.2f} mg/dL**")
            
        st.markdown("**2. Manually Adjust Parameters**")
        editor_mode = st.checkbox("Enable Editor Mode")
        A, t0, MU_perc = A_lit, t0_lit, MU_perc_lit
        if editor_mode:
            A = st.slider("Amplitude A", 1.0, 50.0, float(A_lit), 0.5)
            M = st.slider("Mean M (mg/dL)", 50.0, 250.0, float(M), 1.0, disabled=personalize_mode)
            t0 = st.slider("Acrophase t₀ (h)", 0.0, 24.0, t0_lit, 0.1)
            MU_perc = st.slider("Measurement Uncertainty MU %", 1.0, 50.0, float(MU_perc_lit), 0.5)
        
        mu_abs = M * MU_perc / 100

    with right:
        t_arr = np.linspace(0, 24, 500)
        y_arr_mgdl = circadian(t_arr, M, A, t0)
        y_arr_display = y_arr_mgdl / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_arr_mgdl
        mu_abs_display = mu_abs / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else mu_abs
        
        # --- Plot 1: Time Series ---
        fig_sin, ax_sin = plt.subplots(figsize=(10, 3.5))
        style_plot(fig_sin, ax_sin)
        ax_sin.set_facecolor("white") # Keep graph area white for contrast
        
        ax_sin.set_title(f"Simulated Diurnal Fluctuation for {analyte}", fontsize=12, color='black')
        ax_sin.plot(t_arr, y_arr_display, color="cornflowerblue", label="Expected Profile", lw=3)
        
        # t1 Line (Black Solid)
        ax_sin.axvline(t1_hour, color="black", ls="-", lw=2, label=f"t₁ = {format_time_string(t1_hour)}")
        
        ax_sin.fill_between(t_arr, y_arr_display - mu_abs_display, y_arr_display + mu_abs_display, color="lightblue", alpha=0.3, label=f"Tolerance (±{MU_perc}%)")
        
        if personalize_mode:
            y_t1_display = circadian(t1_hour, M, A, t0) / (GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else 1.0)
            ax_sin.plot(t1_hour, y_t1_display, 'ko', markersize=7, zorder=5) # black dot
        
        ax_sin.set_xlabel("Time of Day (h)"); ax_sin.set_ylabel(f"Concentration ({unit})")
        ax_sin.grid(True, alpha=0.3)
        ax_sin.set_xlim(0, 24)
        ax_sin.legend(fontsize='small', loc='upper right')
        st.pyplot(fig_sin)
        
        # --- Slider for t2 ---
        delta_h = st.slider("Time Difference t₂ ↔ t₁ (h)", 0.0, 24.0, 6.0, 0.25, key="delta_h_slider")
        t2_hour = (t1_hour + delta_h) % 24
        
        y_t1 = circadian(t1_hour, M, A, t0)
        y_t2 = circadian(t2_hour, M, A, t0)
        
        # --- Comparison Plots ---
        c_left, c_right = st.columns(2, gap="medium")
        
        with c_left:
            st.markdown("##### Chronomap")
            T1, T2, delta_values = chronomap_delta(A, M, t0)
            
            fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
            style_plot(fig_cm, ax_cm)
            
            pcm = ax_cm.pcolormesh(T2, T1, delta_values, cmap="coolwarm", shading='gouraud')
            
            # Contour
            ax_cm.contour(T2, T1, delta_values, levels=[mu_abs], colors='black', linestyles='dotted', linewidths=1)
            
            # t1 Line (Black Solid)
            ax_cm.axhline(t1_hour, color='black', ls='-', lw=2, label=f"t₁")
            # t2 Line (Orange Dashed)
            ax_cm.axvline(t2_hour, color='orange', ls='--', lw=2, label=f"t₂")
            
            # Intersection dot
            ax_cm.plot(t2_hour, t1_hour, 'ko', markersize=8, mfc='white', markeredgewidth=2)
            
            ax_cm.set_xlabel("Timepoint t₂ (h)"); ax_cm.set_ylabel("Timepoint t₁ (h)")
            ax_cm.set_xlim(0, 24); ax_cm.set_ylim(0, 24); ax_cm.set_aspect('equal')
            ax_cm.legend(fontsize='x-small', loc='lower right', framealpha=0.8)
            st.pyplot(fig_cm)
            
        with c_right:
            st.markdown("##### 24h Clock")
            
            if np.abs(y_t1 - y_t2) <= mu_abs: 
                st.success(f"**Comparable:** Δ = {abs(y_t1 - y_t2):.2f} (≤ {mu_abs:.2f})")
            else: 
                st.error(f"**Not comparable:** Δ = {abs(y_t1 - y_t2):.2f} (> {mu_abs:.2f})")
            
            # Normalization
            min_val, max_val = M - A, M + A
            norm = lambda y: 0.1 + 0.9 * np.clip((y - min_val) / (max_val - min_val), 0, 1)
            r1, r2 = norm(y_t1), norm(y_t2)
            
            theta1 = (t1_hour / 24.0) * 2 * np.pi
            theta2 = (t2_hour / 24.0) * 2 * np.pi
            
            fig_clk, ax_clk = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(5, 5))
            fig_clk.patch.set_facecolor(BG_COLOR)
            ax_clk.set_facecolor(BG_COLOR)
            ax_clk.spines['polar'].set_edgecolor('black')
            ax_clk.tick_params(axis='x', colors='black')
            
            ax_clk.set_theta_offset(np.pi/2); ax_clk.set_theta_direction(-1)
            ax_clk.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
            ax_clk.set_xticklabels([f"{h*2}" for h in range(12)], fontsize=9, color='black')
            ax_clk.set_yticklabels([]); ax_clk.grid(alpha=0.4, color='gray'); ax_clk.set_rlim(0, 1.1)
            
            y_t1_disp = y_t1 / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_t1
            y_t2_disp = y_t2 / GLUCOSE_CONVERSION_FACTOR if unit == 'mmol/L' else y_t2
            
            # t1 (Black Solid)
            ax_clk.plot([theta1, theta1], [0, r1], color='black', lw=2.5, ls='-', label=f"t₁ ({y_t1_disp:.1f})")
            # t2 (Orange Dashed)
            ax_clk.plot([theta2, theta2], [0, r2], color='orange', lw=2.5, ls='--', label=f"t₂ ({y_t2_disp:.1f})")
            
            ax_clk.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize='small', facecolor=BG_COLOR, edgecolor='gray')
            st.pyplot(fig_clk)

# ==========================================
# TAB 2: DATA ANALYSIS
# ==========================================
with tab2:
    st.markdown("### Analysis of Patient Data")
    
    # 1. Download Template Button
    st.markdown("Don't have a file? Download the data template below:")
    st.download_button(
        label="Download Data Template (CSV)",
        data=generate_template_csv(),
        file_name="ExampleData_Template.csv",
        mime="text/csv"
    )
    
    st.divider()

    # 2. Uploaders
    c_up1, c_up2 = st.columns(2)
    f1 = c_up1.file_uploader("Upload File 1", type=["csv", "xlsx"])
    f2 = c_up2.file_uploader("Upload File 2 (Optional)", type=["csv", "xlsx"])

    @st.cache_data
    def load_data(file, name):
        if file is None: return None
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            df.columns = df.columns.str.strip().str.upper()
            
            if not {'VALUE', 'TIME'}.issubset(df.columns):
                st.error(f"File {name} is missing required columns: VALUE, TIME")
                return None

            # ANALYT
            if 'ANALYT' not in df.columns:
                df['ANALYT'] = 'Unknown'
            # DIM
            if 'DIM' not in df.columns:
                df['DIM'] = ''
            # SEX
            if 'SEX' not in df.columns:
                df['SEX'] = 'All' 
            else:
                df['SEX'] = df['SEX'].astype(str).str.upper().map({
                    'M': 'Male', 'F': 'Female', 'D': 'Divers', 'MALE': 'Male', 'FEMALE': 'Female'
                }).fillna('Other')
            # AGE
            if 'AGE' not in df.columns:
                df['AGE_GROUP'] = 'All'
            else:
                df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
                df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 30, 50, 120], labels=["< 30", "30-50", "> 50"], right=False)
                df['AGE_GROUP'] = df['AGE_GROUP'].cat.add_categories("Unknown").fillna("Unknown")

            # TIME
            df['TIMESTAMP'] = pd.to_datetime(df['TIME'], format='%d.%m.%Y %H:%M', errors='coerce')
            mask_nat = df['TIMESTAMP'].isna()
            if mask_nat.any():
                df.loc[mask_nat, 'TIMESTAMP'] = pd.to_datetime(df.loc[mask_nat, 'TIME'], errors='coerce')
            
            df.dropna(subset=['TIMESTAMP', 'VALUE'], inplace=True)
            df['HOUR'] = df['TIMESTAMP'].dt.hour + df['TIMESTAMP'].dt.minute/60.0
            df['SOURCE'] = name
            
            return df
        except Exception as e:
            st.error(f"Error reading {name}: {e}")
            return None

    df1 = load_data(f1, "File 1")
    df2 = load_data(f2, "File 2")
    
    valid_dfs = [d for d in [df1, df2] if d is not None]
    
    if valid_dfs:
        df_all = pd.concat(valid_dfs, ignore_index=True)
        
        # --- FILTERS ---
        st.markdown("#### Data Filters")
        
        # Analyte Filter
        analytes = sorted(df_all['ANALYT'].unique())
        sel_analyt = st.selectbox("Select Analyte", analytes)
        df_sub = df_all[df_all['ANALYT'] == sel_analyt].copy()
        
        # Group Filters
        c_f1, c_f2 = st.columns(2)
        sex_opts = ["All"] + sorted([x for x in df_sub['SEX'].unique() if x != "All"])
        age_opts = ["All"] + sorted([str(x) for x in df_sub['AGE_GROUP'].unique() if str(x) != "All"])
        
        sel_sex = c_f1.selectbox("Filter Gender", sex_opts)
        sel_age = c_f2.selectbox("Filter Age Group", age_opts)
        
        if sel_sex != "All":
            df_sub = df_sub[df_sub['SEX'] == sel_sex]
        if sel_age != "All":
            df_sub = df_sub[df_sub['AGE_GROUP'].astype(str) == sel_age]
            
        if df_sub.empty:
            st.warning("No data matches filters.")
        else:
            unit_display = df_sub['DIM'].iloc[0] if 'DIM' in df_sub.columns else ""
            st.markdown(f"**Data Overview** ({len(df_sub)} samples) - Unit: {unit_display}")
            
            # --- BOXPLOT ---
            fig_box, ax_box = plt.subplots(figsize=(12, 5))
            style_plot(fig_box, ax_box)
            ax_box.set_facecolor("white")
            
            colors = {'File 1': 'cornflowerblue', 'File 2': 'sandybrown'}
            df_sub['HOUR_INT'] = df_sub['TIMESTAMP'].dt.hour
            sources = df_sub['SOURCE'].unique()
            
            for src in sources:
                dat = df_sub[df_sub['SOURCE'] == src]
                hrs = np.arange(24)
                vals = [dat[dat['HOUR_INT'] == h]['VALUE'].values for h in hrs]
                
                if any(len(v) > 0 for v in vals):
                    bp = ax_box.boxplot(vals, positions=hrs, widths=0.6, patch_artist=True, 
                                        boxprops=dict(facecolor=colors.get(src, 'gray'), alpha=0.6))
                    medians = [np.median(v) if len(v) > 0 else np.nan for v in vals]
                    ax_box.plot(hrs, medians, color=colors.get(src, 'gray'), lw=2, label=src)
            
            ax_box.set_xlabel("Time of Day (h)")
            ax_box.set_ylabel(f"Value {unit_display}")
            ax_box.set_xticks(range(0, 24, 2))
            ax_box.legend(facecolor='white', edgecolor='black')
            ax_box.grid(True, axis='y', alpha=0.3)
            
            st.pyplot(fig_box)
            
            # --- MODEL FITTING ---
            st.markdown("#### Circadian Fit")
            
            def get_fit(dframe):
                if len(dframe) < 10: return None
                grp = dframe.groupby('HOUR_INT')['VALUE'].median()
                x_data, y_data = grp.index.values, grp.values
                if len(x_data) < 4: return None
                
                guess_M, guess_A = np.mean(y_data), (np.max(y_data)-np.min(y_data))/2
                guess_t0 = x_data[np.argmax(y_data)]
                
                try:
                    popt, _ = curve_fit(circadian, x_data, y_data, p0=[guess_M, guess_A, guess_t0], maxfev=5000)
                    return popt
                except:
                    return None

            cols = st.columns(len(sources))
            for i, src in enumerate(sources):
                d_src = df_sub[df_sub['SOURCE'] == src]
                params = get_fit(d_src)
                
                with cols[i]:
                    st.caption(f"Fit for {src}")
                    if params is not None:
                        M_fit, A_fit, t0_fit = params
                        st.write(f"**M:** {M_fit:.1f} | **A:** {A_fit:.1f} | **t₀:** {t0_fit % 24:.1f}h")
                        
                        fig_fit, ax_fit = plt.subplots(figsize=(4, 3))
                        style_plot(fig_fit, ax_fit)
                        ax_fit.set_facecolor("white")
                        
                        ax_fit.scatter(d_src['HOUR'], d_src['VALUE'], alpha=0.3, s=10, color='gray')
                        t_lin = np.linspace(0, 24, 100)
                        ax_fit.plot(t_lin, circadian(t_lin, *params), color='red', lw=2)
                        ax_fit.set_title(f"{sel_analyt} ({src})", color='black')
                        st.pyplot(fig_fit)
                    else:
                        st.write("Not enough data for curve fitting.")

    else:
        st.info("Please upload a CSV file to begin analysis.")
        st.markdown("""
        **Expected Format:**
        * `ANALYT` (String)
        * `VALUE` (Numeric)
        * `DIM` (String, e.g. mg/dl)
        * `TIME` (Format: DD.MM.YYYY HH:MM)
        * Optional: `SEX` (M/F), `AGE` (Numeric)
        """)