import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="🌍 Climate Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    data_path = "data/"

    df_co2 = pd.read_csv(f"{data_path}co2.csv")
    df_co2 = df_co2[df_co2['Entity'] == 'World'].copy()
    df_co2 = df_co2.rename(columns={'Year': 'year', 'Entity': 'region', 'Annual CO₂ emissions': 'co2_emissions'})
    df_co2 = df_co2[['year', 'region', 'co2_emissions']]

    df_pop = pd.read_csv(f"{data_path}population.csv")
    df_pop = df_pop[df_pop['Entity'] == 'World'].copy()
    df_pop = df_pop.rename(columns={'Year': 'year', 'Entity': 'region', 'Percent': 'population_growth_rate_pct'})
    df_pop = df_pop[['year', 'region', 'population_growth_rate_pct']]

    df_sea = pd.read_csv(f"{data_path}sea_level.csv")
    df_sea = df_sea[df_sea['Entity'] == 'World'].copy()
    df_sea = df_sea.rename(columns={'Year': 'year', 'Entity': 'region', 'Global sea level (Average)': 'sea_level_change_mm'})
    df_sea = df_sea[['year', 'region', 'sea_level_change_mm']]

    df_precip = pd.read_csv(f"{data_path}precipitation.csv")
    df_precip = df_precip[df_precip['Entity'] == 'World'].copy()
    df_precip = df_precip.rename(columns={'Year': 'year', 'Entity': 'region', 'Annual precipitation': 'precipitation_mm'})
    df_precip = df_precip[['year', 'region', 'precipitation_mm']]

    df_renew = pd.read_csv(f"{data_path}renewables.csv")
    df_renew = df_renew[df_renew['Entity'] == 'World'].copy()
    df_renew = df_renew.rename(columns={'Year': 'year', 'Entity': 'region', 'Renewables (% equivalent primary energy)': 'renewables_pct'})
    df_renew = df_renew[['year', 'region', 'renewables_pct']]

    df_forest = pd.read_csv(f"{data_path}forest_area.csv")
    df_forest = df_forest[df_forest['Country Name'] == 'World'].copy()
    df_forest = df_forest.rename(columns={'Year': 'year', 'Country Name': 'region', 'Percent': 'forest_area_pct'})
    df_forest = df_forest[['year', 'region', 'forest_area_pct']]

    dfs = [df_co2, df_pop, df_sea, df_precip, df_renew, df_forest]
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = pd.merge(df_final, df, on=['year', 'region'], how='outer')

    numeric_cols = ['co2_emissions', 'population_growth_rate_pct', 'sea_level_change_mm', 'precipitation_mm', 'renewables_pct', 'forest_area_pct']
    for col in numeric_cols:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    df_final = df_final.sort_values('year').reset_index(drop=True)
    return df_final

try:
    df_main = load_data()
    climate_vars = ['co2_emissions', 'population_growth_rate_pct', 'sea_level_change_mm', 'precipitation_mm', 'renewables_pct', 'forest_area_pct']
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

def navigate_to(page_name):
    st.session_state.navigation = page_name
    st.rerun()

def page_home():
    st.markdown("""
    <style>
    .hero-card {
        background-color: #0e1117;
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hero-card h3 {
        color: #ffffff !important;
        margin-top: 0;
    }
    .hero-list {
        list-style-type: none;
        padding-left: 0;
    }
    .hero-list li {
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .hero-list li::before {
        content: "🔹"; 
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.title("Planet's Pulse")
        st.markdown("""
        ### 🌍 Global Climate Dashboard
        
        Explore how **CO₂ emissions**, **sea level**, **forest area**, **renewables**, and **population** are changing over time – and how they are related.
        
        Use this dashboard to visualize trends, discover relationships, see simple forecasts, and learn how data science supports climate action.
        """)
    
    with col2:
        st.markdown("""
        <div class="hero-card">
            <h3>📊 Dashboard Features</h3>
            <ul class="hero-list">
                <li>Time-series trends for key climate indicators</li>
                <li>Correlations between CO₂, sea level, forests & renewables</li>
                <li>Machine Learning forecasts for future metrics</li>
                <li>Data-driven insights & climate action resources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.subheader("📈 Trends")
            st.write("See how climate indicators evolve over time with moving averages.")
            if st.button("Open Trends →", key="btn_trends"):
                navigate_to("Trends")

    with c2:
        with st.container(border=True):
            st.subheader("🔗 Relationships")
            st.write("Explore how CO₂ relates to sea level, forest loss, and renewables.")
            if st.button("Open Relationships →", key="btn_rel"):
                navigate_to("Relationships")

    with c3:
        with st.container(border=True):
            st.subheader("🔮 Predictions")
            st.write("View simple forecasts for emissions and sea level.")
            if st.button("Open Predictions →", key="btn_pred"):
                navigate_to("Predictions")

    with c4:
        with st.container(border=True):
            st.subheader("🌱 Insights")
            st.write("Summarized findings and how data can support climate action.")
            if st.button("Open Insights →", key="btn_act"):
                navigate_to("Insights")

def page_trends():
    st.title("📈 Historical Trends")
    st.write("Analyze the historical progression of individual climate indicators over time.")

    selected_var = st.selectbox("Select a Climate Variable:", climate_vars)
    
    if df_main[selected_var].notna().sum() == 0:
        st.warning(f"No data available for {selected_var}.")
        return

    plot_df = df_main[['year', selected_var]].dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['year'], y=plot_df[selected_var], mode='lines', name=selected_var, line=dict(width=2, color='#1f77b4')))

    show_rolling = st.checkbox("Show 5-Year Rolling Average", value=False)
    if show_rolling:
        rolling_series = plot_df[selected_var].rolling(window=5).mean()
        fig.add_trace(go.Scatter(x=plot_df['year'], y=rolling_series, mode='lines', name='5-Year Moving Avg', line=dict(width=2, color='#ff7f0e', dash='dash')))

    fig.update_layout(title=f"{selected_var} over Time", xaxis_title="Year", yaxis_title=selected_var, hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    min_year = plot_df['year'].min()
    max_year = plot_df['year'].max()
    start_val = plot_df.loc[plot_df['year'] == min_year, selected_var].values[0]
    end_val = plot_df.loc[plot_df['year'] == max_year, selected_var].values[0]
    
    delta = end_val - start_val
    delta_pct = (delta / start_val) * 100 if start_val != 0 else 0
    direction = "increased" if delta > 0 else "decreased"

    st.info(f"Insight: From {min_year} to {max_year}, {selected_var} {direction} by {abs(delta):.2f} ({abs(delta_pct):.1f}%).")
    
    with st.expander("View Raw Data"):
        st.dataframe(plot_df)

def page_relationships():
    st.title("🔗 Variable Relationships")
    st.write("Visualize how two different climate variables co-evolve over time.")

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Select Left Y-Axis Variable:", climate_vars, index=0)
    with col2:
        var2 = st.selectbox("Select Right Y-Axis Variable:", climate_vars, index=2)

    corr_df = df_main[['year', var1, var2]].dropna()

    if corr_df.empty:
        st.warning("Insufficient overlapping data.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=corr_df['year'], y=corr_df[var1], name=var1, line=dict(color='#1f77b4')), secondary_y=False)
    fig.add_trace(go.Scatter(x=corr_df['year'], y=corr_df[var2], name=var2, line=dict(color='#ff7f0e')), secondary_y=True)

    fig.update_layout(title=f"Relationship: {var1} vs {var2}", xaxis_title="Year", hovermode="x unified", template="plotly_white")
    fig.update_yaxes(title_text=var1, secondary_y=False)
    fig.update_yaxes(title_text=var2, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    correlation = corr_df[var1].corr(corr_df[var2])
    abs_corr = abs(correlation)
    strength = "strong" if abs_corr > 0.7 else "moderate" if abs_corr > 0.3 else "weak"
    direction = "positive" if correlation > 0 else "negative"

    st.success(f"**Correlation Analysis:** The trends of `{var1}` and `{var2}` show a **{strength} {direction} correlation** (r ≈ {correlation:.2f}).")

def page_predictions():
    st.title("🔮 Predictive Modeling")
    st.write("Train a machine learning model to predict a climate variable based on others.")

    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        target_var = st.selectbox("Select Target Variable:", climate_vars, index=0)
    feature_options = [v for v in climate_vars if v != target_var]
    with col_input2:
        predictors = st.multiselect("Select Independent Variables:", feature_options, default=[feature_options[0]])

    col_model, col_horizon = st.columns(2)
    with col_model:
        model_type = st.selectbox("Select Model Type:", ["Random Forest", "Linear Regression"])
    with col_horizon:
        horizon = st.slider("Forecast Horizon (Years)", 5, 50, 20)

    if not predictors:
        st.warning("Please select at least one independent variable.")
        return

    model_cols = ['year', target_var] + predictors
    model_df = df_main[model_cols].dropna()

    if len(model_df) < 20:
        st.error("Not enough clean data points to train a model.")
        return

    X = model_df[['year'] + predictors]
    y = model_df[target_var]
    split_idx = int(len(model_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
        
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    c1, c2, c3 = st.columns(3)
    c1.metric("Model R² Score", f"{r2:.2f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")

    last_year = int(model_df['year'].max())
    future_years = np.arange(last_year + 1, last_year + horizon + 1)
    future_data = {'year': future_years}
    
    for feature in predictors:
        lr = LinearRegression()
        X_hist = model_df[['year']].values
        y_hist = model_df[feature].values
        lr.fit(X_hist, y_hist)
        future_data[feature] = lr.predict(future_years.reshape(-1, 1))

    X_future = pd.DataFrame(future_data)
    X_future = X_future[X_train.columns]
    y_future_pred = model.predict(X_future)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=model_df['year'], y=model_df[target_var], mode='lines', name='Historical Data', line=dict(color='gray')))
    fig_pred.add_trace(go.Scatter(x=X_test['year'], y=y_pred_test, mode='lines', name='Validation Prediction', line=dict(color='blue', dash='dot')))
    fig_pred.add_trace(go.Scatter(x=future_years, y=y_future_pred, mode='lines', name='Future Forecast', line=dict(color='red', width=3)))

    fig_pred.update_layout(title=f"Forecast: {target_var} (Next {horizon} Years)", xaxis_title="Year", yaxis_title=target_var, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_pred, use_container_width=True)

    if model_type == "Random Forest":
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.bar_chart(feat_df.set_index('Feature'))
    else:
        st.subheader("Model Coefficients")
        coeffs = model.coef_
        feat_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coeffs}).sort_values(by='Coefficient', ascending=False)
        st.bar_chart(feat_df.set_index('Feature'))

def page_action():
    st.title("🌱 Actions & Insights")
    st.markdown("### 🌍 From Data to Action")
    st.write("Data science helps us understand the urgency of the climate crisis. Here are the key takeaways from our analysis and steps you can take.")
    
    st.info("💡 **Key Insight:** Our analysis shows a **strong positive correlation** between rising CO₂ emissions and global sea level rise. This suggests that cutting emissions is critical to mitigating future coastal risks.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 What the Data Says")
        st.markdown("""
        - **Exponential Growth:** CO₂ emissions have not just increased; they have accelerated over the last century.
        - **Sea Level Response:** Global sea levels are rising consistently, threatening coastal regions.
        - **Renewables are Vital:** While renewable energy adoption is growing, it still lags behind the rate needed to offset fossil fuel dependence completely.
        - **Forests Matter:** Deforestation data shows a decline in natural carbon sinks, exacerbating the greenhouse effect.
        """)
    with col2:
        st.subheader("🚀 Potential Actions")
        st.markdown("""
        - **Advocate:** Support policies that prioritize renewable energy infrastructure and carbon taxes.
        - **Protect Nature:** Support reforestation and conservation organizations to restore natural carbon sinks.
        - **Reduce Waste:** Minimize single-use plastics and food waste, which contribute to landfill methane emissions.
        - **Energy Efficiency:** Switch to LED lighting, energy-efficient appliances, and consider EV transportation.
        """)

    st.markdown("---")
    
    st.subheader("👣 Calculate Your Carbon Footprint")
    st.write("Understanding your own impact is the first step toward change. Use this external calculator to see where you stand.")
    
    st.link_button("Go to Carbon Footprint Calculator ↗", "https://footprint.conservation.org/")

def main():
    if 'navigation' not in st.session_state:
        st.session_state.navigation = 'Home'

    pages = {
        "Home": page_home,
        "Trends": page_trends,
        "Relationships": page_relationships,
        "Predictions": page_predictions,
        "Insights": page_action
    }
    
    if st.session_state.navigation != 'Home':
        nav_cols = st.columns(len(pages))
        for idx, page_name in enumerate(pages.keys()):
            with nav_cols[idx]:
                if page_name == st.session_state.navigation:
                    if st.button(page_name, key=f"nav_btn_{page_name}", type="primary", use_container_width=True):
                        st.session_state.navigation = page_name
                        st.rerun()
                else:
                    if st.button(page_name, key=f"nav_btn_{page_name}", use_container_width=True):
                        st.session_state.navigation = page_name
                        st.rerun()
        st.markdown("---")
    
    pages[st.session_state.navigation]()

if __name__ == "__main__":

    main()
