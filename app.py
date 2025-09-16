
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io
import base64

# Page configuration
st.set_page_config(
    page_title="IPO Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Configure matplotlib for streamlit
plt.switch_backend('Agg')
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

def load_sample_data():
    """Load sample data for demonstration when real data is not available"""
    # This creates sample IPO data for demonstration
    np.random.seed(42)
    n_samples = 100

    sample_data = {
        'IPO_Name': [f'Company_{i}' for i in range(n_samples)],
        'Issue_Size(crores)': np.random.lognormal(4, 1, n_samples),
        'Listing Gain': np.random.normal(15, 25, n_samples),
        'Opening Price': np.random.uniform(100, 1000, n_samples),
        'Closing Price': np.random.uniform(100, 1200, n_samples),
        'Market Cap(crores)': np.random.lognormal(6, 1.5, n_samples),
        'Date': pd.date_range('2010-01-01', periods=n_samples, freq='30D')
    }

    df = pd.DataFrame(sample_data)
    df['Closing Price'] = df['Opening Price'] * (1 + df['Listing Gain'] / 100)
    return df

def create_distribution_plot(df):
    """Create distribution plot for Issue Size"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Issue_Size(crores)'].dropna(), kde=True, ax=ax)
    ax.set_title('Distribution of Issue Size (crores)')
    ax.set_xlabel('Issue Size (crores)')
    ax.set_ylabel('Frequency')
    return fig

def create_top_ipos_plot(df):
    """Create top 10 IPOs plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = df['IPO_Name'].value_counts().iloc[:10]
    sns.barplot(x=top_10.values, y=top_10.index, ax=ax)
    ax.set_title('Top 10 IPOs by Count')
    ax.set_xlabel('Count')
    ax.set_ylabel('IPO Name')
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap of Numeric Features')
        return fig
    return None

def train_model(df):
    """Train linear regression model"""
    numeric_df = df.select_dtypes(include=[np.number])
    target = 'Listing Gain'

    if target in numeric_df.columns and numeric_df.shape[1] > 1:
        model_df = numeric_df.dropna()
        X = model_df.drop(columns=[target])
        y = model_df[target]

        if len(X) > 5:  # Need minimum data for train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            return model, score, X.columns.tolist()
    return None, None, None

def main():
    st.title("📊 IPO Data Analysis Dashboard")
    st.markdown("### Analyze Indian IPO data from 2010-2025")

    # Sidebar
    st.sidebar.header("Dashboard Options")

    # File upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload your IPO data file",
        type=['xlsx', 'csv'],
        help="Upload an Excel or CSV file containing IPO data"
    )

    # Load data
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = load_sample_data()
            st.info("📝 Using sample data instead")
    else:
        df = load_sample_data()
        st.info("📝 Using sample data. Upload your own file to analyze real data!")

    # Data preprocessing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Remove unnamed columns
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("📊 Dataset Statistics")
        st.metric("Total Records", len(df))
        st.metric("Total Columns", len(df.columns))

        if 'Issue_Size(crores)' in df.columns:
            avg_issue_size = df['Issue_Size(crores)'].mean()
            st.metric("Avg Issue Size (cr)", f"₹{avg_issue_size:.2f}")

    # Missing values analysis
    st.subheader("🔍 Data Quality Analysis")
    numeric_df = df.select_dtypes(include=[np.number])
    missing_vals = numeric_df.isnull().sum()
    missing_data = missing_vals[missing_vals > 0]

    if len(missing_data) > 0:
        st.warning("Missing values found in numeric columns:")
        st.dataframe(missing_data.to_frame(columns=['Missing Values']))
    else:
        st.success("✅ No missing values in numeric columns!")

    # Visualizations
    st.header("📈 Data Visualizations")

    viz_tabs = st.tabs(["Distribution Analysis", "Correlation Analysis", "Top IPOs"])

    with viz_tabs[0]:
        if 'Issue_Size(crores)' in df.columns:
            fig = create_distribution_plot(df)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Issue_Size(crores) column not found")

    with viz_tabs[1]:
        corr_fig = create_correlation_heatmap(df)
        if corr_fig:
            st.pyplot(corr_fig)
            plt.close()
        else:
            st.warning("Not enough numeric columns for correlation analysis")

    with viz_tabs[2]:
        if 'IPO_Name' in df.columns:
            fig = create_top_ipos_plot(df)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("IPO_Name column not found")

    # Machine Learning Model
    st.header("🤖 Predictive Modeling")

    with st.expander("Linear Regression Model for Listing Gain Prediction", expanded=True):
        model, score, features = train_model(df)

        if model is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.success(f"✅ Model trained successfully!")
                st.metric("R² Score", f"{score:.4f}")
                st.write("**Features used:**")
                for feature in features:
                    st.write(f"• {feature}")

            with col2:
                st.info("**Model Performance Interpretation:**")
                if score > 0.8:
                    st.write("🟢 Excellent model performance")
                elif score > 0.6:
                    st.write("🟡 Good model performance")
                elif score > 0.4:
                    st.write("🟠 Moderate model performance")
                else:
                    st.write("🔴 Model needs improvement")
        else:
            st.error("❌ Could not train model. Check data quality and ensure 'Listing Gain' column exists.")

    # Download processed data
    st.header("💾 Export Data")

    # Convert dataframe to CSV
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ipo_analysis_results.csv">Download Processed Data as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Built with Streamlit** | "
        "📊 Data Analysis Dashboard for IPO Trends | "
        "🚀 Deploy anywhere with one click!"
    )

if __name__ == "__main__":
    main()
