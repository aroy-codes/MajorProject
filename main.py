import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import requests
import io
import os
import json
from dotenv import load_dotenv
from pandasql import sqldf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Data Analysis & Visualization Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file_buffer, file_type):
    """Load data from various file formats with caching"""
    df = None
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file_buffer)
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file_buffer)
        elif file_type == "json":
            df = pd.read_json(uploaded_file_buffer)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        # Clean column names
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_data
def fig_to_base64(_fig):
    """Convert matplotlib figure to base64 encoded PNG"""
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_bytes = buf.read()
    base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    return base64_encoded_image, "image/png"

def detect_potential_numeric(series):
    """Detect if a string column potentially contains numeric data"""
    patterns = {
        'currency': r'[\$\£\€\¥]?\s*\d+(?:,\d{3})*(?:\.\d{2})?',
        'percentage': r'\d+(?:\.\d+)?%',
        'scientific': r'\d+(?:\.\d+)?[eE][+-]?\d+',
        'fraction': r'\d+/\d+',
        'range': r'\d+\s*-\s*\d+',
        'mixed': r'[\d]+[\d\s\.,]*[\d]+',
        'with_units': r'\d+(?:\.\d+)?\s*(?:kg|m|km|ft|lbs?|g|ml|oz)',
        'ratio': r'\d+:\d+',
        'ordinal': r'\d+(?:st|nd|rd|th)',
    }
    
    sample_size = min(1000, len(series))
    sample = series.dropna().sample(n=sample_size, random_state=42) if len(series) > sample_size else series.dropna()
    
    matches = {pattern_name: 0 for pattern_name in patterns}
    total_matches = 0
    
    for value in sample:
        value = str(value).strip()
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, value, re.IGNORECASE):
                matches[pattern_name] += 1
                total_matches += 1
                break
    
    if len(sample) > 0:
        confidence = (total_matches / len(sample)) * 100
        detected_types = [k for k, v in matches.items() if v > 0]
    else:
        confidence = 0
        detected_types = []
    
    return confidence, detected_types
def find_potential_keys(df):
    """
    Identifies potential primary or unique key columns in a DataFrame.

    A column is considered a potential key if it meets these heuristics:
    1.  High cardinality (at least 95% unique values).
    2.  Low number of missing values (at least 95% non-null values).
    """
    potential_keys = []
    total_rows = len(df)
    
    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)
        non_null_count = df[column].count()

        # Calculate percentages
        unique_percentage = (unique_count / total_rows) * 100
        non_null_percentage = (non_null_count / total_rows) * 100

        # Check against heuristics
        if unique_percentage >= 95 and non_null_percentage >= 95:
            potential_keys.append(column)
            
    return potential_keys

def clean_and_convert_numeric(value):
    """Enhanced cleaning and conversion of string values to numeric"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    
    value = str(value).strip()
    
    special_cases = {
        'infinity': float('inf'),
        'inf': float('inf'),
        '-infinity': float('-inf'),
        '-inf': float('-inf'),
        'nan': np.nan,
        'na': np.nan,
        'n/a': np.nan,
        'null': np.nan,
        '': np.nan
    }
    
    if value.lower() in special_cases:
        return special_cases[value.lower()]
    
    replacements = {
        r'[^\d\.-]': '',
        r'(?<=\d),(?=\d{3})': '',
        r'(?<=\d)\s+(?=\d)': '',
    }
    
    cleaned = value
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    
    fraction_match = re.match(r'(\d+)/(\d+)', cleaned)
    if fraction_match:
        try:
            return float(fraction_match.group(1)) / float(fraction_match.group(2))
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    if '%' in value:
        cleaned = cleaned.replace('%', '')
        try:
            return float(cleaned) / 100
        except ValueError:
            return np.nan
    
    range_match = re.match(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', value)
    if range_match:
        try:
            return (float(range_match.group(1)) + float(range_match.group(2))) / 2
        except ValueError:
            return np.nan
    
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def convert_column_to_numeric(df, column_name):
    """Convert a column to numeric and return success rate"""
    converted_series = df[column_name].apply(clean_and_convert_numeric)
    non_na_count = converted_series.notna().sum()
    total_count = len(converted_series)
    success_rate = (non_na_count / total_count) * 100 if total_count > 0 else 0
    return converted_series, success_rate

def analyze_column_conversion(df, column_name):
    """Analyze column for numeric conversion potential"""
    original_values = df[column_name].head(5).tolist()
    converted_series, success_rate = convert_column_to_numeric(df, column_name)
    confidence, detected_types = detect_potential_numeric(df[column_name])
    
    results = {
        'original_sample': original_values,
        'converted_sample': converted_series.head(5).tolist() if converted_series is not None else [],
        'success_rate': success_rate,
        'confidence': confidence,
        'detected_types': detected_types,
        'converted_series': converted_series
    }
    
    return results

def apply_standardization(df, cols_to_standardize, standardization_method, min_max_feature_range=None):
    """Apply standardization and return the transformed dataframe"""
    processed_df = df.copy()
    
    for col in cols_to_standardize:
        if standardization_method == "Z-score Standardization":
            scaler = StandardScaler()
            processed_df[f"{col}_standardized"] = scaler.fit_transform(processed_df[[col]])
            
        elif standardization_method == "Min-Max Scaling":
            scaler = MinMaxScaler(feature_range=min_max_feature_range)
            processed_df[f"{col}_scaled"] = scaler.fit_transform(processed_df[[col]])
            
        elif standardization_method == "Robust Scaling":
            scaler = RobustScaler()
            processed_df[f"{col}_robust"] = scaler.fit_transform(processed_df[[col]])
    
    return processed_df

def generate_single_file_sql_query(question, df_columns):
    """Generate SQL query for single file using Gemini API"""
    prompt = f"""
    You are a highly skilled Senior SQL Analyst. Your task is to write a single, correct, and robust SQL query to answer a user's question about a dataset.

    The database is a pandas DataFrame named 'df' with the following columns: {df_columns}.
    
    This SQL query will be executed using the `pandasql.sqldf()` library on the **original, uncleaned, and unstandardized** DataFrame.

    **Core Rules for Query Generation:**
    1.  **Syntax:** Use standard SQL syntax.
    2.  **Column Names:** Ensure all column names in the query exactly match the names from the provided list. If a column name contains spaces or special characters, enclose it in backticks (` `).
    3.  **Output:** The final response MUST be ONLY the SQL query string. Do NOT include any explanations, code blocks, or extra text.

    **Reasoning Process (Internal Thought):**
    1.  **Understand Intent:** Analyze the user's question to determine the goal (e.g., aggregation, filtering, sorting, counting).
    2.  **Identify Columns:** Select the precise columns needed to answer the question.
    3.  **Select & Filter:** Formulate the `SELECT` and `WHERE` clauses. For text comparisons, use single quotes (').
    4.  **Aggregate & Group:** If the question involves summaries (e.g., 'total', 'average', 'count'), apply the appropriate aggregate function (`SUM`, `AVG`, `COUNT`) and include a `GROUP BY` clause for any non-aggregated columns.
    5.  **Order & Limit:** Use `ORDER BY` to sort results if the question implies ranking (e.g., 'highest', 'lowest', 'top 5').
    6.  **Return Single Query:** Construct the final, complete SQL query and nothing else.

    **Advanced Examples for Complex Requests:**
    * **Question:** "What is the average sales price for each product in the 'Electronics' category, sorted from highest to lowest?"
    * **SQL:** "SELECT Product, AVG(SalesPrice) AS AverageSales FROM df WHERE Category = 'Electronics' GROUP BY Product ORDER BY AverageSales DESC"

    * **Question:** "Count the number of orders placed in January 2023."
    * **SQL:** "SELECT COUNT(*) FROM df WHERE strftime('%Y-%m', `Order Date`) = '2023-01'"

    * **Question:** "Find all records for customers whose name starts with 'A'."
    * **SQL:** "SELECT * FROM df WHERE `Customer Name` LIKE 'A%'"

    **Final Task:**
    Based on all the rules and examples above, provide the single SQL query to answer the following question.

    Question: "{question}"

    Final SQL Query:
    """

    chat_history = []
    chat_history.append({ "role": "user", "parts": [{ "text": prompt }] })

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "sql_query": { "type": "STRING" }
                }
            }
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("text"):
            json_text = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(json_text)
            return parsed_json.get("sql_query")
        return None
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return None

def generate_multi_file_sql_query(question, multi_dfs_info):
    """Generate SQL query for multiple files using Gemini API"""
    df_info_str = ""
    for df_name, info in multi_dfs_info.items():
        df_info_str += f"- DataFrame '{df_name}' has columns: {info['columns']}. The user has identified '{info['primary_key']}' as a primary key for joining.\n"

    prompt = f"""
    You are given the following pandas DataFrames, which will be available as global variables (e.g., `df_file1`, `df_file2`) for pandasql.sqldf():
    {df_info_str}

    When generating the SQL query, consider the following:
    1.  **Entity Resolution:** Users may ask for information about entities (like products, customers, or individuals) by their **names** (e.g., "Product A", "John Doe"), even if the primary key for joining tables is **ID number** (e.g., "ProductID", "CustomerID").
        * If the question refers to an entity by its name (e.g., "Apple" for a product), first identify a likely 'name' column (e.g., 'ProductName', 'ItemName', 'CustomerName', 'PersonName', 'Title', 'Name') in the relevant DataFrame.
        * Use a `WHERE` clause to filter by this name column.
        * Then, use the corresponding ID column (e.g., 'ProductID', 'CustomerID') to perform `JOIN` operations with other tables that share this ID.
    2.  **Joining Data:** Infer common join conditions based on the identified primary keys. Assume `INNER JOIN` unless the question implies otherwise (e.g., "all customers even if they have no orders").
    3.  **Handling SELECT * with Joins:** If the user asks for "all records" or implies a `SELECT *` across joined tables, instead of `SELECT *`, list all columns from all involved tables. For columns that exist in multiple tables (e.g., a common ID used for joining), explicitly alias them (e.g., `t1.Product_ID AS t1_Product_ID`, `t2.Product_ID AS t2_Product_ID`) or select only one if the context implies it (e.g., `t1.Product_ID`). For other columns, use `t1.ColumnName` or `t2.ColumnName`.
    4.  **Aggregation:** If the question asks for summaries (e.g., "total", "average", "count", "max", "min"), use appropriate aggregate functions (`SUM`, `AVG`, `COUNT`, `MAX`, `MIN`) and include a `GROUP BY` clause on relevant non-aggregated columns.
    5.  **Selecting Columns:** Only select the columns explicitly requested or logically necessary to answer the question. Avoid `SELECT *` unless explicitly asked and handled as per point 3.
    6.  **Strict Syntax:** Ensure column names in the SQL query exactly match the column names provided in the DataFrame info. Use table aliases (e.g., `t1`, `t2`) for clarity in joins.
    7.  **Executable Query:** The query must be executable using `pandasql.sqldf()`.
    8.  **Output Format:** Do NOT include any explanation or extra text, just the SQL query.

    Example:
    Question: "What is the total sales for 'Product A' from df_products and df_sales, joining on ProductID?"
    SQL: "SELECT SUM(s.Amount) AS TotalSales FROM df_products AS p JOIN df_sales AS s ON p.ProductID = s.ProductID WHERE p.ProductName = 'Product A'"

    Question: "List the names of customers who placed orders after 2023 from df_customers and df_orders, joining on CustomerID."
    SQL: "SELECT DISTINCT c.CustomerName FROM df_customers AS c JOIN df_orders AS o ON c.CustomerID = o.CustomerID WHERE o.OrderDate > '2023-01-01'"

    Question: "Show all records for all product id from df_file1 and df_file2."
    SQL: "SELECT t1.*, t2.Product_Name, t2.Category, t2.Launch_Year FROM df_file1 AS t1 JOIN df_file2 AS t2 ON t1.Product_ID = t2.Product_ID"

    Question: "{question}"
    SQL:
    """

    chat_history = []
    chat_history.append({ "role": "user", "parts": [{ "text": prompt }] })

    payload = {
        "contents": chat_history,
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("text"):
            json_text = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(json_text)
            return parsed_json.get("sql_query")
        return None
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return None

def interpret_graph(question, image_b64, image_mime, x_axis_label, y_axis_label, graph_type_name, dataframe_full_str, original_plot_data_for_ai_str, applied_standardization_method):
    """Interpret a graph using Gemini API"""
    standardization_note = ""
    if applied_standardization_method != "Do nothing":
        standardization_note = f"Note: The numerical data used to generate the graph (and provided as 'Data used for Plotting') has undergone '{applied_standardization_method}'. However, the 'Original Unscaled Data for Plotted Columns' is also provided for reference. When answering questions about actual values, magnitudes, or real-world interpretations, please refer to the 'Original Unscaled Data for Plotted Columns' to provide answers in the original units and context of '{x_axis_label}' and '{y_axis_label}'."

    prompt = f"""
    Analyze the provided graph image and the accompanying data.
    The graph displays a '{graph_type_name}' with '{x_axis_label}' on the X-axis and '{y_axis_label}' on the Y-axis.

    {standardization_note}

    --- Data used for Plotting (potentially scaled) ---
    {dataframe_full_str}

    --- Original Unscaled Data for Plotted Columns ---
    {original_plot_data_for_ai_str}

    Answer the following question based on your understanding of this specific graph and the provided data.
    Focus on trends, patterns, key observations, and quantitative insights. Use the ORIGINAL UNSCALED DATA for accuracy where specific values are needed.

    Question: "{question}"
    """

    chat_history = []
    parts = [{"text": prompt}]
    chat_history.append({"role": "user", "parts": parts})

    payload = {
        "contents": chat_history,
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get('text'):
            return result["candidates"][0]["content"]["parts"][0]['text']
        return "No clear answer from the AI. Please try rephrasing your question or uploading a clearer image."
    except Exception as e:
        st.error(f"Error interpreting graph: {e}")
        return f"An error occurred while interpreting the graph: {e}"

def create_enhanced_dashboard(df):
    """Creates an interactive data quality dashboard"""
    with st.container(border=True):
        st.subheader("📊 Data Quality Dashboard")
        
        # 1. Key Metrics with Enhanced Styling
        col1, col2, col3, col4 = st.columns(4)
        
        # Total Records Metric
        with col1:
            total_records = df.shape[0]
            total_cols = df.shape[1]
            st.metric(
                label="📝 Total Records",
                value=f"{total_records:,}",
                delta=f"{total_cols} columns",
                help="Total number of rows and columns in dataset"
            )
        
        # Data Quality Metric
        with col2:
            missing_pct = (df.isnull().sum().sum() / (total_records * total_cols) * 100)
            completeness = 100 - missing_pct
            st.metric(
                label="✨ Data Quality",
                value=f"{completeness:.1f}%",
                delta=f"{missing_pct:.1f}% missing" if missing_pct > 0 else "Complete",
                delta_color="inverse" if missing_pct > 0 else "normal"
            )
        
        # Duplicate Analysis
        with col3:
            duplicates = df.duplicated().sum()
            unique_pct = ((total_records - duplicates) / total_records * 100)
            st.metric(
                label="🎯 Unique Records",
                value=f"{unique_pct:.1f}%",
                delta=f"{duplicates} duplicates" if duplicates > 0 else "No duplicates",
                delta_color="inverse" if duplicates > 0 else "normal"
            )
        
        # Memory Usage
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            st.metric(
                label="💾 Memory Usage",
                value=f"{memory_usage:.1f} MB",
                help="Total memory consumed by the dataset"
            )

        # 2. Data Type Analysis
        st.markdown("### 📊 Column Type Distribution")
        
        # Create data type distribution chart
        dtype_counts = df.dtypes.value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=dtype_counts.index.astype(str),
            values=dtype_counts.values,
            hole=0.4,
            textinfo='label+percent',
            marker_colors=px.colors.qualitative.Set3
        )])
        fig.update_layout(
            title="Column Data Types",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Missing Values Analysis
        st.markdown("### 🔍 Missing Values Analysis")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values('Missing Percentage', ascending=False)
        
        # Only show columns with missing values
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        
        if not missing_data.empty:
            fig = px.bar(
                missing_data,
                x='Column',
                y='Missing Percentage',
                text='Missing Count',
                color='Missing Percentage',
                color_continuous_scale='RdYlBu_r',
                title='Missing Values Distribution'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✨ Perfect! No missing values found in the dataset!")

        # 4. Numerical Columns Overview
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            st.markdown("### 📈 Numerical Columns Summary")
            stats_df = df[numerical_cols].describe()
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Metric'] + list(stats_df.columns),
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[stats_df.index] + [stats_df[col].round(2) for col in stats_df.columns],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                )
            )])
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # 5. Categorical Columns Analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.markdown("### 📑 Categorical Columns Analysis")
            selected_cat_col = st.selectbox(
                "Select a categorical column:",
                categorical_cols,
                key="cat_col_selector"
            )
            
            value_counts = df[selected_cat_col].value_counts()
            top_n = min(10, len(value_counts))
            
            fig = px.pie(
                values=value_counts.head(top_n),
                names=value_counts.head(top_n).index,
                title=f'Top {top_n} Values in {selected_cat_col}',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

        # 6. Download Report
        st.markdown("### 📥 Download Analysis Report")
        report = f"""
        Data Quality Analysis Report
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Dataset Overview:
        ----------------
        Total Records: {total_records:,}
        Total Columns: {total_cols}
        Memory Usage: {memory_usage:.2f} MB
        Data Completeness: {completeness:.1f}%
        Duplicate Records: {duplicates}
        
        Data Types Distribution:
        ----------------------
        {df.dtypes.value_counts().to_string()}
        
        Missing Values Summary:
        ---------------------
        {missing_data.to_string() if not missing_data.empty else 'No missing values found'}
        
        Numerical Columns Summary:
        ------------------------
        {df.describe().to_string() if len(numerical_cols) > 0 else 'No numerical columns found'}
        """
        
        st.download_button(
            label="📄 Download Detailed Report",
            data=report,
            file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# --- Main Application ---
def main():
    # Initialize session state
    if 'current_dataframe' not in st.session_state:
        st.session_state.current_dataframe = None
    if 'original_dataframe' not in st.session_state:
        st.session_state.original_dataframe = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'last_standardization_applied' not in st.session_state:
        st.session_state.last_standardization_applied = "Do nothing"
    if 'conversion_results' not in st.session_state:
        st.session_state.conversion_results = None
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Data Analysis Platform")
        st.markdown("---")
        
        # File upload in sidebar
        uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1]
            df = load_data(uploaded_file, file_extension)
            
            if df is not None:
                # Check if this is a new file upload
                if st.session_state.uploaded_file_name != uploaded_file.name:
                    st.session_state.original_dataframe = df
                    st.session_state.current_dataframe = df.copy()
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.last_standardization_applied = "Do nothing"
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio("Go to", 
                       ["Data Overview", "Data Cleaning", "Numeric Conversion", 
                        "SQL Query", "Visualization", "AI Analysis"])
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This application provides comprehensive data analysis and visualization capabilities powered by Gemini AI.")
    
    # Main content area
    st.title("Data Analysis & Visualization Platform")
    
    # Display based on selected page
    if page == "Data Overview":
        display_data_overview()
    elif page == "Data Cleaning":
        display_data_cleaning()
    elif page == "Numeric Conversion":
        display_numeric_conversion()
    elif page == "SQL Query":
        display_sql_query()
    elif page == "Visualization":
        display_visualization()
    elif page == "AI Analysis":
        display_ai_analysis()

def display_data_overview():
    """Display data overview and quality dashboard"""
    st.header("Data Overview")
    
    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
        st.write("### First 5 Rows")
        st.dataframe(df.head())

        # --- NEW CODE TO ADD ---
        st.write("### Potential Key Columns")
        potential_keys = find_potential_keys(df)
        if potential_keys:
            st.success(f"The following columns were detected as potential keys: **{', '.join(potential_keys)}**")
        else:
            st.info("No potential key columns were found based on the analysis.")
        st.markdown("---")
        # Create tabs for different overview sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Data Preview", "Column Info", "Basic Statistics", "Excel Filter"])
        
        with tab1:
            create_enhanced_dashboard(df)
        
        with tab2:
            st.subheader("Data Preview")
            num_rows = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(df.head(num_rows))
        
        with tab3:
            st.subheader("Column Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                # FIX: Convert dtypes to string to avoid Arrow serialization issues
                dtypes_df = pd.DataFrame(df.dtypes.astype(str), columns=['Data Type'])
                st.dataframe(dtypes_df)
            
            with col2:
                st.write("**Missing Values:**")
                missing_df = pd.DataFrame({
                    'Missing Values': df.isnull().sum(),
                    'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(missing_df)
        
        with tab4:
            st.subheader("Basic Statistics")
            numerical_cols = df.select_dtypes(include=['number']).columns
            
            if len(numerical_cols) > 0:
                st.dataframe(df[numerical_cols].describe())
            else:
                st.info("No numerical columns found for statistical summary.")
        
        # NEW: Add Excel-like filter tab
        with tab5:
            create_excel_filter_interface(df)
            
    else:
        st.info("Please upload a data file to get started.")

def display_data_cleaning():
    """Display data cleaning options"""
    st.header("Data Cleaning & Preprocessing")
    
    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values")
            missing_count = df.isnull().sum().sum()
            
            if missing_count > 0:
                st.warning(f"Found {missing_count} missing values")
                
                handling_method = st.selectbox(
                    "Handling method",
                    ["Do nothing", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
                )
                
                if st.button("Apply Missing Value Handling"):
                    cleaned_df = df.copy()
                    
                    if handling_method == "Drop rows":
                        cleaned_df = cleaned_df.dropna()
                        st.success("Rows with missing values dropped.")
                    elif handling_method == "Fill with mean":
                        for col in cleaned_df.select_dtypes(include=['number']).columns:
                            if cleaned_df[col].isnull().any():
                                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                        st.success("Missing values filled with mean.")
                    elif handling_method == "Fill with median":
                        for col in cleaned_df.select_dtypes(include=['number']).columns:
                            if cleaned_df[col].isnull().any():
                                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                        st.success("Missing values filled with median.")
                    elif handling_method == "Fill with mode":
                        for col in cleaned_df.select_dtypes(include=['object']).columns:
                            if cleaned_df[col].isnull().any():
                                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown"
                                cleaned_df[col].fillna(mode_val, inplace=True)
                        st.success("Missing values filled with mode.")
                    
                    st.session_state.current_dataframe = cleaned_df
                    st.rerun()
            else:
                st.success("No missing values found.")
        
        with col2:
            st.subheader("Duplicates")
            duplicate_count = df.duplicated().sum()
            
            if duplicate_count > 0:
                st.warning(f"Found {duplicate_count} duplicate rows")
                
                if st.button("Remove Duplicates"):
                    cleaned_df = df.drop_duplicates()
                    st.session_state.current_dataframe = cleaned_df
                    st.success("Duplicates removed.")
                    st.rerun()
            else:
                st.success("No duplicate rows found.")
        
        st.subheader("Standardization")
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numerical_cols:
            standardization_method = st.selectbox(
                "Standardization method",
                ["Do nothing", "Z-score Standardization", "Min-Max Scaling", "Robust Scaling"]
            )
            
            cols_to_standardize = st.multiselect(
                "Select columns to standardize",
                numerical_cols
            )
            
            if st.button("Apply Standardization") and cols_to_standardize:
                processed_df = apply_standardization(
                    df, 
                    cols_to_standardize, 
                    standardization_method,
                    min_max_feature_range=(0, 1)
                )
                st.session_state.current_dataframe = processed_df
                st.session_state.last_standardization_applied = standardization_method
                st.success("Standardization applied.")
                st.rerun()
        else:
            st.info("No numerical columns found for standardization.")
        
        # Reset button
        if st.button("Reset to Original Data"):
            st.session_state.current_dataframe = st.session_state.original_dataframe.copy()
            st.session_state.last_standardization_applied = "Do nothing"
            st.success("Data reset to original state.")
            st.rerun()
    else:
        st.info("Please upload a data file to use cleaning features.")

def display_numeric_conversion():
    """Display numeric conversion tools"""
    st.header("Numeric Data Detection & Conversion")
    
    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe
        
        # Get non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        if non_numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                column_to_analyze = st.selectbox(
                    "Select column to analyze:",
                    non_numeric_cols,
                    key="column_to_analyze"
                )
                
                if st.button("Analyze Column", key="analyze_column"):
                    st.session_state.conversion_results = analyze_column_conversion(df, column_to_analyze)
                    
            with col2:
                if st.session_state.conversion_results is not None:
                    results = st.session_state.conversion_results
                    st.write("#### Analysis Results")
                    st.metric(
                        "Numeric Confidence Score",
                        f"{results['confidence']:.1f}%",
                        help="Confidence that this column contains convertible numeric data"
                    )
                    # Add this code in the display_numeric_conversion() function, right after displaying the confidence score

                    if results['confidence'] > 60:
                        st.success("✅ High confidence detected! Consider applying the conversion to use this data in numerical analyses.")
                    elif results['confidence'] > 30:
                        st.warning("⚠️ Moderate confidence detected. You might want to review the conversion results before applying.")
                    else:
                        st.info("ℹ️ Low confidence detected. The conversion may not be reliable for this column.")
                    if results['detected_types']:
                        st.write("Detected number types:")
                        for dtype in results['detected_types']:
                            st.write(f"- {dtype.title()}")
                    
                    comparison_df = pd.DataFrame({
                        'Original': results['original_sample'],
                        'Converted': results['converted_sample']
                    })
                    st.write("Sample Conversion Results:")
                    st.dataframe(comparison_df)
                    
                    if st.button("Apply Conversion", key="apply_conversion"):
                        df[f"{column_to_analyze}_numeric"] = results['converted_series']
                        st.session_state.current_dataframe = df
                        st.success(f"Created new column '{column_to_analyze}_numeric' with converted values!")
                        st.rerun()
        else:
            st.info("No non-numeric columns found in the dataset.")
    else:
        st.info("Please upload a data file to use conversion features.")

def display_sql_query():
    """Display SQL query interface"""
    st.header("SQL Query Generator")
    
    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe
        
        st.info("SQL queries will run on the **original, uncleaned, and unstandardized** dataset.")
        
        sql_question = st.text_area(
            "Enter your question about the data:",
            placeholder="e.g., 'What is the maximum GDP?', 'Show me all data for 2015'",
            height=100
        )
        
        if st.button("Generate & Run SQL Query"):
            if not sql_question.strip():
                st.error("Please enter a question for the SQL query.")
            else:
                with st.spinner("Generating SQL query..."):
                    generated_sql = generate_single_file_sql_query(
                        sql_question, 
                        st.session_state.original_dataframe.columns.tolist()
                    )
                
                if generated_sql:
                    st.subheader("Generated SQL Query:")
                    st.code(generated_sql, language="sql")
                    
                    with st.spinner("Executing SQL query on original data..."):
                        try:
                            sql_result = sqldf(generated_sql, {'df': st.session_state.original_dataframe})
                            st.subheader("Query Result:")
                            if not sql_result.empty:
                                st.dataframe(sql_result)
                                
                                # Download results
                                csv = sql_result.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("The query returned no results.")
                        except Exception as e:
                            st.error(f"Error executing SQL query: {e}")
                else:
                    st.warning("Could not generate a valid SQL query. Please try rephrasing your question.")
    else:
        st.info("Please upload a data file to use SQL query features.")

def display_visualization():
    """Display data visualization tools"""
    st.header("Data Visualization")
    
    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap"]
            )
        
        with col2:
            available_cols = df.columns.tolist()
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
                x_col = st.selectbox("X-axis", available_cols)
                y_col = st.selectbox("Y-axis", numerical_cols)
            elif chart_type == "Histogram":
                x_col = st.selectbox("Column", numerical_cols)
                y_col = None
            elif chart_type == "Box Plot":
                x_col = st.selectbox("Category Column", available_cols)
                y_col = st.selectbox("Value Column", numerical_cols)
            elif chart_type == "Heatmap":
                x_col = None
                y_col = None
                selected_cols = st.multiselect("Select columns for correlation", numerical_cols)
        
        if st.button("Generate Chart"):
            try:
                fig = None
                
                if chart_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                elif chart_type == "Heatmap" and selected_cols:
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                   title="Correlation Heatmap")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store figure data for AI analysis
                    st.session_state.last_fig = fig
                    st.session_state.last_chart_type = chart_type
                    st.session_state.last_x_col = x_col
                    st.session_state.last_y_col = y_col
            except Exception as e:
                st.error(f"Error generating chart: {e}")
    else:
        st.info("Please upload a data file to create visualizations.")

def display_ai_analysis():
    """Display AI analysis features"""
    st.header("AI-Powered Analysis")
    
    if st.session_state.current_dataframe is None:
        st.info("Please upload a data file to use AI analysis features.")
        return
    
    if 'last_fig' not in st.session_state:
        st.info("Please generate a visualization first to use AI analysis.")
        return
    
    st.subheader("Graph Interpretation")
    
    question = st.text_area(
        "Ask a question about your visualization:",
        placeholder="e.g., 'What trends do you observe?', 'What is the correlation between these variables?'",
        height=100
    )
    
    if st.button("Analyze with AI"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("AI is analyzing your visualization..."):
                # Convert figure to base64
                img_bytes = st.session_state.last_fig.to_image(format="png")
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                
                # Prepare data for AI
                df_sample = st.session_state.current_dataframe.head(100)
                
                # Convert DataFrame to string representation
                dataframe_full_str = df_sample.to_string()
                original_plot_data_for_ai_str = st.session_state.original_dataframe.head(100).to_string()
                
                # Call AI interpretation
                ai_response = interpret_graph(
                    question,
                    base64_image,
                    "image/png",
                    st.session_state.last_x_col or "N/A",
                    st.session_state.last_y_col or "N/A",
                    st.session_state.last_chart_type,
                    dataframe_full_str,
                    original_plot_data_for_ai_str,
                    st.session_state.last_standardization_applied
                )
            
            st.subheader("AI Analysis:")
            st.write(ai_response)
def create_excel_filter_interface(df):
    """Create an Excel-like filter interface for data exploration"""
    st.subheader("🔍 Excel-style Data Filtering")

    # Create two columns for filter controls and data display
    filter_col, data_col = st.columns([1, 3])

    with filter_col:
        st.markdown("### Filter Controls")

        # Store filters in session state
        if 'filters' not in st.session_state:
            st.session_state.filters = {}

        # Create filter for each column
        for column in df.columns:
            st.markdown(f"**{column}**")
            
            # --- IMPORTANT FIX for pyarrow.lib.ArrowInvalid ---
            # Ensure the column is of a type that can be filtered.
            # Convert any mixed-type columns (like company_ids) to string to prevent errors.
            df[column] = df[column].astype(str) if 'object' in str(df[column].dtype) else df[column]

            # Get unique values for this column
            unique_vals = df[column].dropna().unique()

            # Handle different data types appropriately
            if df[column].dtype in ['object', 'category']:
                # For categorical data, use multiselect
                selected_vals = st.multiselect(
                    f"Select values for {column}:",
                    options=unique_vals,
                    default=st.session_state.filters.get(column, []),
                    key=f"filter_{column}"
                )
                st.session_state.filters[column] = selected_vals
            else:
                # For numerical data, use range slider
                if len(unique_vals) > 0:
                    min_val = float(df[column].min()) if df[column].dtype in ['int64', 'float64'] else 0
                    max_val = float(df[column].max()) if df[column].dtype in ['int64', 'float64'] else 1

                    # Store range in session state
                    if column not in st.session_state.filters:
                        st.session_state.filters[column] = [min_val, max_val]

                    # --- FIX for StreamlitAPIException and SyntaxError ---
                    # Check if a slider is possible (min and max values are different)
                    if min_val < max_val:
                        range_vals = st.slider(
                            f"Range for {column}:",
                            min_val, max_val,
                            value=st.session_state.filters[column],
                            key=f"filter_{column}"
                        )
                        # This line must be inside the `if` block where range_vals is defined.
                        st.session_state.filters[column] = range_vals
                    else:
                        st.info(f"All values in '{column}' are '{min_val}'. No range to filter.")
                        # This line must be in the `else` block to handle the case with no slider.
                        st.session_state.filters[column] = [min_val, max_val] # Keep the fixed value

        # Add action buttons
        apply_filters = st.button("Apply Filters", type="primary")
        clear_filters = st.button("Clear All Filters")

        if clear_filters:
            st.session_state.filters = {}
            st.rerun()

    with data_col:
        st.markdown("### Filtered Data")

        # Apply filters to dataframe
        filtered_df = df.copy()

        for column, filter_val in st.session_state.filters.items():
            if column in filtered_df.columns:
                if isinstance(filter_val, list) and len(filter_val) > 0:
                    if filtered_df[column].dtype in ['object', 'category']:
                        # Categorical filter
                        filtered_df = filtered_df[filtered_df[column].isin(filter_val)]
                    else:
                        # Numerical range filter
                        if len(filter_val) == 2:
                            filtered_df = filtered_df[
                                (filtered_df[column] >= filter_val[0]) &
                                (filtered_df[column] <= filter_val[1])
                            ]

        # Display filtered data
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )

        # Show filter summary
        st.info(f"Showing {len(filtered_df)} of {len(df)} records ({len(filtered_df)/len(df)*100:.1f}%)")

        # Download filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
# Run the application
if __name__ == "__main__":
    main()