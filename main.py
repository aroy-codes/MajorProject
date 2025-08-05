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
# New imports for VIF calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant # Required for VIF calculation

# Load environment variables from .env file
load_dotenv()

# --- Streamlit Page Configuration ---
# This should be the ONLY place st.set_page_config() is called
st.set_page_config(
    page_title="Data & Graph Interpreter with Gemini AI",
    page_icon="📈",
    layout="wide" # Using wide layout for more space
)

# --- Constants ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Helper Function to Load Data (with caching) ---
@st.cache_data
def load_data(uploaded_file_buffer, file_type):
    """
    Loads a data file (CSV, Excel, JSON) from a buffer into a pandas DataFrame.
    Uses st.cache_data to cache the DataFrame.
    """
    df = None
    if file_type == "csv":
        df = pd.read_csv(uploaded_file_buffer)
    elif file_type == "xlsx":
        df = pd.read_excel(uploaded_file_buffer)
    elif file_type == "json":
        df = pd.read_json(uploaded_file_buffer)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None
    
    # Strip whitespace from column names for consistency
    if df is not None:
        df.columns = df.columns.str.strip()
    return df

# --- Helper Function to Encode Matplotlib Figure to Base64 (with caching) ---
@st.cache_data
def fig_to_base64(_fig):
    """
    Converts a matplotlib figure to a base64 encoded PNG string.
    Uses st.cache_data to cache the image bytes.
    """
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_bytes = buf.read()
    base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    return base64_encoded_image, "image/png"

# --- Function to call LLM for Single-File SQL generation ---
def generate_single_file_sql_query(question, df_columns):
    """
    Uses Gemini API to generate a SQL query based on a natural language question
    and DataFrame columns for a single DataFrame. It explicitly notes that the SQL will run on the ORIGINAL dataset.
    """
    prompt = f"""
    Given a pandas DataFrame named 'df' with the following columns: {df_columns}.
    This SQL query will be executed on the **original, uncleaned, and unstandardized** version of the DataFrame.
    Generate a SQL query that answers the following question.
    The query should be executable using pandasql.sqldf().
    Do NOT include any explanation or extra text, just the SQL query.

    Examples for statistical measures and relationships:
    Question: "What is the average GDP?"
    SQL: "SELECT AVG(GDP_In_Billion_USD) FROM df"

    Question: "Show me the year with the highest percentage growth."
    SQL: "SELECT Year FROM df ORDER BY Percentage_Growth DESC LIMIT 1"

    Question: "What is the total sales for each product category?"
    SQL: "SELECT Category, SUM(Sales) AS TotalSales FROM df GROUP BY Category ORDER BY TotalSales DESC"

    Question: "Find the minimum and maximum values for 'Temperature'."
    SQL: "SELECT MIN(Temperature) AS MinTemperature, MAX(Temperature) AS MaxTemperature FROM df"

    Question: "Count the number of unique customers."
    SQL: "SELECT COUNT(DISTINCT CustomerID) FROM df"

    Question: "{question}"
    SQL:
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

# --- Function to call LLM for Multi-File SQL generation ---
def generate_multi_file_sql_query(question, multi_dfs_info):
    """
    Uses Gemini API to generate a SQL query based on a natural language question
    and information about multiple DataFrames (names, columns, primary keys).
    This function is enhanced to handle queries by name even if the join key is an ID,
    and to prevent duplicate column names in SELECT * scenarios.
    """
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

# --- Function to call LLM for Graph Interpretation ---
def interpret_graph(question, image_b64, image_mime, x_axis_label, y_axis_label, graph_type_name, dataframe_full_str, original_plot_data_for_ai_str, applied_standardization_method):
    """
    Uses Gemini API to interpret a graph image based on a natural language question,
    providing additional context about the graph's axes, type, the full underlying data
    (both plotted and original unscaled), and any applied standardization method.
    """
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

# --- Main Application Logic ---
def main():
    st.title("📈 Data & Graph Interpreter with Gemini AI")
    st.markdown("Upload your data, clean it, ask questions, generate custom graphs, and get AI insights!")

    # --- Debugging Info (Sidebar) ---
    st.sidebar.header("Debugging Info")
    st.sidebar.write(f"Current working directory: `{os.getcwd()}`")
    st.sidebar.write(f"Is '.env' file found? `{os.path.exists('.env')}`")
    st.sidebar.write(f"Value of API_KEY (first 5 chars): `{API_KEY[:5] if API_KEY else 'None'}`")
    st.sidebar.markdown("---")

    # Check if API key is loaded
    if not API_KEY:
        st.error("Google API Key (GOOGLE_API_KEY) not found. Please ensure you have a '.env' file in the same directory as this script, with a line like: `GOOGLE_API_KEY='YOUR_API_KEY_HERE'`")
        st.stop()

    # --- Section 1: Data Upload & Overview ---
    st.header("1. Data Upload & Overview")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Choose a data file...", type=["csv", "xlsx", "json"])

        df = None
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1]
            try:
                df = load_data(uploaded_file, file_extension)
                st.success(f"{file_extension.upper()} file uploaded successfully!")
                st.subheader("Original Data Preview:")
                st.dataframe(df.head())

                if 'uploaded_file_name' not in st.session_state or st.session_state['uploaded_file_name'] != uploaded_file.name:
                    st.session_state['original_dataframe'] = df # Store original untouched dataframe
                    st.session_state['current_dataframe'] = df.copy() # Store a working copy
                    st.session_state['uploaded_file_name'] = uploaded_file.name
                    st.session_state['last_standardization_applied'] = "Do nothing"
                    # Clear graph-related session state variables on new file upload
                    for key in ['generated_image_b64', 'generated_image_mime', 'x_axis_select', 'y_axis_select']:
                        if key in st.session_state:
                            del st.session_state[key]
                    # Clear outlier info on new file upload
                    if 'detected_outliers_info' in st.session_state:
                        del st.session_state['detected_outliers_info']
                    # Clear dependent variable selection on new file upload
                    if 'dependent_variable' in st.session_state:
                        del st.session_state['dependent_variable']
                    st.rerun()
                else:
                    df = st.session_state['current_dataframe']

            except Exception as e:
                st.error(f"Error reading {file_extension.upper()} file: {e}. Please ensure it's a valid {file_extension.upper()} format.")
                df = None
                # Clear all relevant session state on error
                for key in ['original_dataframe', 'current_dataframe', 'uploaded_file_name', 'generated_image_b64', 'generated_image_mime', 'x_axis_select', 'y_axis_select', 'last_standardization_applied', 'detected_outliers_info', 'dependent_variable']:
                    if key in st.session_state:
                        del st.session_state[key]
        elif 'current_dataframe' in st.session_state:
            df = st.session_state['current_dataframe']

    st.markdown("---") # Separator

    if df is not None:
        total_missing_values = df.isnull().sum().sum()

        # --- Column Data Types Section ---
        st.subheader("Column Data Types")
        with st.expander("Click to view column data types"):
            if df is not None and not df.empty:
                # Convert dtypes series to a DataFrame for better handling by Streamlit/PyArrow
                dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                # Ensure the 'Data Type' column is explicitly string type
                dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)
                st.dataframe(dtypes_df)
            else:
                st.info("Upload data to see column types.")
        st.markdown("---")
        # --- End Column Data Types ---

        # --- Data Overview & Statistics Section (within Section 1) ---
        st.subheader("Data Overview & Statistics")
        with st.container(border=True):
            st.markdown("Explore basic statistics for your dataset.")
            show_all_stats = st.checkbox("Show Statistics for All Columns", key="show_all_stats_checkbox")
            
            columns_for_stats = []
            if not show_all_stats:
                columns_for_stats = st.multiselect(
                    "Select columns for detailed statistics:",
                    df.columns.tolist(),
                    key="stats_columns_multiselect"
                )
            else:
                columns_for_stats = df.columns.tolist()

            if st.button("Generate Statistics", key="generate_stats_button"):
                if not columns_for_stats:
                    st.warning("Please select at least one column or check 'Show Statistics for All Columns'.")
                else:
                    st.write("### Calculated Statistics:")
                    # Filter for numerical columns among selected
                    numerical_cols_for_stats = df[columns_for_stats].select_dtypes(include=['number']).columns.tolist()
                    
                    if numerical_cols_for_stats:
                        st.markdown("**Descriptive Statistics (Numerical Columns):**")
                        st.dataframe(df[numerical_cols_for_stats].describe())
                    else:
                        st.info("No numerical columns selected for descriptive statistics.")

                    # Display mode for all selected columns (numerical and non-numerical)
                    st.markdown("**Mode(s) for Selected Columns:**")
                    modes_df = df[columns_for_stats].mode().transpose()
                    
                    if not modes_df.empty:
                        # Filter out rows where all modes are NaN (i.e., no distinct mode found)
                        modes_df_cleaned = modes_df.dropna(how='all')
                        if not modes_df_cleaned.empty:
                            st.dataframe(modes_df_cleaned)
                        else:
                            st.info("No distinct mode found for the selected columns.")
                    else:
                        st.info("No modes found for the selected columns.")
        st.markdown("---")

        # --- Add New Data Row Section (within Section 1) ---
        st.subheader("Add New Data Row")
        st.markdown("Manually add a new record to your current dataset.")

        with st.expander("Click to expand and add a new row"):
            new_row_data = {}
            # Iterate through columns to create input fields
            for col in df.columns:
                # Infer type for input widget based on column dtype
                if pd.api.types.is_numeric_dtype(df[col]):
                    new_row_data[col] = st.number_input(f"Enter value for '{col}' (Numerical)", key=f"add_row_{col}_num")
                elif pd.api.types.is_bool_dtype(df[col]):
                    new_row_data[col] = st.checkbox(f"Check for '{col}' (Boolean)", key=f"add_row_{col}_bool")
                else: # Default to text input for object/string columns
                    new_row_data[col] = st.text_input(f"Enter value for '{col}' (Text)", key=f"add_row_{col}_text")

            if st.button("Add Row to Dataset", key="add_new_row_button"):
                # Convert the new row data to a DataFrame row and append
                new_row_df = pd.DataFrame([new_row_data])
                
                # Ensure column order matches before concatenating
                new_row_df = new_row_df[df.columns]

                # Use pd.concat to append the new row
                st.session_state['current_dataframe'] = pd.concat([st.session_state['current_dataframe'], new_row_df], ignore_index=True)
                st.success("New row added successfully!")
                st.rerun() # Rerun to update the dataframe preview and other sections
        st.markdown("---")
        
        # --- Section 2: Data Cleaning & Preprocessing ---
        st.header("2. Data Cleaning & Preprocessing")
        with st.container(border=True):
            st.markdown("Handle missing values, duplicate rows, outliers, and standardize numerical data.")

            # Missing Value Handling
            st.subheader("Missing Value Handling")
            if total_missing_values > 0:
                st.warning(f"Missing values detected in your data (Total: {total_missing_values}):")
                st.dataframe(df.isnull().sum()[df.isnull().sum() > 0].rename("Missing Count"))

                cleaning_method = st.radio(
                    "How do you want to handle missing values?",
                    ("Do nothing", "Drop rows with any missing values",
                     "Fill numerical with mean", "Fill numerical with median",
                     "Fill categorical with mode"),
                    key="cleaning_method"
                )

                if st.button("Apply Missing Value Cleaning", key="apply_cleaning_button"):
                    cleaned_df = df.copy()

                    if cleaning_method == "Drop rows with any missing values":
                        cleaned_df.dropna(inplace=True)
                        st.success("Rows with missing values dropped.")
                    elif cleaning_method == "Fill numerical with mean":
                        for col in cleaned_df.select_dtypes(include=['number']).columns:
                            if cleaned_df[col].isnull().any():
                                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                        st.success("Missing numerical values filled with mean.")
                    elif cleaning_method == "Fill numerical with median":
                        for col in cleaned_df.select_dtypes(include=['number']).columns:
                            if cleaned_df[col].isnull().any():
                                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                        st.success("Missing numerical values filled with median.")
                    elif cleaning_method == "Fill categorical with mode":
                        for col in cleaned_df.select_dtypes(include=['object', 'category']).columns:
                            if cleaned_df[col].isnull().any():
                                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                                if mode_val is not None:
                                    cleaned_df[col].fillna(mode_val, inplace=True)
                        st.success("Missing categorical values filled with mode.")
                    else:
                        st.info("No cleaning method applied for missing values.")

                    st.session_state['current_dataframe'] = cleaned_df
                    st.rerun()
            else:
                st.info("No missing values detected in your dataset.")
            st.markdown("---")


            # Duplicate Row Handling
            st.subheader("Duplicate Row Handling")
            if df.duplicated().any():
                st.warning(f"Duplicate rows detected: {df.duplicated().sum()} rows.")
                if st.button("Remove Duplicate Rows", key="remove_duplicates_button"):
                    processed_df = df.drop_duplicates().copy()
                    st.session_state['current_dataframe'] = processed_df
                    st.success(f"Removed {df.duplicated().sum()} duplicate rows.")
                    st.rerun()
            else:
                st.info("No duplicate rows detected.")
            st.markdown("---")

            # --- Rename Columns ---
            st.subheader("Rename Columns")
            st.markdown("Change the names of your dataset columns.")
            with st.expander("Configure Column Renaming"):
                col_to_rename = st.selectbox(
                    "Select column to rename:",
                    df.columns.tolist(),
                    key="col_to_rename_select"
                )
                new_col_name = st.text_input(
                    f"Enter new name for '{col_to_rename}':",
                    value=col_to_rename, # Default to current name
                    key="new_col_name_input"
                )
                if st.button("Apply Rename", key="apply_rename_button"):
                    if new_col_name and new_col_name != col_to_rename:
                        if new_col_name in df.columns:
                            st.error(f"Column '{new_col_name}' already exists. Please choose a unique name.")
                        else:
                            df.rename(columns={col_to_rename: new_col_name}, inplace=True)
                            st.session_state['current_dataframe'] = df.copy() # Update session state
                            st.success(f"Column '{col_to_rename}' renamed to '{new_col_name}'.")
                            st.rerun()
                    else:
                        st.info("No change detected or new name is empty.")
            st.markdown("---")

            # --- Outlier Detection and Handling ---
            st.subheader("Outlier Detection and Handling")
            st.markdown("Identify and manage extreme values in your numerical data.")
            
            # This expander is for configuring the outlier handling
            with st.expander("Configure Outlier Handling"):
                outlier_method = st.selectbox(
                    "Select Outlier Detection Method:",
                    ("Do nothing", "IQR Method", "Z-score Method"),
                    key="outlier_method"
                )

                numerical_cols_for_outliers = df.select_dtypes(include=['number']).columns.tolist()
                selected_outlier_cols = []
                if numerical_cols_for_outliers:
                    selected_outlier_cols = st.multiselect(
                        "Select numerical columns for outlier detection:",
                        numerical_cols_for_outliers,
                        key="selected_outlier_cols"
                    )
                else:
                    st.info("No numerical columns available for outlier detection.")

                outlier_action = st.radio(
                    "Select Action for Outliers:",
                    ("Remove rows with outliers", "Cap outliers (replace with boundary values)"),
                    key="outlier_action"
                )

                z_score_threshold = 3.0
                if outlier_method == "Z-score Method":
                    z_score_threshold = st.slider(
                        "Z-score Threshold:",
                        min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                        key="z_score_threshold"
                    )
                    st.info(f"Values with a Z-score greater than {z_score_threshold} will be considered outliers.")

                if st.button("Apply Outlier Handling", key="apply_outlier_button"):
                    if outlier_method != "Do nothing" and not selected_outlier_cols:
                        st.warning("Please select at least one numerical column for outlier handling.")
                    elif outlier_method != "Do nothing" and selected_outlier_cols:
                        processed_df = df.copy()
                        outliers_found_overall = False
                        
                        # Store detected outliers for display
                        detected_outliers_info = {}

                        for col in selected_outlier_cols:
                            lower_bound = None
                            upper_bound = None

                            if outlier_method == "IQR Method":
                                Q1 = processed_df[col].quantile(0.25)
                                Q3 = processed_df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                            elif outlier_method == "Z-score Method":
                                mean = processed_df[col].mean()
                                std = processed_df[col].std()
                                # Avoid division by zero if std is 0 (all values are the same)
                                if std == 0:
                                    st.info(f"Column '{col}' has no variance, no outliers detected by Z-score method.")
                                    continue # Skip this column if no variance
                                lower_bound = mean - z_score_threshold * std
                                upper_bound = mean + z_score_threshold * std
                            
                            # Identify outliers for the current column
                            col_outliers_mask = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
                            col_outliers_df = processed_df[col_outliers_mask]

                            if not col_outliers_df.empty:
                                outliers_found_overall = True
                                detected_outliers_info[col] = {
                                    "method": outlier_method, # Store the method used for this column
                                    "count": len(col_outliers_df),
                                    "values": col_outliers_df[col].tolist(), # Store values for display
                                    "lower_bound": lower_bound,
                                    "upper_bound": upper_bound,
                                    "outlier_rows_preview": col_outliers_df.head(5).to_dict(orient='records') # Store a preview of the actual rows
                                }
                                
                                # Apply action
                                if outlier_action == "Remove rows with outliers":
                                    original_rows = len(processed_df)
                                    processed_df = processed_df[~col_outliers_mask]
                                    st.success(f"Removed {original_rows - len(processed_df)} rows with outliers in '{col}'.")
                                elif outlier_action == "Cap outliers (replace with boundary values)":
                                    processed_df[col] = np.where(processed_df[col] < lower_bound, lower_bound, processed_df[col])
                                    processed_df[col] = np.where(processed_df[col] > upper_bound, upper_bound, processed_df[col])
                                    st.success(f"Capped {len(col_outliers_df)} outliers in '{col}' with values between {lower_bound:.2f} and {upper_bound:.2f}.")
                        
                        if not outliers_found_overall:
                            st.info("No outliers found in the selected columns with the chosen method.")
                        
                        st.session_state['current_dataframe'] = processed_df
                        # Store detected outliers info in session state to display outside the expander
                        st.session_state['detected_outliers_info'] = detected_outliers_info
                        st.rerun()
                    else:
                        st.info("No outlier handling method applied.")
            st.markdown("---")
            # --- End Outlier Detection and Handling Configuration ---

            # --- Display Detected Outliers (Moved outside the 'Configure Outlier Handling' expander) ---
            if 'detected_outliers_info' in st.session_state and st.session_state['detected_outliers_info']:
                st.write("### Detected Outliers Summary:")
                with st.expander("Click to view details of detected outliers"):
                    for col, info in st.session_state['detected_outliers_info'].items():
                        st.markdown(f"**Column: '{col}'**")
                        st.write(f"  - Method: {info.get('method', 'N/A')}") # Use stored method
                        st.write(f"  - Detected Count: {info['count']}")
                        st.write(f"  - Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
                        st.write(f"  - Sample Outlier Values: {info['values'][:10]} {'...' if len(info['values']) > 10 else ''}") # Show first 10 values
                        
                        if info.get('outlier_rows_preview'):
                            st.markdown("**Preview of Outlier Rows (first 5):**")
                            st.dataframe(pd.DataFrame(info['outlier_rows_preview']))
                        else:
                            st.info(f"No specific outlier rows preview available for '{col}'.")
                        st.markdown("---")
            st.markdown("---") # Separator after outlier display

            # Data Standardization Section
            st.subheader("Data Standardization (Numerical Columns)")
            standardization_method = st.radio(
                "Select standardization method:",
                ("Do nothing", "Z-score Standardization", "Min-Max Scaling", "Robust Scaling"),
                key="standardization_method"
            )

            min_max_feature_range = (0, 1)
            if standardization_method == "Min-Max Scaling":
                st.caption("Configure Min-Max Scaling Range:")
                col_min_range, col_max_range = st.columns(2)
                with col_min_range:
                    min_val = st.number_input("Minimum value for scaling:", value=0.0, key="min_range_input")
                with col_max_range:
                    max_val = st.number_input("Maximum value for scaling:", value=1.0, key="max_range_input")
                min_max_feature_range = (min_val, max_val)
                if min_val >= max_val:
                    st.error("Minimum value must be less than maximum value for Min-Max Scaling.")
                    st.stop()


            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            if numerical_columns:
                cols_to_standardize = st.multiselect(
                    "Select numerical columns to standardize:",
                    numerical_columns,
                    key="cols_to_standardize"
                )

                if st.button("Apply Standardization/Scaling", key="apply_standardization_button"):
                    if standardization_method != "Do nothing" and not cols_to_standardize:
                        st.warning("Please select at least one column to standardize/scale.")
                    elif standardization_method != "Do nothing" and cols_to_standardize:
                        processed_df = df.copy()

                        for col in cols_to_standardize:
                            if standardization_method == "Z-score Standardization":
                                scaler = StandardScaler()
                                processed_df[col] = scaler.fit_transform(processed_df[[col]])
                                st.success(f"Applied Z-score standardization to '{col}'.")
                            elif standardization_method == "Min-Max Scaling":
                                scaler = MinMaxScaler(feature_range=min_max_feature_range)
                                processed_df[col] = scaler.fit_transform(processed_df[[col]])
                                st.success(f"Applied Min-Max scaling to '{col}' (range: {min_max_feature_range}).")
                            elif standardization_method == "Robust Scaling":
                                scaler = RobustScaler()
                                processed_df[col] = scaler.fit_transform(processed_df[[col]])
                                st.success(f"Applied Robust Scaling to '{col}'.")
                        
                        st.session_state['current_dataframe'] = processed_df
                        # Store the applied standardization method
                        st.session_state['last_standardization_applied'] = standardization_method
                        st.rerun()
                    else:
                        st.info("No standardization/scaling method applied.")
            else:
                st.info("No numerical columns found for standardization/scaling.")
        st.markdown("---") # End of Cleaning & Preprocessing container

        # --- Always Display Current Data Preview ---
        st.subheader("Current Data Preview (after all applied operations):")
        if st.session_state['current_dataframe'].empty:
            st.warning("The dataset is currently empty after applying the selected operations.")
        else:
            st.dataframe(st.session_state['current_dataframe'].head(10)) # Display first 10 rows

        csv_data = st.session_state['current_dataframe'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Current Data (CSV)",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv",
            key="download_processed_csv"
        )

        if st.button("Reset Data to Original Upload", key="reset_data_button_modified"):
            st.session_state['current_dataframe'] = st.session_state['original_dataframe'].copy()
            # Reset standardization state on reset
            st.session_state['last_standardization_applied'] = "Do nothing" 
            # Clear graph-related session state variables on reset
            for key in ['generated_image_b64', 'generated_image_mime', 'x_axis_select', 'y_axis_select']:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear outlier info on reset
            if 'detected_outliers_info' in st.session_state:
                del st.session_state['detected_outliers_info']
            # Clear dependent variable selection on reset
            if 'dependent_variable' in st.session_state:
                del st.session_state['dependent_variable']
            st.success("Data reset to original uploaded state.")
            st.rerun()

        st.markdown("---")


        # --- Conditional Sections based on Missing Values ---
        if total_missing_values == 0:
            # --- Model Configuration (New Section) ---
            st.header("3. Model Configuration")
            with st.container(border=True):
                st.subheader("Select Dependent Variable")
                all_columns = df.columns.tolist()
                default_dependent_var = st.session_state.get('dependent_variable', None)
                if default_dependent_var and default_dependent_var in all_columns:
                    default_index = all_columns.index(default_dependent_var)
                else:
                    default_index = 0 if all_columns else None

                selected_dependent_variable = st.selectbox(
                    "Choose your dependent variable (Y):",
                    ["-- Select --"] + all_columns,
                    index=default_index + 1 if default_index is not None else 0, # +1 because of "-- Select --"
                    key="dependent_variable_select"
                )

                if selected_dependent_variable != "-- Select --":
                    st.session_state['dependent_variable'] = selected_dependent_variable
                    st.success(f"Dependent variable set to: **{selected_dependent_variable}**")
                else:
                    st.session_state['dependent_variable'] = None
                    st.info("Please select a dependent variable for model-related analyses.")
            st.markdown("---")

            # --- Multicollinearity Analysis (New Section) ---
            st.header("4. Multicollinearity Analysis")
            with st.container(border=True):
                st.markdown("Identify and assess multicollinearity among independent numerical variables.")
                
                numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
                independent_numerical_cols = [col for col in numerical_cols if col != st.session_state.get('dependent_variable')]

                if not independent_numerical_cols:
                    st.warning("No independent numerical columns available for multicollinearity analysis. Ensure you have numerical columns and have selected a dependent variable if applicable.")
                else:
                    # Correlation Heatmap
                    st.subheader("Correlation Heatmap")
                    if st.button("Generate Correlation Heatmap", key="generate_corr_heatmap_button"):
                        if len(independent_numerical_cols) < 2:
                            st.info("Need at least two independent numerical variables to generate a correlation heatmap.")
                        else:
                            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                            sns.heatmap(df[independent_numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
                            ax_corr.set_title("Correlation Heatmap of Independent Numerical Variables")
                            plt.tight_layout()
                            st.pyplot(fig_corr)
                            plt.close(fig_corr)
                    st.markdown("---")

                    # VIF Calculation
                    st.subheader("Variance Inflation Factor (VIF)")
                    st.info("VIF values indicate how much the variance of an estimated regression coefficient is inflated due to multicollinearity. A VIF > 5-10 typically indicates significant multicollinearity.")
                    if st.button("Calculate VIF Values", key="calculate_vif_button"):
                        if len(independent_numerical_cols) < 2:
                            st.info("Need at least two independent numerical variables to calculate VIF.")
                        else:
                            # Drop rows with NaNs only for the columns relevant to VIF calculation
                            vif_df = df[independent_numerical_cols].dropna()
                            
                            if vif_df.empty:
                                st.warning("No complete rows for selected independent numerical variables after dropping missing values. Cannot calculate VIF.")
                            else:
                                try:
                                    # Add a constant to the independent variables for VIF calculation
                                    X = add_constant(vif_df)
                                    
                                    vif_data = pd.DataFrame()
                                    vif_data["feature"] = X.columns
                                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                                    
                                    # Remove the constant row from VIF display
                                    vif_data = vif_data[vif_data['feature'] != 'const']
                                    
                                    st.dataframe(vif_data.sort_values(by="VIF", ascending=False))
                                    st.success("VIF values calculated successfully.")
                                except Exception as e:
                                    st.error(f"Error calculating VIF: {e}. Ensure selected columns are numerical and have sufficient variance.")
            st.markdown("---")

            # --- Text-to-SQL Query Section (now section 5) ---
            st.header("5. Ask Questions about your Data (Text-to-SQL)")
            with st.container(border=True):
                st.info("SQL queries will run on the **original, uncleaned, and unstandardized** dataset.")
                sql_question = st.text_area(
                    "Enter your question about the data:",
                    placeholder="e.g., 'What is the maximum GDP?', 'Show me all data for 2015', 'List years and per capita GDP where growth was above 7%'",
                    key="sql_question_input"
                )

                if st.button("Generate & Run SQL Query", key="run_sql_button"):
                    if not sql_question.strip():
                        st.error("Please enter a question for the SQL query.")
                    else:
                        with st.spinner("Generating SQL query..."):
                            # Pass columns from the original dataframe for SQL generation context
                            generated_sql = generate_single_file_sql_query(sql_question, st.session_state['original_dataframe'].columns.tolist())

                        if generated_sql:
                            st.subheader("Generated SQL Query:")
                            st.code(generated_sql, language="sql")

                            with st.spinner("Executing SQL query on original data..."):
                                try:
                                    # Explicitly use the original_dataframe for SQL queries
                                    temp_df_for_sql = st.session_state['original_dataframe']
                                    sql_result = sqldf(generated_sql, {'df': temp_df_for_sql})
                                    st.subheader("Query Result:")
                                    if not sql_result.empty:
                                        st.dataframe(sql_result)
                                    else:
                                        st.info("The query returned no results.")
                                except Exception as e:
                                    st.error(f"Error executing SQL query: {e}. Please try rephrasing your question or check the generated SQL.")
                        else:
                            st.warning("Could not generate a valid SQL query. Please try rephrasing your question.")

            st.markdown("---")


            # --- Manual Graph Generation Section (now section 6) ---
            st.header("6. Generate Your Graph")
            with st.container(border=True):
                st.subheader("Plotting Options")
                enable_sampling = st.checkbox("Enable Data Sampling (Recommended for large datasets)", key="enable_sampling_checkbox_manual")
                sample_percentage = 100
                if enable_sampling:
                    sample_percentage = st.slider(
                        "Sample Percentage (%) for Plotting:",
                        min_value=1, max_value=100, value=50, step=1,
                        key="sample_percentage_slider_manual"
                    )
                    st.info("Sampling helps improve performance and readability for datasets with many entries by plotting a subset of the data.")
                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    all_columns = df.columns.tolist()
                    x_axis_default_index = 0
                    if 'x_axis_select' in st.session_state and st.session_state['x_axis_select'] in all_columns:
                        x_axis_default_index = all_columns.index(st.session_state['x_axis_select'])
                    x_axis = st.selectbox("Select X-axis column:", all_columns, index=x_axis_default_index, key="x_axis_select")
                with col2:
                    # Changed to st.multiselect for Y-axis
                    numerical_columns_for_y = df.select_dtypes(include=['number']).columns.tolist()
                    y_axis_options = [col for col in numerical_columns_for_y if col != x_axis] # Exclude X-axis from Y options
                    
                    # Default selection for multiselect
                    default_y_selection = []
                    if 'y_axis_select' in st.session_state:
                        # Ensure previously selected Ys are still valid and numerical
                        default_y_selection = [y for y in st.session_state['y_axis_select'] if y in y_axis_options]
                    
                    y_axes = st.multiselect("Select Y-axis column(s) (numerical only):", y_axis_options, default=default_y_selection, key="y_axis_select")


                graph_type = st.selectbox(
                    "Select Graph Type:",
                    ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot"],
                    key="graph_type_select_manual"
                )

                generate_plot_button = st.button("Generate Plot", type="primary", key="generate_plot_button_manual")

                if generate_plot_button and x_axis:
                    try:
                        plot_df = df.copy() # This is the current_dataframe, potentially standardized

                        # Prepare original data for AI BEFORE any plotting-specific type conversions or NaNs are dropped
                        # This uses the original_dataframe from session state
                        original_df_for_ai = st.session_state['original_dataframe'].copy()
                        cols_for_ai_data = [x_axis]
                        if y_axes:
                            cols_for_ai_data.extend(y_axes)
                        
                        # Ensure original_df_for_ai has the columns and types that make sense for the AI to interpret as "original"
                        # For simplicity, we'll just take the selected columns from the original_dataframe
                        # and convert them to numeric, coercing errors, then dropna if they were used for plotting.
                        # This ensures the AI sees a clean, relevant subset of the original data.
                        original_plot_data_for_ai = original_df_for_ai[cols_for_ai_data].copy()
                        for col in cols_for_ai_data:
                            original_plot_data_for_ai[col] = pd.to_numeric(original_plot_data_for_ai[col], errors='coerce')
                        original_plot_data_for_ai.dropna(subset=cols_for_ai_data, inplace=True)
                        st.session_state['last_plot_original_dataframe_str'] = original_plot_data_for_ai.to_string()


                        if enable_sampling and sample_percentage < 100:
                            num_samples = max(1, int(len(plot_df) * (sample_percentage / 100)))
                            plot_df = plot_df.sample(n=num_samples, random_state=42)
                            st.info(f"Plotting {num_samples} samples ({sample_percentage}%) from the dataset.")

                        # Handle data type conversion for selected columns for plotting
                        # This is for the `plot_df` which might be standardized
                        if graph_type in ["Line Plot", "Scatter Plot", "Bar Plot"]:
                            plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors='coerce')
                            if y_axes:
                                for y_col in y_axes:
                                    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
                            cols_to_check_na = [x_axis] + y_axes if y_axes else [x_axis]
                            plot_df.dropna(subset=cols_to_check_na, inplace=True)
                        elif graph_type == "Histogram":
                            if not y_axes: # For histogram, X-axis is the primary numerical column
                                plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors='coerce')
                                plot_df.dropna(subset=[x_axis], inplace=True)
                            else: # If Y-axis is selected for histogram, use the first Y
                                plot_df[y_axes[0]] = pd.to_numeric(plot_df[y_axes[0]], errors='coerce')
                                plot_df.dropna(subset=[y_axes[0]], inplace=True)
                        elif graph_type == "Box Plot":
                            if not y_axes:
                                st.warning("Please select at least one Y-axis (numerical) column for Box Plot.")
                                plt.close(fig)
                                st.stop()
                            for y_col in y_axes:
                                plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
                            plot_df.dropna(subset=[x_axis] + y_axes, inplace=True)


                        if plot_df.empty:
                            st.warning("No valid data points after cleaning and/or sampling for selected columns. Please check your data or column types.")
                        else:
                            fig, ax = plt.subplots(figsize=(10, 6))

                            if graph_type == "Line Plot":
                                if not y_axes:
                                    st.warning("Please select at least one Y-axis column for Line Plot.")
                                    plt.close(fig)
                                    st.stop()
                                for y_col in y_axes:
                                    sns.lineplot(x=plot_df[x_axis], y=plot_df[y_col], ax=ax, marker='o', label=y_col)
                                ax.set_title(f'Trends over {x_axis} (Line Plot)')
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel('Value')
                                ax.legend(title="Metrics")
                            elif graph_type == "Bar Plot":
                                if not y_axes or len(y_axes) > 1:
                                    st.warning("Please select exactly one Y-axis column for Bar Plot. For multiple Y-axes, consider Line Plot or Scatter Plot.")
                                    plt.close(fig)
                                    st.stop()
                                y_col = y_axes[0]
                                sns.barplot(x=plot_df[x_axis], y=plot_df[y_col], ax=ax)
                                if len(plot_df[x_axis].unique()) > 10:
                                    ax.tick_params(axis='x', rotation=45)
                                ax.set_title(f'{y_col} vs. {x_axis} (Bar Plot)')
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel(y_col)
                            elif graph_type == "Scatter Plot":
                                if not y_axes:
                                    st.warning("Please select at least one Y-axis column for Scatter Plot.")
                                    plt.close(fig)
                                    st.stop()
                                for y_col in y_axes:
                                    sns.scatterplot(x=plot_df[x_axis], y=plot_df[y_col], ax=ax, alpha=0.5, label=y_col)
                                ax.set_title(f'Relationship with {x_axis} (Scatter Plot)')
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel('Value')
                                ax.legend(title="Metrics")
                            elif graph_type == "Histogram":
                                if not y_axes or len(y_axes) > 1:
                                    st.warning("Please select exactly one Y-axis column for Histogram. Histograms are for single numerical distributions.")
                                    plt.close(fig)
                                    st.stop()
                                y_col = y_axes[0]
                                sns.histplot(data=plot_df, x=y_col, kde=True, ax=ax)
                                ax.set_title(f'Distribution of {y_col} (Histogram)')
                                ax.set_xlabel(y_col)
                                ax.set_ylabel('Frequency')
                                st.info(f"Showing the distribution of '{y_col}'. For histograms, only one numerical column is typically used.")
                            elif graph_type == "Box Plot":
                                if not y_axes:
                                    st.warning("Please select at least one Y-axis (numerical) column for Box Plot.")
                                    plt.close(fig)
                                    st.stop()
                                y_col = y_axes[0]
                                sns.boxplot(x=plot_df[x_axis].astype(str), y=plot_df[y_col], ax=ax)
                                if len(plot_df[x_axis].unique()) > 10:
                                    ax.tick_params(axis='x', rotation=45)
                                ax.set_title(f'Box Plot of {y_col} by {x_axis}')
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel(y_col)
                                st.info(f"Showing the distribution of '{y_col}' for each category in '{x_axis}'.")


                            plt.grid(True, linestyle='--', alpha=0.7)
                            plt.tight_layout()

                            st.pyplot(fig)

                            base64_image, mime_type = fig_to_base64(fig)
                            st.session_state['generated_image_b64'] = base64_image
                            st.session_state['generated_image_mime'] = mime_type
                            
                            st.session_state['last_plot_x_axis_label'] = x_axis
                            st.session_state['last_plot_y_axis_label'] = ", ".join(y_axes) if y_axes else "N/A (Histogram/No Y selected)"
                            st.session_state['last_plot_type'] = graph_type
                            
                            # Dataframe string for the AI to interpret (this is the potentially scaled data)
                            st.session_state['last_plot_dataframe_full_str'] = plot_df[cols_for_ai_data].to_string()
                            
                            plt.close(fig)

                    except Exception as e:
                        st.error(f"Error generating plot: {e}. Please check your column selections and data types.")
                else:
                    st.info("Select X-axis and Y-axis columns and a graph type, then click 'Generate Plot'.")
            st.markdown("---") # End of Generate Graph container

            # --- AI Interpretation Section (now section 7) ---
            st.header("7. Ask About the Generated Graph")
            with st.container(border=True):
                if 'generated_image_b64' in st.session_state and st.session_state['generated_image_b64'] is not None:
                    graph_question = st.text_area(
                        "Enter your question here:",
                        placeholder="e.g., 'What trends do you observe?' or 'What is the highest value on this graph and when did it occur?'",
                        height=100,
                        key="ai_graph_question_input"
                    )

                    if st.button("Ask AI about this graph", key="ask_ai_graph_button"):
                        if not graph_question.strip():
                            st.error("Please enter a question.")
                        else:
                            with st.spinner("AI is interpreting the graph..."):
                                image_b64 = st.session_state['generated_image_b64']
                                image_mime = st.session_state['generated_image_mime']
                                x_label = st.session_state.get('last_plot_x_axis_label', 'Unknown X-axis')
                                y_label = st.session_state.get('last_plot_y_axis_label', 'N/A')
                                plot_type = st.session_state.get('last_plot_type', 'Unknown Plot Type')
                                dataframe_full_str = st.session_state.get('last_plot_dataframe_full_str', 'No data available.')
                                original_plot_data_for_ai_str = st.session_state.get('last_plot_original_dataframe_str', 'No original data available.')
                                applied_std_method = st.session_state.get('last_standardization_applied', 'Do nothing')

                                ai_answer = interpret_graph(graph_question, image_b64, image_mime, x_label, y_label, plot_type, dataframe_full_str, original_plot_data_for_ai_str, applied_std_method)

                            st.subheader("AI's Answer:")
                            st.info(ai_answer)
                            st.markdown(
                                """
                                <small>
                                *Note: AI interpretation of precise numerical values from images may have slight inaccuracies.
                                For exact figures, please refer to your original data source.*
                                </small>
                                """,
                                unsafe_allow_html=True
                            )
                            if len(dataframe_full_str) > 5000:
                                st.warning("Note: A large amount of data was sent to the AI for interpretation. This might affect response time and could potentially hit API token limits for extremely large datasets.")
                else:
                    st.info("Generate a graph first (Section 6) to enable AI interpretation.")
            st.markdown("---") # End of AI Interpretation container
        else:
            st.warning("Sections 3, 4, 5, 6, and 7 are disabled because your dataset contains missing values. Please handle missing values in section 2 first.")
            st.markdown("---")
    else:
        st.info("Upload a data file in Section 1 to begin.")

    # --- Section 8: SQL Operations on Multiple Data Files (Text-to-SQL) ---
    st.header("8. Text-to-SQL for Multiple Data Files")
    with st.container(border=True):
        st.markdown("Combine and query data from two or more files using natural language.")
        uploaded_multi_files = st.file_uploader(
            "Choose Data Files (2 or more)",
            type=["csv", "xlsx", "json"],
            accept_multiple_files=True,
            key="multi_file_uploader"
        )

        multi_dfs = {} # Dictionary to store DataFrames: {'df_file1': df1, 'df_file2': df2, ...}
        multi_dfs_info_for_ai = {} # Dictionary to store info for AI: {'df_file1': {'columns': [], 'primary_key': ''}, ...}

        if uploaded_multi_files:
            if len(uploaded_multi_files) < 2:
                st.warning("Please upload at least two data files to perform multi-file SQL operations.")
            else:
                st.subheader("Uploaded Files and Primary Key Selection:")
                for i, file_buffer in enumerate(uploaded_multi_files):
                    df_name = f"df_file{i+1}"
                    file_extension = file_buffer.name.split('.')[-1]
                    try:
                        current_multi_df = load_data(file_buffer, file_extension)
                        multi_dfs[df_name] = current_multi_df
                        st.write(f"**{df_name}** (from `{file_buffer.name}`):")
                        st.dataframe(current_multi_df.head())
                        st.markdown(f"**Columns in {df_name}:** `{', '.join(current_multi_df.columns.tolist())}`")


                        # Select primary key for each DataFrame
                        pk_options = current_multi_df.columns.tolist()
                        selected_pk = st.selectbox(
                            f"Select primary key for {df_name}:",
                            pk_options,
                            key=f"pk_select_{df_name}"
                        )
                        multi_dfs_info_for_ai[df_name] = {
                            'columns': current_multi_df.columns.tolist(),
                            'primary_key': selected_pk
                        }
                        st.markdown("---")

                    except Exception as e:
                        st.error(f"Error reading file '{file_buffer.name}': {e}")
                        continue

                if len(multi_dfs) >= 2: # Only proceed if at least two files were successfully loaded
                    st.subheader("Ask a Question about the Combined Data:")
                    st.info("The AI will generate a SQL query based on your question and the uploaded files. You can refer to tables by their generated names (e.g., `df_file1`, `df_file2`) and columns by their exact names. If you ask about an entity by its name (e.g., 'Product X'), the AI will try to find the corresponding ID for joining.")
                    
                    multi_sql_question_nl = st.text_area(
                        "Enter your natural language question:",
                        placeholder="e.g., 'Show me the customer names from df_file1 and their corresponding order amounts from df_file2, joining on CustomerID. Only include orders greater than 100.'",
                        key="sql_multi_question_nl_input",
                        height=150
                    )

                    if st.button("Generate & Run Multi-File SQL Query", type="primary", key="run_multi_sql_button"):
                        if not multi_sql_question_nl.strip():
                            st.error("Please enter a question.")
                        else:
                            with st.spinner("Generating multi-file SQL query..."):
                                generated_multi_sql = generate_multi_file_sql_query(multi_sql_question_nl, multi_dfs_info_for_ai)

                            if generated_multi_sql:
                                st.subheader("Generated SQL Query:")
                                st.code(generated_sql, language="sql")

                                with st.spinner("Executing multi-file SQL query..."):
                                    try:
                                        # Execute the SQL query using the dictionary of dataframes
                                        multi_sql_result = sqldf(generated_multi_sql, multi_dfs)
                                        st.subheader("Multi-File SQL Query Result:")
                                        if not multi_sql_result.empty:
                                            st.dataframe(multi_sql_result)
                                        else:
                                            st.info("The multi_file SQL query returned no results.")
                                    except Exception as e:
                                        st.error(f"Error executing multi-file SQL query: {e}. Please check the generated SQL or your data.")
                            else:
                                st.warning("Could not generate a valid SQL query. Please try rephrasing your question.")
                else:
                    st.info("Upload at least two valid data files to enable multi-file SQL operations.")
        else:
            st.info("Upload data files to enable multi-file SQL operations.")

    st.markdown("---") # Final separator

# --- Run the application ---
if __name__ == "__main__":
    main()
