<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>README - Data Analysis & Visualization Platform</title>
</head>
<body>
  <h1>📊 Data Analysis & Visualization Platform</h1>
  <p>This project is a <strong>Streamlit-based web application</strong> that provides data analysis, cleaning, visualization, and AI-powered insights. It integrates Gemini AI for SQL generation and graph interpretation.</p>

  <h2>⚙️ Features</h2>
  <ul>
    <li>Upload CSV, Excel, or JSON files for analysis</li>
    <li>Interactive data quality dashboard</li>
    <li>Data cleaning (handle missing values, duplicates, standardization)</li>
    <li>Automatic detection and conversion of numeric data</li>
    <li>SQL query generator powered by Gemini AI</li>
    <li>Data visualization (line, bar, scatter, histogram, box, heatmap)</li>
    <li>AI-powered graph interpretation</li>
    <li>Excel-like filter interface</li>
  </ul>

  <h2>📂 File Overview</h2>
  <p><code>main.py</code> is the core application file containing all functions, UI components, and logic.</p>

  <h2>📝 Functions Overview</h2>
  <ul>
    <li><code>load_data(uploaded_file_buffer, file_type)</code> - Load datasets (CSV, XLSX, JSON)</li>
    <li><code>fig_to_base64(_fig)</code> - Convert matplotlib figures to Base64 PNG</li>
    <li><code>detect_potential_numeric(series)</code> - Detect numeric patterns in string columns</li>
    <li><code>find_potential_keys(df)</code> - Identify potential primary/unique keys in a DataFrame</li>
    <li><code>clean_and_convert_numeric(value)</code> - Clean string values and convert to numeric</li>
    <li><code>convert_column_to_numeric(df, column_name)</code> - Convert entire column to numeric</li>
    <li><code>analyze_column_conversion(df, column_name)</code> - Analyze conversion feasibility</li>
    <li><code>apply_standardization(df, cols_to_standardize, method)</code> - Apply Z-score, Min-Max, or Robust scaling</li>
    <li><code>generate_single_file_sql_query(question, df_columns)</code> - Generate SQL for single dataset using Gemini AI</li>
    <li><code>generate_multi_file_sql_query(question, multi_dfs_info)</code> - Generate SQL for multiple datasets</li>
    <li><code>interpret_graph(question, image_b64, ...)</code> - Interpret charts with AI</li>
    <li><code>create_enhanced_dashboard(df)</code> - Build interactive data quality dashboard</li>
    <li><code>display_data_overview()</code> - Display dataset overview & tabs</li>
    <li><code>display_data_cleaning()</code> - Handle missing values, duplicates, and standardization</li>
    <li><code>display_numeric_conversion()</code> - Detect and convert numeric values from text columns</li>
    <li><code>display_sql_query()</code> - AI SQL query interface</li>
    <li><code>display_visualization()</code> - Generate various plots (line, bar, scatter, histogram, box, heatmap)</li>
    <li><code>display_ai_analysis()</code> - Analyze visualizations with AI</li>
    <li><code>create_excel_filter_interface(df)</code> - Excel-style interactive data filtering</li>
    <li><code>main()</code> - Entry point for the Streamlit app</li>
  </ul>

  <h2>🚀 Running the App</h2>
  <pre>
  streamlit run main.py
  </pre>

  <h2>📌 Notes</h2>
  <ul>
    <li>Requires a valid <code>GOOGLE_API_KEY</code> in a <code>.env</code> file to use Gemini API features.</li>
    <li>Dependencies: <code>streamlit, pandas, plotly, seaborn, sklearn, statsmodels, dotenv, pandasql</code>.</li>
  </ul>
</body>
</html>
