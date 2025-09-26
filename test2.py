# Stock Analytics & AI Platform - Enhanced Stock Analysis Integration

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
import json
import pickle
import io
import glob
import re
from pathlib import Path
from typing import List, Dict, Optional, Union
import base64

# Time series and ML imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Vector database imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.warning("⚠️ FAISS not installed. Install with: pip install faiss-cpu")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    st.warning("⚠️ ChromaDB not installed. Install with: pip install chromadb")

def map_csvs_to_sector_industry(
    mapping_excel_path="final_stock_classification_6col.xlsx", 
    csv_folder_path="cleaned"
):
    import pandas as pd
    import glob
    import os
    
    mapping_df = pd.read_excel(mapping_excel_path, sheet_name=0)
    mapping_df['Ticker'] = mapping_df['Ticker'].astype(str).str.strip().str.upper()
    mapping_df.fillna('', inplace=True)
    
    csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))
    mapping_results = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.split('_')[0].split('.')[0].upper()
        
        row = mapping_df[mapping_df['Ticker'] == ticker]
        if not row.empty:
            info = row.iloc[0].to_dict()
            info['csv_filename'] = filename
            info['csv_path'] = filepath
            mapping_results.append(info)
    
    return pd.DataFrame(mapping_results)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="📈 Stock Analytics & AI Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.section-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #2e8b57;
    border-bottom: 3px solid #2e8b57;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.2rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}

.success-box {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 2px solid #28a745;
    color: #155724;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.info-box {
    background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    border: 2px solid #17a2b8;
    color: #0c5460;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.warning-box {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 2px solid #ffc107;
    color: #856404;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.stock-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: all 0.2s;
}

.stock-card:hover {
    border-color: #667eea;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.executive-summary {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 5px solid #007bff;
    padding: 1.2rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 1.1rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# Initialize Gemini
@st.cache_resource
def init_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Please set GOOGLE_API_KEY in your .env file")
        st.info("Get your API key from: https://makersuite.google.com/")
        st.stop()
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini: {e}")
        st.stop()

# Session state initialization
def init_session_state():
    defaults = {
        'messages': [],
        'current_dataset': None,
        'selected_stocks': [],
        'vector_store': None,
        'embeddings': None,
        'arima_results': None,
        'analysis_results': {},
        'available_files': [],
        'ai_context_cache': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================================
# DATA LOADING AND MANAGEMENT
# ================================

@st.cache_data
def scan_cleaned_folder(folder_path: str = "cleaned") -> List[Dict]:
    """Scan the cleaned folder for CSV files and extract metadata + sector info"""
    mapping_excel = "final_stock_classification_6col.xlsx"
    
    if not os.path.exists(folder_path):
        st.warning(f"⚠️ Folder '{folder_path}' not found. Please create it and add your CSV files.")
        return []
    
    if not os.path.exists(mapping_excel):
        st.error(f"❌ Mapping Excel file '{mapping_excel}' not found in this folder!")
        return []
    
    # Load sector mapping
    mapping_df = pd.read_excel(mapping_excel)
    mapping_df['Ticker'] = mapping_df['Ticker'].astype(str).str.strip().str.upper()
    mapping_df.fillna('', inplace=True)
    
    # Scan for CSVs
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        st.warning(f"⚠️ No CSV files found in '{folder_path}' folder.")
        return []
    
    records = []
    not_found = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        ticker = filename.split('_')[0].split('.')[0].upper()
        
        row = mapping_df[mapping_df['Ticker'] == ticker]
        
        try:
            df_info = pd.read_csv(file_path, usecols=['Date'])
            total_rows = len(df_info)
            dates = pd.to_datetime(df_info['Date'])
            start_date = dates.min()
            end_date = dates.max()
            df_sample = pd.read_csv(file_path, nrows=5)
            
            if not row.empty:
                info = row.iloc[0].to_dict()
                info.update({
                    'filename': filename,
                    'filepath': file_path,
                    'symbol': ticker,
                    'total_rows': total_rows,
                    'start_date': start_date,
                    'end_date': end_date,
                    'columns': list(df_sample.columns)
                })
                records.append(info)
            else:
                not_found.append(filename)
        except Exception as e:
            st.error(f"❌ Error reading {file_path}: {e}")
            continue
    
    if not_found:
        st.info(f"🔎 Could not map the following files to a ticker in your Excel mapping: {not_found}")
    
    return sorted(records, key=lambda x: x['symbol'])

@st.cache_data
def load_stock_csv(file_path: str) -> pd.DataFrame:
    """Load and process stock CSV file"""
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def create_comprehensive_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive technical indicators and features"""
    df = data.copy()
    
    if 'Close' in df.columns:
        # Basic features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']).diff()
        df['HL_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
        
        # Moving averages
        windows = [5, 10, 20, 50, 100, 200]
        for window in windows:
            if len(df) >= window:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
                df[f'SMA_{window}_ratio'] = df['Close'] / df[f'SMA_{window}']
        
        # Volatility measures
        for window in [10, 20, 30]:
            if len(df) >= window:
                df[f'Volatility_{window}D'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
                df[f'Price_Std_{window}D'] = df['Close'].rolling(window=window).std()
        
        # RSI (Relative Strength Index)
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if len(df) >= 20:
            bb_window = 20
            bb_std_multiplier = 2
            df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
            bb_std = df['Close'].rolling(window=bb_window).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * bb_std_multiplier)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * bb_std_multiplier)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD
        if len(df) >= 26:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume-based indicators
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['Price_Volume'] = df['Close'] * df['Volume']
            df['VPT'] = (df['Volume'] * df['Returns']).cumsum()
            df['OBV'] = (np.sign(df['Returns']) * df['Volume']).cumsum()
        
        # Support and Resistance levels
        for window in [20, 50]:
            if len(df) >= window:
                df[f'Resistance_{window}D'] = df['High'].rolling(window=window).max()
                df[f'Support_{window}D'] = df['Low'].rolling(window=window).min()
        
        # Price position within recent range
        for window in [10, 20, 50]:
            if len(df) >= window:
                period_high = df['High'].rolling(window=window).max()
                period_low = df['Low'].rolling(window=window).min()
                df[f'Price_Position_{window}D'] = (df['Close'] - period_low) / (period_high - period_low)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else None
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    return df

# ================================
# ENHANCED DATA SLICING AND PROCESSING
# ================================

@st.cache_data
def get_data_slice(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Get processed data slice for a symbol within date range with downsampling"""
    # Find the symbol's data
    available_files = st.session_state.available_files
    stock_file = None
    
    for file_info in available_files:
        if file_info['symbol'] == symbol:
            stock_file = file_info['filepath']
            break
    
    if stock_file is None:
        return pd.DataFrame()
    
    # Load raw data
    raw_data = load_stock_csv(stock_file)
    if raw_data.empty:
        return pd.DataFrame()
    
    # Apply date range filter
    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
    filtered_data = raw_data.loc[mask].copy()
    
    if filtered_data.empty:
        return pd.DataFrame()
    
    # Create features
    enhanced_data = create_comprehensive_features(filtered_data)
    
    # Determine sampling frequency based on date range
    date_range_days = (end_date - start_date).days
    
    if date_range_days <= 365:  # Less than 1 year - daily
        sampled_data = enhanced_data
    elif date_range_days <= 1825:  # 1-5 years - weekly
        sampled_data = enhanced_data.resample('W').last()
    else:  # More than 5 years - monthly
        sampled_data = enhanced_data.resample('M').last()
    
    # Drop NaNs to avoid plotting warnings
    sampled_data = sampled_data.dropna(subset=['Close'])
    
    return sampled_data

def generate_executive_summary(symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> str:
    """Generate executive summary for selected stocks"""
    if not symbols or not data_dict:
        return "No data available for analysis."
    
    summaries = []
    
    for symbol in symbols:
        if symbol not in data_dict or data_dict[symbol].empty:
            continue
            
        data = data_dict[symbol]
        latest = data.iloc[-1]
        
        # Calculate key metrics
        current_price = latest['Close']
        price_change = latest.get('Returns', 0) * 100
        rsi = latest.get('RSI', 50)
        
        # Determine trend
        if 'SMA_50' in data.columns and not pd.isna(latest.get('SMA_50')):
            trend = "bullish" if current_price > latest['SMA_50'] else "bearish"
        else:
            trend = "neutral"
        
        # Determine momentum
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        else:
            momentum = "neutral"
        
        summary = f"{symbol} trades at ${current_price:.2f} ({price_change:+.1f}%) with {trend} trend and {momentum} momentum."
        summaries.append(summary)
    
    if len(summaries) == 1:
        return summaries[0]
    elif len(summaries) <= 3:
        return " ".join(summaries)
    else:
        return f"Analysis covers {len(summaries)} stocks with mixed market conditions and varying technical signals."

def downsample_for_plotting(data: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    """Downsample data for plotting performance"""
    if len(data) <= max_points:
        return data
    
    # Calculate step size
    step = len(data) // max_points
    return data.iloc[::step].copy()

# ================================
# EXPORT FUNCTIONALITY
# ================================

def create_download_link(data, filename, file_format='csv'):
    """Create download link for data"""
    if file_format == 'csv':
        csv = data.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    return href

def export_chart_as_png(fig, filename):
    """Export plotly chart as PNG"""
    img_bytes = fig.to_image(format="png", engine="kaleido")
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download PNG</a>'
    return href

def generate_technical_report(symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> str:
    """Generate AI-based technical report"""
    try:
        model = init_gemini()
        
        # Prepare context for each symbol
        contexts = []
        for symbol in symbols:
            if symbol in data_dict and not data_dict[symbol].empty:
                context = generate_ai_context(data_dict[symbol], symbol)
                contexts.append(f"--- {symbol} Analysis ---\n{context}")
        
        combined_context = "\n\n".join(contexts)
        
        prompt = f"""
Generate a comprehensive technical analysis report for the following stock(s). Include:
1. Executive Summary (2-3 sentences)
2. Technical Outlook for each stock
3. Risk Assessment
4. Trading Recommendations
5. Market Outlook

Context:
{combined_context}

Keep the report professional, concise (max 500 words), and actionable. Include appropriate disclaimers.
"""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating report: {str(e)}"

# ================================
# AI CHAT FUNCTIONALITY
# ================================

def generate_ai_context(context_data: pd.DataFrame, stock_symbol: str) -> str:
    """Generate comprehensive AI context once and cache it"""
    if context_data is None or context_data.empty:
        return ""
    
    latest_price = context_data['Close'].iloc[-1] if 'Close' in context_data.columns else 0
    price_change = context_data['Returns'].iloc[-1] if 'Returns' in context_data.columns else 0
    
    if 'Returns' in context_data.columns:
        returns = context_data['Returns'].dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).mean() * 100
        avg_return = returns.mean() * 252 * 100
    else:
        volatility = sharpe = max_drawdown = win_rate = avg_return = 0
    
    current_rsi = context_data['RSI'].iloc[-1] if 'RSI' in context_data.columns else None
    bb_position = context_data['BB_Position'].iloc[-1] if 'BB_Position' in context_data.columns else None
    
    if current_rsi is not None and not pd.isna(current_rsi):
        rsi_display = f"{current_rsi:.1f}"
        if current_rsi < 30:
            rsi_signal = " (Oversold)"
        elif current_rsi > 70:
            rsi_signal = " (Overbought)"
        else:
            rsi_signal = " (Neutral)"
    else:
        rsi_display = "N/A"
        rsi_signal = ""
    
    if bb_position is not None and not pd.isna(bb_position):
        bb_display = f"{bb_position:.2f}"
        if bb_position > 0.8:
            bb_signal = " (Near Upper Band)"
        elif bb_position < 0.2:
            bb_signal = " (Near Lower Band)"
        else:
            bb_signal = " (Middle Range)"
    else:
        bb_display = "N/A"
        bb_signal = ""
    
    if 'Volume' in context_data.columns:
        avg_volume = context_data['Volume'].mean()
        volume_display = f"{avg_volume:,.0f}" if not pd.isna(avg_volume) else "N/A"
    else:
        volume_display = "N/A"
    
    context = f"""
Stock Analysis Context for {stock_symbol or 'Selected Stock'}:

📊 Price Metrics:
- Current Price: ${latest_price:.2f}
- Latest Daily Change: {price_change*100:.2f}%
- 52-Week High: ${context_data['Close'].max():.2f}
- 52-Week Low: ${context_data['Close'].min():.2f}
- Current vs 52W High: {((latest_price / context_data['Close'].max()) - 1)*100:.1f}%

📈 Performance Metrics:
- Annual Return: {avg_return:.1f}%
- Annual Volatility: {volatility*100:.1f}%
- Sharpe Ratio: {sharpe:.2f}
- Maximum Drawdown: {max_drawdown*100:.1f}%
- Win Rate: {win_rate:.1f}%

📊 Technical Indicators:
- RSI: {rsi_display}{rsi_signal}
- Bollinger Band Position: {bb_display}{bb_signal}

📅 Data Summary:
- Data Range: {context_data.index[0].strftime('%Y-%m-%d')} to {context_data.index[-1].strftime('%Y-%m-%d')}
- Total Trading Days: {len(context_data):,}
- Average Daily Volume: {volume_display}

Recent 5-day performance:
{context_data[['Close', 'Volume', 'Returns']].tail(5).round(3).to_string() if len(context_data) >= 5 else 'Insufficient data'}
"""
    return context

def get_ai_response(prompt: str, context_data: pd.DataFrame = None, stock_symbol: str = None) -> str:
    """Enhanced AI response with cached comprehensive stock context"""
    try:
        model = init_gemini()
        
        context_key = f"{stock_symbol}_{hash(str(context_data.index[-1]) if context_data is not None and not context_data.empty else 'none')}"
        
        if context_key not in st.session_state.ai_context_cache:
            if context_data is not None and not context_data.empty:
                st.session_state.ai_context_cache[context_key] = generate_ai_context(context_data, stock_symbol)
            else:
                st.session_state.ai_context_cache[context_key] = ""
        
        context = st.session_state.ai_context_cache[context_key]
        
        full_prompt = f"""
You are a professional financial analyst and data scientist with expertise in stock market analysis, technical indicators, and investment strategies. You have access to comprehensive market data and can provide detailed insights about stock performance, trends, and trading opportunities.

{context}

User Question: {prompt}

Provide detailed, actionable insights based on the data. Reference specific metrics and technical indicators where relevant. Be professional yet conversational. Use emojis strategically for emphasis. If discussing trading strategies, always include appropriate risk disclaimers.
"""
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ I encountered an error: {str(e)}. Please check your API key and try again."

# ================================
# ENHANCED VECTOR DATABASE FUNCTIONALITY
# ================================

# Text chunking implementation
class RecursiveCharacterTextSplitter:
    """Simple text splitter for document chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_ends = ['.', '!', '?', '\n']
                for i in range(end - 1, max(start + self.chunk_size // 2, start), -1):
                    if text[i] in sentence_ends:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - self.chunk_overlap
        
        return chunks

class EnhancedVectorStore:
    """Enhanced vector store for financial document search with full management capabilities"""
    
    def __init__(self, collection_name: str = "financial_documents"):
        self.collection_name = collection_name
        self.embeddings = []
        self.texts = []
        self.metadatas = []
        self.faiss_index = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        self.load_persistent_state()
    
    def load_persistent_state(self):
        """Load vector store state from disk if available"""
        try:
            persistence_dir = "vector_store_data"
            os.makedirs(persistence_dir, exist_ok=True)
            
            faiss_path = os.path.join(persistence_dir, f"{self.collection_name}.index")
            texts_path = os.path.join(persistence_dir, f"{self.collection_name}_texts.pkl")
            metadata_path = os.path.join(persistence_dir, f"{self.collection_name}_metadata.pkl")
            
            if os.path.exists(faiss_path) and os.path.exists(texts_path) and os.path.exists(metadata_path):
                if FAISS_AVAILABLE:
                    self.faiss_index = faiss.read_index(faiss_path)
                
                with open(texts_path, 'rb') as f:
                    self.texts = pickle.load(f)
                
                with open(metadata_path, 'rb') as f:
                    self.metadatas = pickle.load(f)
                
                embeddings_path = os.path.join(persistence_dir, f"{self.collection_name}_embeddings.pkl")
                if os.path.exists(embeddings_path):
                    with open(embeddings_path, 'rb') as f:
                        self.embeddings = pickle.load(f)
                
                st.success(f"✅ Loaded {len(self.texts)} documents from persistent storage")
                
        except Exception as e:
            st.warning(f"⚠️ Could not load persistent state: {e}")
    
    def save_persistent_state(self):
        """Save vector store state to disk"""
        try:
            persistence_dir = "vector_store_data"
            os.makedirs(persistence_dir, exist_ok=True)
            
            faiss_path = os.path.join(persistence_dir, f"{self.collection_name}.index")
            texts_path = os.path.join(persistence_dir, f"{self.collection_name}_texts.pkl")
            metadata_path = os.path.join(persistence_dir, f"{self.collection_name}_metadata.pkl")
            embeddings_path = os.path.join(persistence_dir, f"{self.collection_name}_embeddings.pkl")
            
            if FAISS_AVAILABLE and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, faiss_path)
            
            with open(texts_path, 'wb') as f:
                pickle.dump(self.texts, f)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadatas, f)
                
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
        except Exception as e:
            st.error(f"❌ Could not save persistent state: {e}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 5) -> np.ndarray:
        """Generate embeddings using Gemini with optimized batch processing"""
        if not texts:
            return np.array([])
        
        embeddings = []
        progress_bar = st.progress(0)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch_texts,
                    task_type="retrieval_document"
                )
                
                if isinstance(result['embedding'], list) and len(result['embedding']) > 0:
                    if isinstance(result['embedding'][0], dict):
                        batch_embeddings = [emb['embedding'] for emb in result['embedding']]
                    else:
                        batch_embeddings = result['embedding']
                else:
                    batch_embeddings = [result['embedding']]
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                st.error(f"❌ Embedding error for batch {i//batch_size + 1}: {e}")
                batch_embeddings = [np.zeros(768).tolist() for _ in batch_texts]
                embeddings.extend(batch_embeddings)
            
            progress = min((i + batch_size) / len(texts), 1.0)
            progress_bar.progress(progress)
            time.sleep(0.1)
        
        progress_bar.empty()
        return np.array(embeddings, dtype=np.float32)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to vector store with chunking and improved error handling"""
        if not texts:
            return
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        st.info("🔄 Processing documents with chunking and generating embeddings...")
        
        all_chunks = []
        all_chunk_metadata = []
        
        for text, metadata in zip(texts, metadatas):
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk),
                    'parent_document_id': len(self.texts) + len(all_chunks) - len(chunks) + i
                })
                all_chunk_metadata.append(chunk_metadata)
        
        st.info(f"📝 Created {len(all_chunks)} chunks from {len(texts)} documents")
        
        embeddings = self.generate_embeddings(all_chunks)
        if len(embeddings) == 0:
            st.error("❌ Failed to generate embeddings")
            return
        
        # FIX: Ensure embeddings array is 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        self.texts.extend(all_chunks)
        self.embeddings.extend(embeddings.tolist())
        self.metadatas.extend(all_chunk_metadata)
        
        self._rebuild_faiss_index()
        self.save_persistent_state()
        
        st.success(f"✅ Added {len(all_chunks)} document chunks to vector store")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from current embeddings"""
        if not self.embeddings or not FAISS_AVAILABLE:
            return
        
        embeddings_array = np.array(self.embeddings, dtype=np.float32)
        
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings_array)
        self.faiss_index.add(embeddings_array)
    
    def delete_document(self, doc_index: int):
        """Delete document by index and rebuild FAISS index"""
        if doc_index < 0 or doc_index >= len(self.texts):
            st.error("❌ Invalid document index")
            return False
        
        try:
            del self.texts[doc_index]
            del self.embeddings[doc_index]
            del self.metadatas[doc_index]
            
            self._rebuild_faiss_index()
            self.save_persistent_state()
            
            st.success(f"✅ Document {doc_index} deleted successfully")
            return True
            
        except Exception as e:
            st.error(f"❌ Error deleting document: {e}")
            return False
    
    def clear_database(self):
        """Clear entire database and persistent files"""
        try:
            self.texts = []
            self.embeddings = []
            self.metadatas = []
            self.faiss_index = None
            
            persistence_dir = "vector_store_data"
            if os.path.exists(persistence_dir):
                import shutil
                shutil.rmtree(persistence_dir)
            
            st.success("✅ Database cleared successfully")
            
        except Exception as e:
            st.error(f"❌ Error clearing database: {e}")
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories from metadata"""
        categories = set()
        for metadata in self.metadatas:
            if 'category' in metadata:
                categories.add(metadata['category'])
        return sorted(list(categories))
    
    def search(self, query: str, k: int = 5, category_filter: List[str] = None, title_filter: str = None) -> Dict:
        """Enhanced search with metadata filtering"""
        if not self.texts:
            return {'documents': [], 'distances': [], 'metadatas': []}
        
        query_embedding = self.generate_embeddings([query])
        if len(query_embedding) == 0:
            return {'documents': [], 'distances': [], 'metadatas': []}
        
        query_vec = query_embedding[0]
        
        search_multiplier = 5 if (category_filter or title_filter) else 1
        search_k = min(k * search_multiplier, len(self.texts))
        
        if FAISS_AVAILABLE and self.faiss_index is not None:
            query_vec_norm = query_vec.reshape(1, -1)
            faiss.normalize_L2(query_vec_norm)
            scores, indices = self.faiss_index.search(query_vec_norm.astype(np.float32), search_k)
            
            initial_results = {
                'documents': [self.texts[idx] for idx in indices[0] if idx < len(self.texts)],
                'distances': scores[0].tolist(),
                'metadatas': [self.metadatas[idx] for idx in indices[0] if idx < len(self.metadatas)]
            }
        else:
            if len(self.embeddings) > 0:
                embeddings_array = np.array(self.embeddings)
                query_norm = np.linalg.norm(query_vec)
                embeddings_norms = np.linalg.norm(embeddings_array, axis=1)
                
                similarities = np.dot(embeddings_array, query_vec) / (embeddings_norms * query_norm)
                top_indices = np.argsort(similarities)[-search_k:][::-1]
                
                initial_results = {
                    'documents': [self.texts[i] for i in top_indices],
                    'distances': [similarities[i] for i in top_indices],
                    'metadatas': [self.metadatas[i] for i in top_indices]
                }
            else:
                return {'documents': [], 'distances': [], 'metadatas': []}
        
        # Apply metadata filters
        if category_filter or title_filter:
            filtered_docs = []
            filtered_distances = []
            filtered_metadatas = []
            
            for doc, distance, metadata in zip(initial_results['documents'], 
                                             initial_results['distances'], 
                                             initial_results['metadatas']):
                if category_filter and metadata.get('category') not in category_filter:
                    continue
                
                if title_filter and title_filter.lower() not in metadata.get('title', '').lower():
                    continue
                
                filtered_docs.append(doc)
                filtered_distances.append(distance)
                filtered_metadatas.append(metadata)
                
                if len(filtered_docs) >= k:
                    break
            
            return {
                'documents': filtered_docs,
                'distances': filtered_distances,
                'metadatas': filtered_metadatas
            }
        
        return {
            'documents': initial_results['documents'][:k],
            'distances': initial_results['distances'][:k],
            'metadatas': initial_results['metadatas'][:k]
        }

# ================================
# ARIMA ANALYSIS FUNCTIONALITY
# ================================

def test_stationarity(series: pd.Series) -> dict:
    """Enhanced stationarity testing"""
    result = adfuller(series.dropna(), autolag='AIC')
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05,
        'n_lags': result[2]
    }

def make_stationary(series: pd.Series) -> tuple:
    """Enhanced stationarity transformation"""
    original_result = test_stationarity(series)
    
    if original_result['is_stationary']:
        return series, 0, "Original series is stationary"
    
    diff_1 = series.diff().dropna()
    diff_1_result = test_stationarity(diff_1)
    
    if diff_1_result['is_stationary']:
        return diff_1, 1, "Achieved stationarity with first differencing"
    
    diff_2 = diff_1.diff().dropna()
    diff_2_result = test_stationarity(diff_2)
    
    if diff_2_result['is_stationary']:
        return diff_2, 2, "Achieved stationarity with second differencing"
    else:
        return diff_2, 2, "Series may need additional transformation"

def find_optimal_arima(series: pd.Series, max_p: int = 5, max_q: int = 5) -> tuple:
    """Enhanced ARIMA parameter optimization"""
    stationary_series, d, stationarity_msg = make_stationary(series)
    st.info(f"📊 {stationarity_msg}")
    
    best_aic = np.inf
    best_order = None
    results = []
    
    progress_bar = st.progress(0)
    total_combinations = (max_p + 1) * (max_q + 1) - 1
    current_combination = 0
    
    st.info(f"🔍 Testing ARIMA models with d={d}...")
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                bic = fitted_model.bic
                
                results.append({
                    'order': (p, d, q),
                    'aic': aic,
                    'bic': bic
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                
            except Exception as e:
                st.warning(f"Failed to fit ARIMA({p},{d},{q}): {e}")
                continue
            
            current_combination += 1
            progress_bar.progress(current_combination / total_combinations)
    
    progress_bar.empty()
    
    if best_order is None:
        st.warning("⚠️ Could not find suitable ARIMA model. Using default (1,1,1)")
        best_order = (1, 1, 1)
    
    if results:
        results_df = pd.DataFrame(results).sort_values('aic').head(5)
        st.subheader("🏆 Top 5 ARIMA Models")
        st.dataframe(results_df.round(2))
    
    return best_order, best_aic, results

def run_comprehensive_backtest(data: pd.Series, order: tuple, test_size: float = 0.2) -> dict:
    """Optimized backtesting - fit model once and predict iteratively"""
    split_point = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    predictions = []
    actuals = []
    confidence_intervals = []
    
    st.info(f"🔄 Running optimized backtest on {len(test_data)} test samples...")
    progress_bar = st.progress(0)
    
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        
        history = train_data.copy()
        
        for i, (date, actual) in enumerate(test_data.items()):
            try:
                temp_model = ARIMA(history, order=order)
                temp_fitted = temp_model.fit()
                
                forecast_result = temp_fitted.get_forecast(steps=1)
                pred_value = forecast_result.predicted_mean.iloc[0]
                conf_int = forecast_result.conf_int().iloc[0]
                
                predictions.append(pred_value)
                actuals.append(actual)
                confidence_intervals.append((conf_int.iloc[0], conf_int.iloc[1]))
                
                history = pd.concat([history, pd.Series([actual], index=[date])])
                
            except Exception as e:
                st.warning(f"⚠️ Backtest prediction failed at step {i}: {e}. Using last value as fallback.")
                predictions.append(actuals[-1] if actuals else actual)
                actuals.append(actual)
                confidence_intervals.append((actual*0.95, actual*1.05))
            
            progress_bar.progress((i + 1) / len(test_data))
        
    except Exception as e:
        st.error(f"❌ Model fitting failed: {e}")
        return {}
    
    progress_bar.empty()
    
    actuals_array = np.array(actuals)
    predictions_array = np.array(predictions)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals_array - predictions_array) / actuals_array)) * 100
    
    actual_directions = np.diff(actuals) > 0
    pred_directions = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_directions == pred_directions) * 100
    
    naive_forecast = actuals[:-1]
    actual_changes = actuals[1:]
    predicted_changes = predictions[1:]
    naive_mse = mean_squared_error(actual_changes, naive_forecast)
    forecast_mse = mean_squared_error(actual_changes, predicted_changes)
    theil_u = np.sqrt(forecast_mse) / np.sqrt(naive_mse) if naive_mse > 0 else np.inf
    
    in_interval = [(a >= ci[0] and a <= ci[1]) for a, ci in zip(actuals, confidence_intervals)]
    coverage_ratio = np.mean(in_interval) * 100
    
    results_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual': actuals,
        'Predicted': predictions,
        'Lower_CI': [ci[0] for ci in confidence_intervals],
        'Upper_CI': [ci[1] for ci in confidence_intervals],
        'Error': np.array(actuals) - np.array(predictions),
        'Error_Pct': ((np.array(actuals) - np.array(predictions)) / np.array(actuals)) * 100
    })
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy,
        'R2': r2_score(actuals, predictions),
        'Theil_U': theil_u,
        'Coverage_Ratio': coverage_ratio,
        'Mean_Error': np.mean(results_df['Error']),
        'Std_Error': np.std(results_df['Error'])
    }
    
    return {
        'results_df': results_df,
        'metrics': metrics,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'order': order
    }

# ================================
# SIDEBAR STOCK SELECTOR
# ================================

def show_sidebar_stock_selector():
    """Central stock selector that automatically loads data"""
    st.sidebar.subheader("📊 Stock Selector")
    
    available_files = st.session_state.available_files
    
    if available_files:
        stock_options = {f['symbol']: f for f in available_files}
        stock_symbols = list(stock_options.keys())
        
        current_selection = st.sidebar.selectbox(
            "Select Stock:",
            options=stock_symbols,
            key="selected_stock_main"
        )
        
        if current_selection and (
            not st.session_state.selected_stocks or 
            st.session_state.selected_stocks[0] != current_selection or 
            st.session_state.current_dataset is None
        ):
            file_info = stock_options[current_selection]
            
            with st.spinner(f"Loading {current_selection} data..."):
                stock_data = load_stock_csv(file_info['filepath'])
                
                if not stock_data.empty:
                    enhanced_data = create_comprehensive_features(stock_data)
                    st.session_state.current_dataset = enhanced_data
                    st.session_state.selected_stocks = [current_selection]
                    
                    context_key = f"{current_selection}_{hash(str(enhanced_data.index[-1]))}"
                    st.session_state.ai_context_cache[context_key] = generate_ai_context(enhanced_data, current_selection)
                    
                    st.sidebar.success(f"✅ Loaded {len(stock_data)} records")
        
        if current_selection and st.session_state.current_dataset is not None:
            file_info = stock_options[current_selection]
            st.sidebar.info(f"""
**📈 {current_selection}**
Company: {file_info.get('Company', 'N/A')}
Sector: {file_info.get('Sector', 'N/A')}
Records: {file_info['total_rows']:,}
Period: {file_info['start_date'].strftime('%Y-%m-%d')} to {file_info['end_date'].strftime('%Y-%m-%d')}
""")
    else:
        st.sidebar.info("No stock data found. Please add CSV files to the 'cleaned' folder.")

# ================================
# ENHANCED STOCK ANALYSIS FUNCTION
# ================================

def show_stock_analysis():
    """Enhanced Stock Analysis with advanced controls, comparison, and export features"""
    st.markdown('<div class="section-header">📊 Advanced Stock Analysis & Comparison</div>', unsafe_allow_html=True)
    
    # Validation - Check if any stock data is available
    if not st.session_state.available_files:
        st.error("❌ No stock data found. Please add CSV files to the 'cleaned' folder.")
        return
    
    available_symbols = [f['symbol'] for f in st.session_state.available_files]
    
    # ================================
    # USER CONTROLS AT THE TOP
    # ================================
    
    st.subheader("🎛️ Analysis Controls")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock comparison multiselect (max 3)
        default_stocks = [st.session_state.selected_stocks[0]] if st.session_state.selected_stocks else [available_symbols[0]]
        selected_symbols = st.multiselect(
            "Select stocks for analysis (max 3):",
            options=available_symbols,
            default=default_stocks,
            max_selections=3,
            help="Choose 1-3 stocks to analyze and compare"
        )
    
    with col2:
        # Technical indicators checkboxes
        st.write("**Technical Indicators:**")
        show_sma = st.checkbox("Moving Averages (SMA)", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_bollinger = st.checkbox("Bollinger Bands", value=True)
        show_volume = st.checkbox("Volume Analysis", value=True)
    
    # Date range picker
    col1, col2 = st.columns(2)
    
    # Get date range from available data
    if selected_symbols:
        all_dates = []
        for symbol in selected_symbols:
            symbol_file = next((f for f in st.session_state.available_files if f['symbol'] == symbol), None)
            if symbol_file:
                all_dates.extend([symbol_file['start_date'], symbol_file['end_date']])
        
        if all_dates:
            min_date = min(all_dates).date()
            max_date = max(all_dates).date()
        else:
            min_date = datetime.now().date() - timedelta(days=365)
            max_date = datetime.now().date()
    else:
        min_date = datetime.now().date() - timedelta(days=365)
        max_date = datetime.now().date()
    
    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Validation for selected stocks and date range
    if not selected_symbols:
        st.warning("⚠️ Please select at least one stock for analysis.")
        return
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return
    
    # ================================
    # DATA LOADING AND PROCESSING
    # ================================
    
    with st.spinner("Loading and processing stock data..."):
        # Load data for all selected symbols
        data_dict = {}
        
        for symbol in selected_symbols:
            data = get_data_slice(symbol, pd.Timestamp(start_date), pd.Timestamp(end_date))
            if not data.empty:
                data_dict[symbol] = data
            else:
                st.warning(f"⚠️ No data available for {symbol} in the selected date range.")
        
        if not data_dict:
            st.error("❌ No data available for the selected stocks and date range.")
            return
    
    # ================================
    # EXECUTIVE SUMMARY
    # ================================
    
    executive_summary = generate_executive_summary(selected_symbols, data_dict)
    st.markdown(f"""
<div class="executive-summary">
<h4>📋 Executive Summary</h4>
<p>{executive_summary}</p>
</div>
""", unsafe_allow_html=True)
    
    # ================================
    # EXPORT OPTIONS
    # ================================
    
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Data as CSV"):
            if len(data_dict) == 1:
                symbol = list(data_dict.keys())[0]
                csv_data = data_dict[symbol]
                csv_str = csv_data.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name=f"{symbol}_analysis_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
            else:
                # Multiple stocks - create combined CSV
                combined_data = pd.DataFrame()
                for symbol, data in data_dict.items():
                    data_copy = data.copy()
                    data_copy['Symbol'] = symbol
                    combined_data = pd.concat([combined_data, data_copy])
                
                csv_str = combined_data.to_csv()
                st.download_button(
                    label="Download Combined CSV",
                    data=csv_str,
                    file_name=f"multi_stock_analysis_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📈 Generate Technical Report"):
            with st.spinner("Generating AI-powered technical report..."):
                report = generate_technical_report(selected_symbols, data_dict)
                
                # Display report
                st.subheader("📋 Technical Analysis Report")
                st.markdown(report)
                
                # Download button for report
                st.download_button(
                    label="Download Report as Text",
                    data=report,
                    file_name=f"technical_report_{'-'.join(selected_symbols)}_{start_date}.txt",
                    mime="text/plain"
                )
    
    with col3:
        if st.button("🖼️ Export Charts"):
            st.info("Chart export functionality will be available after viewing the charts below.")
    
    # ================================
    # ANALYSIS TABS
    # ================================
    
    # Package technical indicators into a dict for passing to functions
    technical_indicators = {
        'show_sma': show_sma,
        'show_rsi': show_rsi,
        'show_macd': show_macd,
        'show_bollinger': show_bollinger,
        'show_volume': show_volume
    }
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Analysis", "⚖️ Risk Analysis", "📉 Performance"])
    
    with tab1:
        show_enhanced_stock_overview(data_dict, technical_indicators)
    
    with tab2:
        show_enhanced_technical_analysis(data_dict, technical_indicators)
    
    with tab3:
        show_enhanced_risk_analysis(data_dict)
    
    with tab4:
        show_enhanced_performance_analysis(data_dict)

# ================================
# ENHANCED ANALYSIS HELPER FUNCTIONS
# ================================

def show_enhanced_stock_overview(data_dict: Dict[str, pd.DataFrame], indicators: Dict[str, bool]):
    """Enhanced stock overview with comparison support"""
    
    if len(data_dict) == 1:
        # Single stock overview
        symbol, data = next(iter(data_dict.items()))
        latest = data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price", 
                f"${latest['Close']:.2f}", 
                f"{latest.get('Returns', 0)*100:.2f}%" if 'Returns' in data.columns else None
            )
        
        with col2:
            if 'RSI' in data.columns and indicators['show_rsi']:
                rsi_value = latest['RSI']
                st.metric(
                    "RSI", 
                    f"{rsi_value:.1f}", 
                    "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                )
        
        with col3:
            if 'Volume_Ratio' in data.columns and indicators['show_volume']:
                vol_ratio = latest['Volume_Ratio']
                st.metric(
                    "Volume Ratio", 
                    f"{vol_ratio:.2f}x", 
                    "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
                )
        
        with col4:
            if 'Volatility_20D' in data.columns:
                volatility = latest['Volatility_20D']
                st.metric(
                    "20D Volatility", 
                    f"{volatility:.1%}",
                    "High" if volatility > 0.3 else "Low" if volatility < 0.15 else "Medium"
                )
        
        # Single stock price chart
        fig = go.Figure()
        
        # Downsample for performance
        plot_data = downsample_for_plotting(data)
        
        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=plot_data['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add moving averages if enabled
        if indicators['show_sma']:
            for ma in [20, 50]:
                if f'SMA_{ma}' in plot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_data.index,
                        y=plot_data[f'SMA_{ma}'],
                        mode='lines',
                        name=f'SMA {ma}',
                        line=dict(width=1, dash='dash'),
                        visible='legendonly'
                    ))
        
        fig.update_layout(
            title=f'{symbol} - Price Overview',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            xaxis=dict(rangeslider=dict(visible=True))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Multiple stocks comparison
        st.subheader("📊 Multi-Stock Comparison")
        
        # Comparison metrics table
        comparison_data = []
        for symbol, data in data_dict.items():
            latest = data.iloc[-1]
            comparison_data.append({
                'Symbol': symbol,
                'Price': f"${latest['Close']:.2f}",
                'Daily Change': f"{latest.get('Returns', 0)*100:.2f}%",
                'RSI': f"{latest.get('RSI', 0):.1f}" if 'RSI' in data.columns else "N/A",
                'Volume Ratio': f"{latest.get('Volume_Ratio', 0):.2f}x" if 'Volume_Ratio' in data.columns else "N/A",
                'Volatility (20D)': f"{latest.get('Volatility_20D', 0):.1%}" if 'Volatility_20D' in data.columns else "N/A"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Small multiples chart (one per symbol)
        for i, (symbol, data) in enumerate(data_dict.items()):
            fig = go.Figure()
            
            plot_data = downsample_for_plotting(data)
            
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data['Close'],
                mode='lines',
                name=f'{symbol}',
                line=dict(width=2)
            ))
            
            if indicators['show_sma'] and 'SMA_20' in plot_data.columns:
                fig.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(width=1, dash='dash'),
                    visible='legendonly'
                ))
            
            fig.update_layout(
                title=f'{symbol} - Price Movement',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_enhanced_technical_analysis(data_dict: Dict[str, pd.DataFrame], indicators: Dict[str, bool]):
    """Enhanced technical analysis with configurable indicators"""
    
    for symbol, data in data_dict.items():
        st.subheader(f"📈 Technical Analysis - {symbol}")
        
        plot_data = downsample_for_plotting(data)
        
        # Main price chart
        fig = go.Figure()
        
        # Candlestick chart if OHLC data available
        if all(col in plot_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig.add_trace(go.Candlestick(
                x=plot_data.index,
                open=plot_data['Open'],
                high=plot_data['High'],
                low=plot_data['Low'],
                close=plot_data['Close'],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
        
        # Moving averages
        if indicators['show_sma']:
            for ma in [20, 50, 200]:
                if f'SMA_{ma}' in plot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_data.index,
                        y=plot_data[f'SMA_{ma}'],
                        mode='lines',
                        name=f'SMA {ma}',
                        line=dict(width=1)
                    ))
        
        # Bollinger Bands
        if indicators['show_bollinger'] and all(col in plot_data.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dot'),
                visible='legendonly'
            ))
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                visible='legendonly'
            ))
        
        fig.update_layout(
            title=f'{symbol} - Price & Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            xaxis=dict(rangeslider=dict(visible=True))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Secondary indicators
        secondary_indicators = []
        
        if indicators['show_rsi'] and 'RSI' in plot_data.columns:
            secondary_indicators.append(('RSI', plot_data['RSI'], 'RSI'))
        
        if indicators['show_macd'] and 'MACD' in plot_data.columns:
            secondary_indicators.append(('MACD', plot_data['MACD'], 'MACD'))
        
        if secondary_indicators:
            for name, series, title in secondary_indicators:
                fig_secondary = go.Figure()
                
                fig_secondary.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=series,
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ))
                
                # Add reference lines for RSI
                if name == 'RSI':
                    fig_secondary.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                    fig_secondary.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                
                fig_secondary.update_layout(
                    title=f'{symbol} - {title}',
                    xaxis_title='Date',
                    yaxis_title=title,
                    hovermode='x unified',
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig_secondary, use_container_width=True)
        
        # Volume analysis
        if indicators['show_volume'] and 'Volume' in plot_data.columns:
            fig_volume = go.Figure()
            
            fig_volume.add_trace(go.Bar(
                x=plot_data.index,
                y=plot_data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            
            if 'Volume_SMA_20' in plot_data.columns:
                fig_volume.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data['Volume_SMA_20'],
                    mode='lines',
                    name='Volume SMA 20',
                    line=dict(color='orange', width=2)
                ))
            
            fig_volume.update_layout(
                title=f'{symbol} - Volume Analysis',
                xaxis_title='Date',
                yaxis_title='Volume',
                hovermode='x unified',
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)

def show_enhanced_risk_analysis(data_dict: Dict[str, pd.DataFrame]):
    """Enhanced risk analysis with comparison support"""
    
    if len(data_dict) == 1:
        # Single stock risk analysis
        symbol, data = next(iter(data_dict.items()))
        st.subheader(f"⚖️ Risk Analysis - {symbol}")
        
        if 'Returns' in data.columns:
            returns = data['Returns'].dropna()
            
            # Calculate comprehensive risk metrics
            volatility = returns.std() * np.sqrt(252)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (returns.mean() * 252) / downside_volatility if downside_volatility > 0 else 0
            
            # Maximum drawdown calculation
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            risk_stats = {
                'Metric': [
                    'Annual Volatility', 'Downside Volatility', 'Skewness', 'Kurtosis', 
                    'Value at Risk (95%)', 'Value at Risk (99%)', 'Conditional VaR (95%)',
                    'Maximum Drawdown', 'Sharpe Ratio', 'Sortino Ratio'
                ],
                'Value': [
                    f"{volatility*100:.2f}%", f"{downside_volatility*100:.2f}%", 
                    f"{skewness:.3f}", f"{kurtosis:.3f}", f"{var_95*100:.2f}%", 
                    f"{var_99*100:.2f}%", f"{returns[returns <= var_95].mean()*100:.2f}%",
                    f"{max_drawdown*100:.2f}%",
                    f"{(returns.mean() / returns.std() * np.sqrt(252)):.2f}",
                    f"{sortino_ratio:.2f}"
                ],
                'Description': [
                    'Annualized standard deviation of returns',
                    'Volatility of negative returns only',
                    'Measure of return distribution asymmetry',
                    'Measure of return distribution tail risk',
                    '5% worst-case daily loss',
                    '1% worst-case daily loss',
                    'Average loss beyond VaR 95%',
                    'Peak-to-trough decline',
                    'Risk-adjusted return measure',
                    'Downside risk-adjusted return'
                ]
            }
            
            risk_df = pd.DataFrame(risk_stats)
            st.dataframe(risk_df, use_container_width=True)
            
            # Drawdown chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=drawdown * 100,
                mode='lines',
                name='Drawdown %',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'{symbol} - Drawdown Analysis',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Multiple stocks risk comparison
        st.subheader("⚖️ Comparative Risk Analysis")
        
        risk_comparison = []
        for symbol, data in data_dict.items():
            if 'Returns' in data.columns:
                returns = data['Returns'].dropna()
                volatility = returns.std() * np.sqrt(252)
                sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                var_95 = returns.quantile(0.05)
                
                # Maximum drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                risk_comparison.append({
                    'Symbol': symbol,
                    'Annual Volatility': f"{volatility*100:.2f}%",
                    'Sharpe Ratio': f"{sharpe:.2f}",
                    'VaR (95%)': f"{var_95*100:.2f}%",
                    'Max Drawdown': f"{max_drawdown*100:.2f}%"
                })
        
        risk_df = pd.DataFrame(risk_comparison)
        st.dataframe(risk_df, use_container_width=True)

def show_enhanced_performance_analysis(data_dict: Dict[str, pd.DataFrame]):
    """Enhanced performance analysis with comparison support"""
    
    if len(data_dict) == 1:
        # Single stock performance
        symbol, data = next(iter(data_dict.items()))
        st.subheader(f"📉 Performance Analysis - {symbol}")
        
        if 'Returns' in data.columns:
            returns = data['Returns'].dropna()
            
            # Performance metrics
            total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            annual_return = (returns.mean() * 252) * 100
            win_rate = (returns > 0).mean() * 100
            avg_win = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
            avg_loss = returns[returns < 0].mean() * 100 if len(returns[returns < 0]) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{total_return:.1f}%")
            
            with col2:
                st.metric("Annual Return", f"{annual_return:.1f}%")
            
            with col3:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col4:
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            
            # Cumulative returns chart
            cumulative_returns = (1 + returns).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title=f'{symbol} - Cumulative Returns',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Multiple stocks performance comparison
        st.subheader("📉 Comparative Performance Analysis")
        
        performance_comparison = []
        
        # Normalize all returns to start from 1.0 for comparison
        fig = go.Figure()
        
        for symbol, data in data_dict.items():
            if 'Returns' in data.columns:
                returns = data['Returns'].dropna()
                
                # Performance metrics
                total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                annual_return = (returns.mean() * 252) * 100
                win_rate = (returns > 0).mean() * 100
                volatility = returns.std() * np.sqrt(252) * 100
                
                performance_comparison.append({
                    'Symbol': symbol,
                    'Total Return': f"{total_return:.1f}%",
                    'Annual Return': f"{annual_return:.1f}%",
                    'Volatility': f"{volatility:.1f}%",
                    'Win Rate': f"{win_rate:.1f}%"
                })
                
                # Add to comparison chart
                cumulative_returns = (1 + returns).cumprod()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
        
        # Performance comparison table
        performance_df = pd.DataFrame(performance_comparison)
        st.dataframe(performance_df, use_container_width=True)
        
        # Comparative returns chart
        fig.update_layout(
            title='Comparative Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ================================
# MAIN APPLICATION
# ================================

def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header">📈 Stock Analytics & AI Platform</h1>', unsafe_allow_html=True)
    st.markdown("Comprehensive analysis of your cleaned stock data with AI-powered insights")
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found in environment variables!")
        st.info("Please create a .env file with your Google API key")
        st.code("GOOGLE_API_KEY=your_api_key_here")
        st.stop()
    
    if not st.session_state.available_files:
        with st.spinner("Scanning cleaned folder for stock data..."):
            st.session_state.available_files = scan_cleaned_folder()
    
    show_sidebar_stock_selector()
    
    st.sidebar.title("Navigation")
    
    pages = {
        'Dashboard': 'dashboard',
        'Stock Analysis': 'analysis', 
        'AI Chat': 'chat',
        'Vector Search': 'vectorsearch',
        'Time Series ARIMA': 'timeseries',
        'Settings': 'settings'
    }
    
    selected_page = st.sidebar.selectbox("Choose a section:", list(pages.keys()))
    page_key = pages[selected_page]
    
    if page_key == 'dashboard':
        show_dashboard()
    elif page_key == 'analysis':
        show_stock_analysis()  # Now using the enhanced version
    elif page_key == 'chat':
        show_chat_interface()
    elif page_key == 'vectorsearch':
        show_vector_search()
    elif page_key == 'timeseries':
        show_timeseries_analysis()
    elif page_key == 'settings':
        show_settings()

def show_dashboard():
    """Enhanced dashboard with automatic data loading"""
    st.markdown('<div class="section-header">📊 Stock Data Overview</div>', unsafe_allow_html=True)
    
    available_files = st.session_state.available_files
    
    if not available_files:
        st.markdown('<div class="warning-box">❌ No stock data found. Please add CSV files to the "cleaned" folder.</div>', unsafe_allow_html=True)
        st.info("""
Expected folder structure:
- cleaned/
  - AAPL_full.csv
  - GOOGL_full.csv  
  - MSFT_full.csv
  - ...

Expected CSV format:
- Date, Symbol, open, high, low, close, volume
- Date should be in YYYY-MM-DD format
""")
        
        if st.button("Refresh File List"):
            st.session_state.available_files = scan_cleaned_folder()
            st.success("File list refreshed!")
        
        return
    
    st.success(f"✅ Found {len(available_files)} stock data files")
    
    cols_per_row = 3
    for i in range(0, len(available_files), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(available_files):
                file_info = available_files[idx]
                
                with col:
                    st.markdown(f"""
<div class="stock-card">
<h4>📈 {file_info['symbol']}</h4>
<p><strong>Company:</strong> {file_info.get('Company', 'N/A')}</p>
<p><strong>Sector:</strong> {file_info.get('Sector', 'N/A')}</p>
<p><strong>Industry:</strong> {file_info.get('Industry', 'N/A')}</p>
<p><strong>Records:</strong> {file_info['total_rows']:,}</p>
<p><strong>Period:</strong><br/>{file_info['start_date'].strftime('%Y-%m-%d')} to<br/>{file_info['end_date'].strftime('%Y-%m-%d')}</p>
<p><strong>Years:</strong> {(file_info['end_date'] - file_info['start_date']).days / 365.25:.1f}</p>
</div>
""", unsafe_allow_html=True)
    
    if st.session_state.selected_stocks and st.session_state.current_dataset is not None:
        st.subheader("📈 Current Selection")
        selected_stock = st.session_state.selected_stocks[0]
        st.success(f"Currently analyzing: **{selected_stock}** with {len(st.session_state.current_dataset):,} records")
        
        stock_data = st.session_state.current_dataset
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data.index[-252:], 
            y=stock_data['Close'].iloc[-252:],
            mode='lines', 
            name=f'{selected_stock} Price',
            line=dict(color='#667eea', width=2)
        ))
        fig.update_layout(
            title=f'{selected_stock} - Last 12 Months',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_chat_interface():
    """Enhanced AI chat interface with stock context"""
    st.markdown('<div class="section-header">🤖 AI Stock Analysis Chat</div>', unsafe_allow_html=True)
    
    if st.session_state.current_dataset is not None and st.session_state.selected_stocks:
        st.markdown(f"""
<div class="info-box">
<strong>Current Context:</strong> Analyzing {', '.join(st.session_state.selected_stocks)}<br/>
<strong>Data Points:</strong> {len(st.session_state.current_dataset):,} records<br/>
<strong>Period:</strong> {st.session_state.current_dataset.index[0].strftime('%Y-%m-%d')} to {st.session_state.current_dataset.index[-1].strftime('%Y-%m-%d')}
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Select a stock from the sidebar for contextualized AI responses.")
    
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "What's the current trend analysis for this stock?",
        "How does the RSI indicator look right now?",
        "What are the key support and resistance levels?",
        "Is this stock currently overbought or oversold?",
        "What does the volume analysis suggest?",
        "How risky is this stock based on volatility?",
        "What's the recommendation based on technical indicators?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"💬 {question}", key=f"sample_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                
                stock_symbol = st.session_state.selected_stocks[0] if st.session_state.selected_stocks else None
                response = get_ai_response(question, st.session_state.current_dataset, stock_symbol)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me about your stock data or any financial question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                stock_symbol = st.session_state.selected_stocks[0] if st.session_state.selected_stocks else None
                response = get_ai_response(prompt, st.session_state.current_dataset, stock_symbol)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.sidebar:
        st.subheader("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
        
        if st.session_state.messages:
            chat_json = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                "📥 Export Chat",
                data=chat_json,
                file_name=f"stock_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ================================
# ENHANCED VECTOR SEARCH INTERFACE
# ================================

def show_vector_search():
    """Enhanced vector search with comprehensive document management"""
    st.markdown('<div class="section-header">🔍 Advanced Financial Document Search & Management</div>', unsafe_allow_html=True)
    
    if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
        st.session_state.vector_store = EnhancedVectorStore()
    
    vector_store = st.session_state.vector_store
    
    tab1, tab2, tab3 = st.tabs(["📝 Add Documents", "🔍 Search Documents", "📊 Document Management"])
    
    # TAB 1: Add Documents
    with tab1:
        st.subheader("📝 Add Financial Documents")
        
        input_method = st.radio(
            "Input method:", 
            ["Manual Text", "Stock Analysis", "Sample Documents", "Bulk Upload"]
        )
        
        if input_method == "Manual Text":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                text_input = st.text_area(
                    "Enter financial document or analysis:",
                    height=150,
                    placeholder="Enter stock analysis, earnings report, market commentary, etc."
                )
            
            with col2:
                title_input = st.text_input("Title:")
                category_input = st.selectbox(
                    "Category:", 
                    ["Analysis", "News", "Earnings", "Research", "Market Update", "Other"]
                )
                date_input = st.date_input("Date:", value=pd.Timestamp.now().date())
                source_input = st.text_input("Source (optional):")
            
            if st.button("📝 Add Document") and text_input.strip():
                metadata = {
                    'title': title_input or 'Untitled',
                    'category': category_input,
                    'date': date_input.strftime('%Y-%m-%d'),
                    'source': source_input,
                    'length': len(text_input),
                    'added_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                vector_store.add_documents([text_input.strip()], [metadata])
        
        elif input_method == "Stock Analysis":
            if st.session_state.current_dataset is not None and st.session_state.selected_stocks:
                selected_stock = st.session_state.selected_stocks[0]
                
                if st.button("📊 Generate Technical Analysis Document"):
                    data = st.session_state.current_dataset
                    latest = data.iloc[-1]
                    
                    analysis_text = f"""
Technical Analysis Report: {selected_stock}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

Executive Summary:
{selected_stock} is currently trading at ${latest['Close']:.2f} with a daily change of {latest['Returns']*100:.2f}%. 
The stock shows {"bullish" if latest['Close'] > latest.get('SMA_50', latest['Close']) and latest.get('RSI', 50) < 70 else "bearish" if latest['Close'] < latest.get('SMA_50', latest['Close']) and latest.get('RSI', 50) > 30 else "neutral"} technical signals.

Price Analysis:
- Current Price: ${latest['Close']:.2f}
- Daily Change: {latest['Returns']*100:.2f}%
- 52-Week High: ${data['Close'].max():.2f}
- 52-Week Low: ${data['Close'].min():.2f}
- Distance from 52W High: {((latest['Close'] / data['Close'].max()) - 1)*100:.1f}%

Technical Indicators:
- RSI (14): {latest.get('RSI', 0):.1f} - {"Overbought" if latest.get('RSI', 50) > 70 else "Oversold" if latest.get('RSI', 50) < 30 else "Neutral"}
- Bollinger Band Position: {latest.get('BB_Position', 0.5):.2f} - {"Upper band pressure" if latest.get('BB_Position', 0.5) > 0.8 else "Lower band support" if latest.get('BB_Position', 0.5) < 0.2 else "Middle range"}

Moving Averages Analysis:
- 20-day SMA: ${latest.get('SMA_20', latest['Close']):.2f} - Price is {"above" if latest['Close'] > latest.get('SMA_20', latest['Close']) else "below"} short-term trend
- 50-day SMA: ${latest.get('SMA_50', latest['Close']):.2f} - {"Bullish" if latest['Close'] > latest.get('SMA_50', latest['Close']) else "Bearish"} medium-term trend

Volume Analysis:
- Current Volume Ratio: {latest.get('Volume_Ratio', 1.0):.1f}x average
- Volume trend indicates {"high" if latest.get('Volume_Ratio', 1.0) > 1.5 else "low" if latest.get('Volume_Ratio', 1.0) < 0.5 else "normal"} market interest
- On-Balance Volume trend: {"Positive" if latest.get('OBV', 0) > 0 else "Negative" if latest.get('OBV', 0) < 0 else "Neutral"} accumulation

Risk Assessment:
- 20-day volatility: {latest.get('Volatility_20D', 0)*100:.1f}%
- Price position in 20-day range: {latest.get('Price_Position_20D', 0.5)*100:.0f}%
- Risk Level: {"High" if latest.get('Volatility_20D', 0) > 0.3 else "Low" if latest.get('Volatility_20D', 0) < 0.15 else "Medium"}

Trading Signals:
- Short-term outlook: {"Bullish" if latest['Close'] > latest.get('SMA_20', latest['Close']) else "Bearish"}
- Medium-term outlook: {"Bullish" if latest['Close'] > latest.get('SMA_50', latest['Close']) else "Bearish"}
- Momentum: {"Strong" if abs(latest.get('MACD', 0)) > 0.5 else "Weak"} based on MACD

Recommendation: Based on current technical indicators, {selected_stock} appears to be in a {"bullish" if latest['Close'] > latest.get('SMA_50', latest['Close']) and latest.get('RSI', 50) < 70 else "bearish" if latest['Close'] < latest.get('SMA_50', latest['Close']) and latest.get('RSI', 50) > 30 else "neutral"} phase. 
{"Consider buying on dips" if latest.get('RSI', 50) < 40 else "Consider profit-taking" if latest.get('RSI', 50) > 70 else "Monitor for breakout signals"}.

Disclaimer: This is a technical analysis for educational purposes only and should not be considered as investment advice.
"""
                    
                    metadata = {
                        'title': f'{selected_stock} Technical Analysis',
                        'category': 'Analysis',
                        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                        'symbol': selected_stock,
                        'source': 'Generated Analysis',
                        'length': len(analysis_text),
                        'added_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    vector_store.add_documents([analysis_text], [metadata])
            else:
                st.info("Select a stock from the sidebar to generate technical analysis documents.")
        
        elif input_method == "Sample Documents":
            if st.button("📚 Load Sample Financial Documents"):
                sample_docs = [
                    ("Apple Inc. (AAPL) reported exceptional Q4 earnings with revenue of $394.3B, beating analyst estimates by 5.2%. iPhone sales drove growth despite ongoing supply chain challenges in the semiconductor sector. The company's services segment showed remarkable resilience with 16% YoY growth, indicating strong ecosystem stickiness.", 
                     {'company': 'Apple', 'category': 'Earnings', 'sentiment': 'Positive'}),
                    
                    ("Tesla (TSLA) stock surged 15% after announcing record vehicle deliveries of 936,172 units in Q4, significantly exceeding guidance of 900,000. The electric vehicle manufacturer continues to benefit from strong demand in both domestic and international markets, particularly in China and Europe.", 
                     {'company': 'Tesla', 'category': 'News', 'sentiment': 'Positive'}),
                     
                    ("Federal Reserve Chair Powell signals potential rate cuts in 2024 as inflation shows consistent signs of cooling toward the 2% target. Markets rallied strongly on dovish commentary, with technology stocks leading the charge. Bond yields fell sharply across all maturities.", 
                     {'company': 'Federal Reserve', 'category': 'Market Update', 'sentiment': 'Neutral'}),
                     
                    ("Nvidia (NVDA) continues its AI dominance with data center revenue jumping 217% year-over-year to reach $18.4B. Strong demand for H100 and upcoming H200 chips from major cloud providers and enterprises drives unprecedented growth in the AI infrastructure space.", 
                     {'company': 'Nvidia', 'category': 'Earnings', 'sentiment': 'Positive'}),
                     
                    ("Microsoft (MSFT) Azure cloud revenue grows 30% as enterprise adoption of AI services accelerates rapidly. Copilot integration across Office 365 shows promising early adoption metrics, with over 1 million paid subscribers within the first quarter of launch.", 
                     {'company': 'Microsoft', 'category': 'Earnings', 'sentiment': 'Positive'}),
                     
                    ("Amazon (AMZN) AWS segment demonstrates resilience with 13% growth despite broader economic headwinds affecting cloud spending. Prime membership reaches new record highs of 200 million globally, supporting the retail ecosystem's long-term growth trajectory.", 
                     {'company': 'Amazon', 'category': 'Earnings', 'sentiment': 'Positive'}),
                     
                    ("Market volatility expected to persist through 2024 as geopolitical tensions in Eastern Europe and monetary policy uncertainty continue to weigh heavily on investor sentiment. Defensive sectors outperforming growth stocks in current environment.", 
                     {'company': 'Market', 'category': 'Analysis', 'sentiment': 'Cautious'}),
                     
                    ("Energy sector significantly outperforms broader market as oil prices stabilize above $80/barrel amid OPEC+ production discipline. Chevron (CVX) and ExxonMobil (XOM) lead gains among large-cap energy stocks, benefiting from improved refining margins.", 
                     {'company': 'Energy Sector', 'category': 'Analysis', 'sentiment': 'Positive'})
                ]
                
                texts = [doc[0] for doc in sample_docs]
                metadatas = []
                
                for i, (text, base_metadata) in enumerate(sample_docs):
                    metadata = base_metadata.copy()
                    metadata.update({
                        'title': f"Sample Document {i+1}: {base_metadata['company']}",
                        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                        'source': 'Sample Data',
                        'length': len(text),
                        'added_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    metadatas.append(metadata)
                
                vector_store.add_documents(texts, metadatas)
    
    # TAB 2: Search Documents
    with tab2:
        st.subheader("🔍 Search Financial Documents")
        
        if vector_store.texts:
            st.success(f"📚 {len(vector_store.texts)} document chunks in search index")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_query = st.text_input(
                    "Enter search query:",
                    placeholder="e.g., 'Apple earnings performance' or 'Federal Reserve interest rates'"
                )
            
            with col2:
                search_k = st.slider("Max Results:", 1, min(20, len(vector_store.texts)), 5)
            
            st.subheader("🔧 Advanced Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                available_categories = vector_store.get_all_categories()
                if available_categories:
                    category_filter = st.multiselect(
                        "Filter by Category:",
                        options=available_categories,
                        default=None
                    )
                else:
                    category_filter = None
                    st.info("No categories available yet")
            
            with col2:
                title_filter = st.text_input(
                    "Title contains:",
                    placeholder="Filter by title keywords"
                )
                if not title_filter.strip():
                    title_filter = None
            
            st.write("**Quick Searches:**")
            quick_searches = [
                "earnings performance", "market volatility", "AI technology", 
                "Federal Reserve policy", "stock price movements", "technical analysis"
            ]
            
            cols = st.columns(len(quick_searches))
            for i, search_term in enumerate(quick_searches):
                with cols[i]:
                    if st.button(f"🔎 {search_term}", key=f"quick_search_{i}"):
                        search_query = search_term
            
            if st.button("🔍 Search", type="primary") or search_query:
                if search_query and search_query.strip():
                    with st.spinner("Searching documents..."):
                        results = vector_store.search(
                            search_query.strip(), 
                            k=search_k,
                            category_filter=category_filter,
                            title_filter=title_filter
                        )
                    
                    if results['documents']:
                        st.subheader(f"📄 Search Results ({len(results['documents'])} found)")
                        
                        for i, (doc, score, metadata) in enumerate(zip(
                            results['documents'], results['distances'], results['metadatas']
                        )):
                            with st.expander(f"📄 Result {i+1} - Relevance: {score:.3f}"):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write("**Content:**")
                                    st.write(doc)
                                
                                with col2:
                                    st.write("**Metadata:**")
                                    for key, value in metadata.items():
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.warning("🔍 No relevant documents found. Try different keywords or adjust filters.")
                else:
                    st.info("💡 Enter a search query to find documents.")
        else:
            st.info("📝 Add some financial documents first to enable search functionality.")
    
    # TAB 3: Document Management
    with tab3:
        st.subheader("📊 Document Management")
        
        if vector_store.texts:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Chunks", len(vector_store.texts))
            
            with col2:
                categories = vector_store.get_all_categories()
                st.metric("Categories", len(categories))
            
            with col3:
                unique_titles = len(set(meta.get('title', 'Unknown') for meta in vector_store.metadatas))
                st.metric("Documents", unique_titles)
            
            with col4:
                avg_length = np.mean([len(text) for text in vector_store.texts])
                st.metric("Avg Chunk Length", f"{avg_length:.0f} chars")
            
            st.subheader("📋 All Documents")
            
            doc_data = []
            for i, (text, metadata) in enumerate(zip(vector_store.texts, vector_store.metadatas)):
                doc_data.append({
                    'Index': i,
                    'Title': metadata.get('title', 'Untitled')[:50] + ('...' if len(metadata.get('title', '')) > 50 else ''),
                    'Category': metadata.get('category', 'Unknown'),
                    'Date': metadata.get('date', 'Unknown'),
                    'Chunk': f"{metadata.get('chunk_id', 0) + 1}/{metadata.get('total_chunks', 1)}",
                    'Length': metadata.get('chunk_length', len(text)),
                    'Source': metadata.get('source', 'Unknown'),
                    'Preview': text[:100] + ('...' if len(text) > 100 else '')
                })
            
            if doc_data:
                doc_df = pd.DataFrame(doc_data)
                
                items_per_page = 10
                total_pages = (len(doc_df) - 1) // items_per_page + 1
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    current_page = st.selectbox(
                        "Page:", 
                        range(1, total_pages + 1),
                        key="doc_management_page"
                    )
                
                start_idx = (current_page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_df = doc_df.iloc[start_idx:end_idx]
                
                for _, row in page_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.write(f"**{row['Title']}** ({row['Category']}) - {row['Date']}")
                            st.write(f"Chunk {row['Chunk']} | {row['Length']} chars | Source: {row['Source']}")
                            st.write(f"*Preview:* {row['Preview']}")
                        
                        with col2:
                            if st.button(f"🗑️ Delete", key=f"delete_{row['Index']}"):
                                if vector_store.delete_document(row['Index']):
                                    st.experimental_rerun()
                        
                        st.divider()
                
                st.info(f"Showing {len(page_df)} of {len(doc_df)} documents")
            
            st.subheader("🔧 Bulk Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export Data:**")
                if st.button("📥 Export All Documents as JSON"):
                    export_data = {
                        'texts': vector_store.texts,
                        'metadatas': vector_store.metadatas,
                        'exported_at': pd.Timestamp.now().isoformat()
                    }
                    
                    json_str = json.dumps(export_data, indent=2, default=str)
                    st.download_button(
                        "Download JSON",
                        data=json_str,
                        file_name=f"vector_store_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                st.write("**Danger Zone:**")
                if st.checkbox("⚠️ I understand this will delete ALL documents", key="confirm_clear"):
                    if st.button("🗑️ Clear Entire Database", type="secondary"):
                        vector_store.clear_database()
                        st.experimental_rerun()
        
        else:
            st.info("📝 No documents in the database yet. Add some documents first!")
            
            persistence_dir = "vector_store_data"
            if os.path.exists(persistence_dir):
                files = os.listdir(persistence_dir)
                if files:
                    st.info(f"Found {len(files)} persistent files in {persistence_dir}/")
                    for file in files:
                        st.text(f"  - {file}")

def show_timeseries_analysis():
    """Enhanced time series analysis with comprehensive ARIMA modeling"""
    st.markdown('<div class="section-header">📈 Time Series Analysis & Forecasting</div>', unsafe_allow_html=True)
    
    if st.session_state.current_dataset is None:
        st.info("Please select a stock from the sidebar to begin time series analysis.")
        return
    
    data = st.session_state.current_dataset
    
    st.subheader("⚙️ Analysis Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        target_col = st.selectbox("Target variable:", numeric_cols, 
                                 index=numeric_cols.index('Close') if 'Close' in numeric_cols else 0)
    
    with col2:
        test_size = st.slider("Test set size %:", 10, 40, 20) / 100
    
    with col3:
        max_params = st.slider("Max p,q parameters:", 3, 8, 5)
    
    series = data[target_col].dropna()
    
    if len(series) < 50:
        st.warning("⚠️ Series too short for reliable ARIMA analysis (< 50 observations).")
        return
    
    st.subheader("📊 Time Series Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observations", f"{len(series):,}")
    
    with col2:
        st.metric("Date Range", f"{(series.index[-1] - series.index[0]).days} days")
    
    with col3:
        st.metric("Latest Value", f"{series.iloc[-1]:.2f}")
    
    with col4:
        recent_change = (series.iloc[-1] / series.iloc[-30] - 1) * 100 if len(series) >= 30 else 0
        st.metric("30-Day Change", f"{recent_change:.1f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines', name=target_col,
        line=dict(color='#1f77b4', width=1)
    ))
    fig.update_layout(
        title=f'{target_col} Time Series - {st.session_state.selected_stocks[0] if st.session_state.selected_stocks else "Selected Stock"}',
        xaxis_title='Date', yaxis_title=target_col,
        hovermode='x unified', template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🚀 Run Comprehensive ARIMA Analysis", type="primary"):
        with st.spinner("Finding optimal ARIMA parameters..."):
            best_order, best_aic, all_results = find_optimal_arima(series, max_params, max_params)
            st.success(f"✅ Optimal ARIMA order: {best_order} (AIC: {best_aic:.2f})")
            
            st.info("Running comprehensive backtest...")
            backtest_results = run_comprehensive_backtest(series, best_order, test_size)
            
            if backtest_results:
                st.session_state.arima_results = {
                    'order': best_order,
                    'aic': best_aic,
                    'backtest': backtest_results,
                    'target_column': target_col,
                    'symbol': st.session_state.selected_stocks[0] if st.session_state.selected_stocks else "Unknown",
                    'all_results': all_results
                }
    
    if 'arima_results' in st.session_state and st.session_state.arima_results is not None:
        results = st.session_state.arima_results
        
        st.subheader("📊 ARIMA Model Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Order", str(results['order']))
        with col2:
            st.metric("AIC Score", f"{results['aic']:.2f}")
        with col3:
            st.metric("Target", results['target_column'])
        with col4:
            st.metric("Stock", results['symbol'])
        
        results_df = results['backtest']['results_df']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Actual'],
            mode='lines', name='Actual',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Predicted'],
            mode='lines', name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Upper_CI'],
            mode='lines', name='Upper CI',
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Lower_CI'],
            mode='lines', name='Lower CI',
            line=dict(width=0), fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)', showlegend=False
        ))
        
        fig.update_layout(
            title=f'ARIMA{results["order"]} Backtest Results - {results["symbol"]}',
            xaxis_title='Date', yaxis_title=results['target_column'],
            hovermode='x unified', template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        metrics = results['backtest']['metrics']
        
        st.subheader("📈 Backtest Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
        with col2:
            st.metric("MAE", f"{metrics['MAE']:.4f}")
        with col3:
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        with col4:
            st.metric("Directional Accuracy", f"{metrics['Directional_Accuracy']:.1f}%")
        with col5:
            st.metric("Coverage Ratio", f"{metrics['Coverage_Ratio']:.1f}%")

def show_settings():
    """Enhanced settings and system information"""
    st.markdown('<div class="section-header">⚙️ Settings & System Information</div>', unsafe_allow_html=True)
    
    st.subheader("🔑 API Configuration")
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    if api_key:
        masked_key = api_key[:8] + "*" * max(0, len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else api_key
        st.success(f"✅ Google API Key configured: {masked_key}")
        
        if st.button("🧪 Test API Connection"):
            with st.spinner("Testing API..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content("Hello! This is a test.")
                    st.success("✅ API connection successful!")
                    st.info(f"Response: {response.text[:100]}...")
                except Exception as e:
                    st.error(f"❌ API test failed: {e}")
    else:
        st.error("❌ Google API Key not found")
        st.info("""
To set up your API key:
1. Get your key from https://makersuite.google.com/
2. Create a .env file in your project folder  
3. Add: GOOGLE_API_KEY=your_api_key_here
""")
    
    st.subheader("💻 System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Libraries:**")
        libraries = {
            'FAISS': FAISS_AVAILABLE,
            'ChromaDB': CHROMA_AVAILABLE,
            'Streamlit': True,
            'Plotly': True,
            'Pandas': True,
            'NumPy': True,
            'Google Generative AI': True,
            'Statsmodels': True,
            'Scikit-learn': True
        }
        
        for lib, available in libraries.items():
            status = "✅" if available else "❌"
            st.write(f"{lib}: {status}")
    
    with col2:
        st.write("**Session State:**")
        session_info = {
            'Current Dataset': st.session_state.current_dataset is not None,
            'Selected Stocks': len(st.session_state.selected_stocks) > 0,
            'Available Files': len(st.session_state.available_files),
            'Vector Store': st.session_state.vector_store and len(st.session_state.vector_store.texts) > 0,
            'Chat Messages': len(st.session_state.messages),
            'ARIMA Results': st.session_state.arima_results is not None
        }
        
        for item, status in session_info.items():
            if isinstance(status, bool):
                icon = "✅" if status else "❌"
                st.write(f"{item}: {icon}")
            else:
                st.write(f"{item}: {status}")
    
    st.subheader("📊 Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Session:**")
        if st.session_state.current_dataset is not None:
            data_info = st.session_state.current_dataset
            st.write(f"Records: {len(data_info):,}")
            st.write(f"Columns: {len(data_info.columns)}")
            st.write(f"Memory usage: {data_info.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            missing_values = data_info.isnull().sum().sum()
            st.write(f"Missing values: {missing_values:,}")
        else:
            st.write("No dataset loaded")
    
    with col2:
        st.write("**Actions:**")
        if st.button("🔄 Refresh File List"):
            st.session_state.available_files = scan_cleaned_folder()
            st.success("File list refreshed!")
        
        if st.button("🗑️ Clear All Session Data"):
            if st.checkbox("I understand this will clear all data"):
                st.session_state.current_dataset = None
                st.session_state.selected_stocks = []
                st.session_state.vector_store = None
                st.session_state.arima_results = None
                st.session_state.messages = []
                st.success("All session data cleared")
    
    st.subheader("🚀 Performance Tips")
    st.info("""
To optimize performance:
- Use smaller datasets for initial testing
- Limit ARIMA parameter ranges (p,q ≤ 5)  
- Process embeddings in small batches
- Clear session data when switching between different analyses
- Use the "Clear All Session Data" button if the app becomes slow
""")

if __name__ == "__main__":
    main()
    