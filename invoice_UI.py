# =========================
# Invoice Extractor (Qwen3-VL via RunPod vLLM) - Batch Mode with Tax Validation
# UPDATED: Fixed tax percentage parsing + time format quantities + scrolling issue
# =========================
import os
from pathlib import Path

# -----------------------------
# Environment hardening (HF Spaces, /.cache issue)
# -----------------------------
_home = os.environ.get("HOME", "")
if _home in ("", "/", None):
    repo_dir = os.getcwd()
    safe_home = repo_dir if os.access(repo_dir, os.W_OK) else "/tmp"
    os.environ["HOME"] = safe_home
    print(f"[startup] HOME not set or unwritable — setting HOME={safe_home}")

streamlit_dir = Path(os.environ["HOME"]) / ".streamlit"
try:
    streamlit_dir.mkdir(parents=True, exist_ok=True)
    print(f"[startup] ensured {streamlit_dir}")
except Exception as e:
    print(f"[startup] WARNING: could not create {streamlit_dir}: {e}")

# -----------------------------
# Imports
# -----------------------------
import json
from io import BytesIO
import hashlib
from typing import Dict, Any
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

# Optional: pdf2image is only needed for PDFs
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# -----------------------------
# RunPod vLLM Configuration (from environment variables)
# -----------------------------
import requests
import base64
import re

POD_URL = os.getenv("POD_URL", "")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
MODEL_NAME = "qwen-v2-merged"

st.set_page_config(page_title="Invoice Extractor (Qwen3-VL) - Batch Mode", layout="wide")

# Validate secrets are set
if not POD_URL or not VLLM_API_KEY:
    st.error("⚠️ API credentials not configured. Please set POD_URL and VLLM_API_KEY in Space settings.")
    st.stop()
# -----------------------------
# Page config & CSS
# -----------------------------

st.title("Invoice Extraction")

st.markdown(
    """
    <style>
        .stApp { background-color: #ECECEC !important; }
        div.block-container { padding-top: 3rem; padding-bottom: 1rem; }
        [data-testid="stSidebar"] { background-color: #F7F7F7 !important; }
        div[data-testid="stTabs"] > div > div { padding-bottom: 6px !important; }
        /* Keep right column steady on first render post-extraction */
        [data-testid="column"]:nth-of-type(2) { min-height: 780px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Fixed sizes to prevent reflow wobble
FIXED_IMG_WIDTH = 640
DATA_EDITOR_HEIGHT = 380

# -----------------------------
# Helpers
# -----------------------------
def ensure_state(k: str, default):
    """Initialize a session_state key once, then let widgets bind to it via key=... (no value=...)."""
    if k not in st.session_state:
        st.session_state[k] = default

def parse_time_to_minutes(x) -> float:
    """
    Parse time format quantities to minutes.
    
    Examples:
        "0:35"  → 35.0   (0 hours, 35 minutes = 35 minutes)
        "1:30"  → 90.0   (1 hour, 30 minutes = 90 minutes)
        "2:15"  → 135.0  (2 hours, 15 minutes = 135 minutes)
        "0:05"  → 5.0    (5 minutes)
        "123"   → 123.0  (regular number, not time format)
        "1.5"   → 1.5    (regular decimal, not time format)
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    
    s = str(x).strip()
    if s == "":
        return 0.0
    
    # Check if it's in time format (H:MM or HH:MM)
    time_pattern = r'^(\d+):(\d{1,2})$'
    match = re.match(time_pattern, s)
    
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        total_minutes = (hours * 60) + minutes
        return float(total_minutes)
    
    # Not time format, treat as regular number
    return 0.0

def clean_quantity(x) -> float:
    """
    Parse quantity - handles both time format (H:MM) and regular numbers.
    
    Examples:
        "0:35"     → 35.0   (time format: 35 minutes)
        "1:30"     → 90.0   (time format: 90 minutes)
        "123"      → 123.0  (regular number)
        "1,234.56" → 1234.56 (US format with decimals)
        "1.234,56" → 1234.56 (EUR format with decimals)
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    
    s = str(x).strip()
    if s == "":
        return 0.0
    
    # First check if it's time format (H:MM or HH:MM)
    time_value = parse_time_to_minutes(s)
    if time_value > 0.0:
        return time_value
    
    # Not time format, use regular number parsing
    return clean_float(s)

def clean_tax_percentage(x) -> float:
    """
    Parse tax percentage - ALWAYS treats periods as decimals, never as thousands.
    
    Tax percentages are typically: 8.875, 19.5, 2.75, 0.0875, etc.
    We should NEVER interpret "8.875" as "8875"
    
    Examples:
        "8.875"    → 8.875   (decimal percentage)
        "8,875"    → 8.875   (European decimal)
        "19.5"     → 19.5    (standard decimal)
        "2,75"     → 2.75    (European decimal)
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    
    s = str(x).strip()
    if s == "":
        return 0.0
    
    # Handle negative signs (could be leading or trailing)
    is_negative = False
    if s.startswith('-'):
        is_negative = True
        s = s[1:].strip()
    elif s.endswith('-'):
        is_negative = True
        s = s[:-1].strip()
    elif s.startswith('(') and s.endswith(')'):
        is_negative = True
        s = s[1:-1].strip()
    
    # Remove percent signs, currency symbols and spaces
    s = re.sub(r'[%€$£¥₹\s]', '', s)
    
    if s == "":
        return 0.0
    
    # Count occurrences
    comma_count = s.count(',')
    period_count = s.count('.')
    
    # For tax percentages, logic is simpler:
    # - If both comma and period exist, the LAST one is decimal separator
    # - If only comma exists, it's decimal separator (European style)
    # - If only period exists, it's ALWAYS decimal separator (no thousands logic)
    
    if comma_count > 0 and period_count > 0:
        # Both present - LAST one is decimal
        last_comma = s.rfind(',')
        last_period = s.rfind('.')
        
        if last_comma > last_period:
            # European: 1.234,56 → comma is decimal
            s = s.replace('.', '').replace(',', '.')
        else:
            # US: 1,234.56 → period is decimal
            s = s.replace(',', '')
    
    elif comma_count > 0:
        # Only comma - treat as European decimal separator
        s = s.replace(',', '.')
    
    # If only period(s) exist, keep as-is (always decimal for tax percentages)
    
    # Clean any remaining non-numeric characters except period and minus
    s = re.sub(r'[^\d.]', '', s)
    
    if s == "" or s == ".":
        return 0.0
    
    try:
        result = float(s)
        return -result if is_negative else result
    except ValueError:
        return 0.0

def clean_float(x, currency=None) -> float:
    """
    Parse a number string handling both US and European formats.
    Use this for MONETARY AMOUNTS only, NOT for tax percentages.

    US Format:      1,234,567.89  (comma = thousands, period = decimal)
    European:       1.234.567,89  (period = thousands, comma = decimal)

    Currency-aware for ambiguous cases (3 digits after comma):
        - EUR: "10,000" → 10.0 (European decimal)
        - USD/other: "10,000" → 10000 (thousands separator)

    Examples:
        "1,234.56"    → 1234.56  (US)
        "1.234,56"    → 1234.56  (European)
        "3.000,2234"  → 3000.2234 (European with 4 decimal places)
        "261,49"      → 261.49   (European decimal only)
        "39,22-"      → -39.22   (European with trailing minus)
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    
    s = str(x).strip()
    if s == "":
        return 0.0
    
    # Handle negative signs (could be leading or trailing)
    is_negative = False
    if s.startswith('-'):
        is_negative = True
        s = s[1:].strip()
    elif s.endswith('-'):
        is_negative = True
        s = s[:-1].strip()
    elif s.startswith('(') and s.endswith(')'):
        # Accounting format: (123.45) means negative
        is_negative = True
        s = s[1:-1].strip()
    
    # Remove currency symbols and spaces
    s = re.sub(r'[€$£¥₹\s]', '', s)
    
    if s == "":
        return 0.0
    
    # Count occurrences
    comma_count = s.count(',')
    period_count = s.count('.')
    
    # Find positions of last comma and last period
    last_comma = s.rfind(',')
    last_period = s.rfind('.')
    
    # Determine format based on which separator comes last
    if comma_count > 0 and period_count > 0:
        # Both separators present - the LAST one is the decimal separator
        if last_comma > last_period:
            # European format: 1.234,56 → comma is decimal
            # Remove periods (thousands), replace comma with period
            s = s.replace('.', '').replace(',', '.')
        else:
            # US format: 1,234.56 → period is decimal
            # Remove commas (thousands)
            s = s.replace(',', '')
    
    elif comma_count > 0 and period_count == 0:
        # Only commas present
        # Check what comes after the LAST comma
        after_last_comma = s[last_comma + 1:] if last_comma < len(s) - 1 else ""

        if comma_count == 1 and len(after_last_comma) == 3 and after_last_comma.isdigit():
            # AMBIGUOUS CASE: Single comma with exactly 3 digits after
            # Could be US thousands ("10,000" = 10000) or European decimal ("10,000" = 10.0)
            # Use currency to decide:
            if currency and currency.upper() == 'EUR':
                # European: treat comma as decimal separator
                # "10,000" → 10.000 → 10.0
                s = s.replace(',', '.')
            else:
                # USD/other: treat comma as thousands separator
                # "10,000" → 10000
                s = s.replace(',', '')
        elif comma_count == 1 and len(after_last_comma) <= 4 and after_last_comma.isdigit():
            # Single comma with 1, 2, or 4 digits after → European decimal
            # "261,49" → 261.49, "1234,5678" → 1234.5678
            s = s.replace(',', '.')
        elif len(after_last_comma) == 3 and comma_count >= 1:
            # Multiple commas with 3 digits after last → thousands separator
            # "1,234,567" → 1234567
            s = s.replace(',', '')
        else:
            # Multiple commas → thousands separator
            # "1,234,567" → 1234567
            s = s.replace(',', '')
    
    elif period_count > 0 and comma_count == 0:
        # Only periods present
        # Check what comes after the LAST period
        after_last_period = s[last_period + 1:] if last_period < len(s) - 1 else ""
        
        if period_count > 1:
            # Multiple periods → definitely thousands separator (European: "1.234.567")
            s = s.replace('.', '')
        elif len(after_last_period) == 3 and after_last_period.isdigit():
            # Single period with exactly 3 digits after
            before_period = s[:last_period]
            # Only treat as thousands if there are 2+ digits before period
            # "10.000" or "123.000" → thousands
            # "8.875" or "9.123" → decimal
            if before_period.isdigit() and len(before_period) >= 2 and len(before_period) <= 3:
                s = s.replace('.', '')  # European thousands: "10.000" → 10000
            # Otherwise keep as decimal: "8.875" → 8.875
        # Otherwise keep as is (standard decimal like "1.50", "123.45")
    
    # Clean any remaining non-numeric characters except period and minus
    s = re.sub(r'[^\d.]', '', s)
    
    if s == "" or s == ".":
        return 0.0
    
    try:
        result = float(s)
        return -result if is_negative else result
    except ValueError:
        return 0.0

def normalize_date(date_str, currency=None) -> str:
    """
    Normalize various date formats:
    - Full dates (day-month-year) → dd-MMM-yyyy (e.g., 01-Jan-2025)
    - Month-year only → MMM-yyyy (e.g., Aug-2025)
    
    Currency-aware parsing:
    - If currency is USD and date is numeric format (11/09/2025, 11-09-2025), 
      treat as MM/DD/YYYY
    - For text formats (06-Nov-2025, December 6, 2025), parse normally
    
    Returns empty string if date cannot be parsed
    """
    if not date_str or date_str == "":
        return ""

    if isinstance(date_str, str):
        date_str = date_str.strip()
        if date_str == "":
            return ""
        
        # EXTRA CLEANING: Replace various unicode spaces and clean up
        # Non-breaking space, thin space, etc. → regular space
        date_str = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', date_str)
        # Remove zero-width characters
        date_str = re.sub(r'[\u200B-\u200D\uFEFF]', '', date_str)
        # Normalize multiple spaces to single space
        date_str = re.sub(r'\s+', ' ', date_str).strip()

    # Clean ordinal suffixes FIRST (1st, 2nd, 3rd, 4th, 06th, etc.)
    cleaned_date = date_str
    if isinstance(date_str, str):
        # Handle ordinals: "06th December 2025" → "06 December 2025"
        # Also handles: "December 6th, 2025" → "December 6, 2025"
        cleaned_date = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', date_str, flags=re.IGNORECASE)

    # Check if date is NUMERIC format (contains only digits and separators)
    # Pattern: XX/XX/XXXX, XX-XX-XXXX, XX.XX.XXXX (with 2 or 4 digit year)
    is_numeric_format = bool(re.match(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$', cleaned_date))
    
    # US FORMAT PRIORITY: If currency is USD and date is numeric, try MM/DD/YYYY first
    if currency and currency.upper() == 'USD' and is_numeric_format:
        us_formats = [
            "%m/%d/%Y",           # 01/15/2025
            "%m-%d-%Y",           # 01-15-2025
            "%m.%d.%Y",           # 01.15.2025
            "%m/%d/%y",           # 01/15/25
            "%m-%d-%y",           # 01-15-25
            "%m.%d.%y",           # 01.15.25
        ]
        for fmt in us_formats:
            try:
                parsed_date = datetime.strptime(cleaned_date, fmt)
                return parsed_date.strftime("%d-%b-%Y")
            except (ValueError, TypeError):
                continue

    # FULL DATE FORMATS (day-month-year) - standard parsing
    full_date_formats = [
        # ISO formats (4-digit year) - these are unambiguous
        "%Y-%m-%d",           # 2025-01-15
        "%Y/%m/%d",           # 2025/01/15
        "%Y.%m.%d",           # 2025.01.15
        "%Y %m %d",           # 2025 01 15
        "%Y%m%d",             # 20250115 (compact)
        
        # European formats with full month names (4-digit year) - UNAMBIGUOUS
        "%d %B, %Y",          # 15 December, 2025 (with comma)
        "%d %b, %Y",          # 15 Dec, 2025 (with comma)
        "%d %B %Y",           # 15 January 2025
        "%d %b %Y",           # 15 Jan 2025
        "%d-%B-%Y",           # 15-January-2025
        "%d-%b-%Y",           # 15-Jan-2025
        "%d.%B.%Y",           # 15.January.2025
        "%d.%b.%Y",           # 15.Jan.2025
        "%d/%B/%Y",           # 15/January/2025
        "%d/%b/%Y",           # 15/Jan/2025
        
        # US formats with full month names (4-digit year) - UNAMBIGUOUS
        "%B %d, %Y",          # January 15, 2025
        "%b %d, %Y",          # Jan 15, 2025
        "%B %d %Y",           # January 15 2025
        "%b %d %Y",           # Jan 15 2025
        "%B-%d-%Y",           # January-15-2025
        "%b-%d-%Y",           # Jan-15-2025
        "%B %d,%Y",           # January 15,2025 (no space after comma)
        "%b %d,%Y",           # Jan 15,2025
        
        # European formats - Day first (4-digit year)
        "%d-%m-%Y",           # 15-01-2025
        "%d/%m/%Y",           # 15/01/2025
        "%d.%m.%Y",           # 15.01.2025
        "%d %m %Y",           # 15 01 2025
        
        # US formats - Month first (4-digit year) - only if not USD or not numeric
        "%m-%d-%Y",           # 01-15-2025
        "%m/%d/%Y",           # 01/15/2025
        "%m.%d.%Y",           # 01.15.2025
        "%m %d %Y",           # 01 15 2025
        
        # European formats with 2-digit year - Day first
        "%d-%m-%y",           # 15-01-25
        "%d/%m/%y",           # 15/01/25
        "%d.%m.%y",           # 15.01.25
        "%d %m %y",           # 15 01 25
        
        # US formats with 2-digit year - Month first
        "%m-%d-%y",           # 01-15-25
        "%m/%d/%y",           # 01/15/25
        "%m.%d.%y",           # 01.15.25
        "%m %d %y",           # 01 15 25
        
        # ISO with 2-digit year
        "%y-%m-%d",           # 25-01-15
        "%y/%m/%d",           # 25/01/15
        "%y.%m.%d",           # 25.01.15
        "%y %m %d",           # 25 01 15
        
        # Compact formats with 2-digit year
        "%y%m%d",             # 250115
        "%d%m%y",             # 150125
        "%m%d%y",             # 011525
        
        # European formats with abbreviated month (2-digit year) - UNAMBIGUOUS
        "%d %B, %y",          # 15 December, 25 (with comma)
        "%d %b, %y",          # 15 Dec, 25 (with comma)
        "%d-%b-%y",           # 15-Jan-25
        "%d/%b/%y",           # 15/Jan/25
        "%d.%b.%y",           # 15.Jan.25
        "%d %b %y",           # 15 Jan 25
        "%d-%B-%y",           # 15-January-25
        "%d/%B/%y",           # 15/January/25
        
        # US formats with abbreviated month (2-digit year) - UNAMBIGUOUS
        "%b %d, %y",          # Jan 15, 25
        "%b %d %y",           # Jan 15 25
        "%B %d, %y",          # January 15, 25
        "%B %d %y",           # January 15 25
        "%b-%d-%y",           # Jan-15-25
        "%B-%d-%y",           # January-15-25
        
        # Compact 8-digit formats
        "%d%m%Y",             # 15012025
        "%m%d%Y",             # 01152025
        "%Y%d%m",             # 20251501
    ]

    # Try full date formats → output as dd-MMM-yyyy
    for fmt in full_date_formats:
        try:
            parsed_date = datetime.strptime(cleaned_date, fmt)
            return parsed_date.strftime("%d-%b-%Y")
        except (ValueError, TypeError):
            continue

    # MONTH-YEAR ONLY FORMATS - output as MMM-yyyy
    month_year_formats = [
        # Full month name with year
        "%B %Y",              # August 2025
        "%b %Y",              # Aug 2025
        "%B, %Y",             # August, 2025
        "%b, %Y",             # Aug, 2025
        "%B-%Y",              # August-2025
        "%b-%Y",              # Aug-2025
        "%B/%Y",              # August/2025
        "%b/%Y",              # Aug/2025
        
        # Numeric month-year (4-digit year)
        "%m/%Y",              # 08/2025
        "%m-%Y",              # 08-2025
        "%m.%Y",              # 08.2025
        "%m %Y",              # 08 2025
        "%Y-%m",              # 2025-08
        "%Y/%m",              # 2025/08
        "%Y.%m",              # 2025.08
        "%Y %m",              # 2025 08
        
        # Numeric month-year (2-digit year)
        "%m/%y",              # 08/25
        "%m-%y",              # 08-25
        "%m.%y",              # 08.25
        "%m %y",              # 08 25
        "%y-%m",              # 25-08
        "%y/%m",              # 25/08
        
        # Full month name with 2-digit year
        "%B %y",              # August 25
        "%b %y",              # Aug 25
        "%B-%y",              # August-25
        "%b-%y",              # Aug-25
    ]

    # Try month-year formats → output as MMM-yyyy (no day)
    for fmt in month_year_formats:
        try:
            parsed_date = datetime.strptime(cleaned_date, fmt)
            return parsed_date.strftime("%b-%Y")  # Aug-2025 format
        except (ValueError, TypeError):
            continue

    # If no format matched, return empty string
    return ""

def parse_date_to_object(date_str, currency=None):
    """
    Parse a date string to a datetime.date object for date_input widget
    Currency-aware: If USD and numeric format, treat as MM/DD/YYYY
    Returns None if date cannot be parsed
    """
    if not date_str or date_str == "":
        return None

    if isinstance(date_str, str):
        date_str = date_str.strip()
        if date_str == "":
            return None
        
        # EXTRA CLEANING: Replace various unicode spaces and clean up
        date_str = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', date_str)
        date_str = re.sub(r'[\u200B-\u200D\uFEFF]', '', date_str)
        date_str = re.sub(r'\s+', ' ', date_str).strip()

    # Clean ordinal suffixes FIRST (1st, 2nd, 3rd, 4th, 06th, etc.)
    cleaned_date = str(date_str)
    if isinstance(date_str, str):
        cleaned_date = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', date_str, flags=re.IGNORECASE)

    # Check if date is NUMERIC format (contains only digits and separators)
    is_numeric_format = bool(re.match(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$', cleaned_date))
    
    # US FORMAT PRIORITY: If currency is USD and date is numeric, try MM/DD/YYYY first
    if currency and currency.upper() == 'USD' and is_numeric_format:
        us_formats = [
            "%m/%d/%Y",           # 01/15/2025
            "%m-%d-%Y",           # 01-15-2025
            "%m.%d.%Y",           # 01.15.2025
            "%m/%d/%y",           # 01/15/25
            "%m-%d-%y",           # 01-15-25
            "%m.%d.%y",           # 01.15.25
        ]
        for fmt in us_formats:
            try:
                parsed_date = datetime.strptime(cleaned_date, fmt)
                return parsed_date.date()
            except (ValueError, TypeError):
                continue

    # Standard formats
    formats = [
        # ISO formats (4-digit year)
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y %m %d", "%Y%m%d",
        
        # Text month formats with comma - MUST BE FIRST for "06 December, 2025"
        "%d %B, %Y", "%d %b, %Y",          # 06 December, 2025 / 06 Dec, 2025
        
        # Text month formats - UNAMBIGUOUS
        "%d %B %Y", "%d %b %Y", "%d-%B-%Y", "%d-%b-%Y",
        "%d.%B.%Y", "%d.%b.%Y", "%d/%B/%Y", "%d/%b/%Y",
        "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
        "%B-%d-%Y", "%b-%d-%Y", "%B %d,%Y", "%b %d,%Y",
        
        # European formats - Day first
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %m %Y",
        "%d-%m-%y", "%d/%m/%y", "%d.%m.%y", "%d %m %y",
        
        # US formats - Month first
        "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y", "%m %d %Y",
        "%m-%d-%y", "%m/%d/%y", "%m.%d.%y", "%m %d %y",
        
        # ISO with 2-digit year
        "%y-%m-%d", "%y/%m/%d", "%y.%m.%d", "%y %m %d",
        
        # Compact formats
        "%y%m%d", "%d%m%y", "%m%d%y", "%d%m%Y", "%m%d%Y", "%Y%d%m",
        
        # Text month with 2-digit year (with comma)
        "%d %B, %y", "%d %b, %y",          # 06 December, 25 / 06 Dec, 25
        "%d-%b-%y", "%d/%b/%y", "%d.%b.%y", "%d %b %y",
        "%d-%B-%y", "%d/%B/%y",
        "%b %d, %y", "%b %d %y", "%B %d, %y", "%B %d %y",
        "%b-%d-%y", "%B-%d-%y",
        
        # Month-year only
        "%B %Y", "%b %Y", "%B, %Y", "%b, %Y",
        "%B-%Y", "%b-%Y", "%B/%Y", "%b/%Y",
        "%m/%Y", "%m-%Y", "%m.%Y", "%m %Y",
        "%Y-%m", "%Y/%m", "%Y.%m", "%Y %m",
        "%m/%y", "%m-%y", "%m.%y", "%m %y",
        "%y-%m", "%y/%m",
        "%B %y", "%b %y", "%B-%y", "%b-%y",
    ]

    for fmt in formats:
        try:
            parsed_date = datetime.strptime(cleaned_date, fmt)
            return parsed_date.date()
        except (ValueError, TypeError):
            continue

    return None


# -----------------------------
# vLLM Inference Function (RunPod API)
# -----------------------------
def run_inference_vllm(image: Image.Image):
    """Run inference using RunPod vLLM API"""

    # Extraction prompt (JSON format)
    EXTRACTION_PROMPT = """Please carefully examine this invoice image and extract all the information into the following structured JSON format. Pay close attention to details and ensure accuracy in number formatting and text extraction.
Extract the data into this exact JSON structure:
{
  "header": {
    "invoice_no": "Invoice number or reference ID",
    "invoice_date": "Date the invoice was issued (maintain original format)",
    "due_date": "Payment due date if specified",
    "sender_name": "Name of the company/person issuing the invoice",
    "sender_addr": "Complete address of the sender/issuer",
    "rcpt_name": "Name of the recipient/customer",
    "rcpt_addr": "Address of the recipient/customer",
    "bank_iban": "International Bank Account Number",
    "bank_name": "Name of the bank",
    "bank_acc_no": "Bank account number",
    "bank_routing": "Bank routing number",
    "bank_swift": "SWIFT/BIC code",
    "bank_acc_name": "Account holder name",
    "bank_branch": "Bank branch information"
  },
  "items": [
    {
      "descriptions": "Detailed description of the item/service",
      "SKU": "Stock Keeping Unit or item code",
      "quantity": "Quantity of items",
      "unit_price": "Price per unit",
      "amount": "Total amount for this line item",
      "tax": "Tax for this item",
      "Line_total": "Total amount including tax for this line"
    }
  ],
  "summary": {
    "subtotal": "Subtotal amount before tax",
    "tax_rate": "Tax rate percentage or description",
    "tax_amount": "Total tax amount",
    "total_amount": "Final total amount to be paid",
    "currency": "Currency code (USD, EUR, etc.)"
  }
}
IMPORTANT GUIDELINES:
- Extract only the bank account details matching the invoice currency.
  Example:
  Invoice currency = USD → extract the USD bank account.
  Invoice currency = GBP → extract the GBP bank account.
  - REQUIRED: Always populate "bank_acc_name". RULES:
1) If a field explicitly labeled as bank account (examples: "Account name", "Account holder", "Beneficiary", "Beneficiary name", "Account:", "Account Name:", "Bank acc name:") exists, set bank_acc_name to that exact text.
2) Otherwise set bank_acc_name = sender_name (the value extracted from explicit sender/company fields).
3) Always also extract sender_name separately from sender/company sections.
4) Never leave bank_acc_name empty unless both bank and sender/company are absent — then set bank_acc_name = ""
- Preserve original number formatting (including commas, decimals), Do not include currency symbol for amount field.
- If multiple line items exist, include all of them in the items array
- Use empty string "" for any field that is not present or cannot be clearly identified
- Maintain accuracy in financial figures - double-check all numbers
- Do not round the tax percentage.For example, If the invoice shows "8.875" or "2.75" (or your calculation yields "8.875", or "2.75"), use 8.875, 2.75  exactly — do not round it to "8.87", "8.88" or "2.8". Store tax_rate as the numeric string without the percent sign (e.g., "8.875").
- Extract text exactly as it appears, including special characters and formatting
- For dates, preserve the original format shown in the invoice
- If both sender and receiver addresses are in the United States, extract ACH; otherwise extract Wire transfer (WT).
- If payment terms specify a number of days (e.g., "payment terms 30 days", "payable within 15 days", "terms 45 days", "Net 30", or any similar phrase), compute: due_date = invoice_date + N days. If the invoice states "due on receipt", "due upon receipt" ,"Immediate" or any similar phrase meaning immediate payment, then: due_date = invoice_date. Use the same date format as the invoice. Output only the computed due_date.
- if tax_rate is not given in invoice but tax_amount is given, calculate the tax_rate using tax_amount and subtotal.
- line-item wise tax calculation has to be done properly based ONLY on the tax_rate given in the summary, and the same tax_rate must be used for every line item in that invoice.
- If currency symbols are present, note them appropriately
-for amount fields, give only NUMERIC VALUE, do not include symbol($) or letter("EUR", "USD") to the amount fields.
- If a discount is present, first subtract the discount amount from the item's (or invoice's) actual amount, then calculate tax on the discounted amount. Tax must be computed on the net (post-discount) value.
- If discount is shown only in the summary (after subtotal), subtract it from subtotal to get the taxable base and then calculate tax; if discounts are line-item, subtract each from its line to get line_total — do NOT apportion summary discounts to line items.
- If any line item includes a discount, subtract the discount amount from that line item's total price. The resulting value should be recorded as the "Line Total" for that item.
- If a tax rate is given (for example, "20%") but the invoice explicitly shows the tax amount as zero (for example, "0.00"), do not calculate or infer any tax; keep the tax amount as shown (0.00).
Return only the JSON object with the extracted information"""

    try:
        # Resize image if too large (max dimension 2048px to avoid payload size issues)
        max_dimension = 2048
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.info(f"Image resized from {width}x{height} to {new_size[0]}x{new_size[1]} to reduce payload size")

        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Check payload size
        payload_size_mb = len(image_base64) / (1024 * 1024)
        if payload_size_mb > 10:
            st.warning(f"Warning: Large image payload ({payload_size_mb:.2f} MB). This might cause issues.")

        data_url = f"data:image/png;base64,{image_base64}"

        # Build payload
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Extract invoice data."}
                ]}
            ],
            "temperature": 0,
            "max_tokens": 1536
        }

        headers = {
            "Authorization": f"Bearer {VLLM_API_KEY}",
            "Content-Type": "application/json"
        }

        # Call API
        st.info(f"Sending request to API (payload size: {payload_size_mb:.2f} MB)...")
        response = requests.post(
            f"{POD_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            # Show detailed error for debugging
            st.error(f"❌ API Error {response.status_code}")
            try:
                error_detail = response.json()
                st.json(error_detail)  # Show as formatted JSON
            except:
                st.code(response.text)  # Show raw text
            return None

    except Exception as e:
        st.error(f"Error calling vLLM: {str(e)}")
        return None


# -----------------------------
# JSON Parser for vLLM Output
# -----------------------------
def parse_vllm_json(raw_json_text):
    """Parse vLLM JSON output into structured format"""
    try:
        # Try to parse the JSON - handle potential markdown code blocks
        text_to_parse = raw_json_text.strip()
        
        # Remove markdown code fences if present
        if text_to_parse.startswith("```json"):
            text_to_parse = text_to_parse[7:]
        elif text_to_parse.startswith("```"):
            text_to_parse = text_to_parse[3:]
        if text_to_parse.endswith("```"):
            text_to_parse = text_to_parse[:-3]
        text_to_parse = text_to_parse.strip()
        
        data = json.loads(text_to_parse)

        header = data.get("header", {})
        summary = data.get("summary", {})
        items = data.get("items", [])

        # Get currency first for date parsing and amount parsing
        currency = summary.get("currency", "")

        def clean_amount(value):
            """Parse monetary amounts using clean_float with currency awareness"""
            return clean_float(value, currency)

        result = {
            "Invoice Number": header.get("invoice_no", ""),
            "Invoice Date": normalize_date(header.get("invoice_date", ""), currency),
            "Due Date": normalize_date(header.get("due_date", ""), currency),
            "Sender Name": header.get("sender_name", ""),
            "Sender Address": header.get("sender_addr", ""),
            "Sender": {
                "Name": header.get("sender_name", ""),
                "Address": header.get("sender_addr", "")
            },
            "Recipient Name": header.get("rcpt_name", ""),
            "Recipient Address": header.get("rcpt_addr", ""),
            "Recipient": {
                "Name": header.get("rcpt_name", ""),
                "Address": header.get("rcpt_addr", "")
            },
            "Bank Details": {
                "bank_iban": header.get("bank_iban", ""),
                "bank_name": header.get("bank_name", ""),
                "bank_account_number": header.get("bank_acc_no", ""),
                "bank_routing": header.get("bank_routing", ""),
                "bank_swift": header.get("bank_swift", ""),
                "bank_acc_name": header.get("bank_acc_name", ""),
                "bank_branch": header.get("bank_branch", "")
            },
            "Subtotal": clean_amount(summary.get("subtotal", "0")),
            "Tax Percentage": clean_tax_percentage(summary.get("tax_rate", "0")),  # ✅ USE clean_tax_percentage
            "Total Tax": clean_amount(summary.get("tax_amount", "0")),
            "Total Amount": clean_amount(summary.get("total_amount", "0")),
            "Currency": currency,
            "Itemized Data": []
        }

        for item in items:
            # Store raw tax value to distinguish empty ("") from explicit "0" or "0.00"
            raw_tax = item.get("tax", "")

            result["Itemized Data"].append({
                "Description": item.get("descriptions", ""),
                "SKU": item.get("SKU", ""),
                "Quantity": clean_quantity(item.get("quantity", "0")),  # ✅ USE clean_quantity for time format support
                "Unit Price": clean_amount(item.get("unit_price", "0")),
                "Amount": clean_amount(item.get("amount", "0")),
                "Tax": clean_amount(raw_tax),
                "Tax_Raw": raw_tax,  # Keep original to distinguish empty vs 0.00
                "Line Total": clean_amount(item.get("Line_total", "0"))
            })

        return result

    except Exception as e:
        st.error(f"JSON parse error: {str(e)}")
        return None


# -----------------------------
# Tax Validation Function
# -----------------------------
def validate_and_calculate_taxes(structured_data):
    """
    Enhanced tax validation with smart line-item calculation:
    1. Skip calculation if tax is empty ("") - tax not provided
    2. Skip calculation if tax is explicitly 0.00 - tax-exempt item
    3. Calculate tax ONLY when line item has a non-zero tax value
    4. Skip validation if tax_amount is 0 but tax_rate exists
    5. Ensure both Tax Percentage and Total Tax are properly filled
    """

    subtotal = structured_data.get("Subtotal", 0.0)
    total_amount = structured_data.get("Total Amount", 0.0)
    model_tax_rate = structured_data.get("Tax Percentage", 0.0)
    model_tax_amount = structured_data.get("Total Tax", 0.0)
    items = structured_data.get("Itemized Data", [])

    # SKIP VALIDATION if: No tax detected (subtotal >= total) OR subtotal is invalid
    if subtotal >= total_amount or subtotal <= 0:
        structured_data["tax_validated"] = False
        structured_data["tax_skip_reason"] = "No tax detected"
        return structured_data

    # SKIP if tax_rate exists but tax_amount is 0 (incomplete data)
    if model_tax_rate > 0 and model_tax_amount == 0.0:
        structured_data["tax_validated"] = False
        structured_data["tax_skip_reason"] = "Tax rate exists but tax amount is 0"
        return structured_data

    # FIRST PASS: Identify which items are taxable (BEFORE determining authoritative rate)
    # This is critical because we need to know the taxable subtotal to calculate the correct rate
    taxable_items = []
    non_taxable_items = []

    for item in items:
        amount = item.get("Amount", 0.0)
        raw_tax_value = item.get("Tax_Raw", "")  # Original string value from JSON

        # If item amount is 0, it's non-taxable
        if amount == 0.0:
            item["Tax"] = 0.0
            item["Line Total"] = 0.0
            non_taxable_items.append(item)
            continue

        # Distinguish between:
        # 1. Empty ("") = tax not provided → NON-TAXABLE
        # 2. Explicit "0", "0.0", "0.00" = tax-exempt → NON-TAXABLE
        # 3. Non-zero value = TAXABLE (calculate tax for this item)

        is_empty = False
        is_explicitly_zero = False

        if isinstance(raw_tax_value, str):
            cleaned = raw_tax_value.strip()
            if cleaned == "":
                # Empty string means tax was not provided
                is_empty = True
            else:
                # Check if it's explicitly set to some form of zero
                try:
                    cleaned_value = float(re.sub(r'[^\d\.-]', '', cleaned) or '0')
                    if cleaned_value == 0.0:
                        is_explicitly_zero = True
                except (ValueError, TypeError):
                    pass
        elif raw_tax_value is None or raw_tax_value == "":
            is_empty = True
        elif raw_tax_value == 0 or raw_tax_value == 0.0:
            # If it's a number 0, treat as explicit zero
            is_explicitly_zero = True

        # If empty - tax not provided, NON-TAXABLE
        if is_empty:
            item["Tax"] = 0.0
            item["Line Total"] = amount
            non_taxable_items.append(item)
            continue

        # If explicitly 0.00 - tax-exempt item, NON-TAXABLE
        if is_explicitly_zero:
            item["Tax"] = 0.0
            item["Line Total"] = amount
            non_taxable_items.append(item)
            continue

        # This item is TAXABLE
        taxable_items.append(item)

    # SECOND PASS: Determine authoritative tax rate from available sources
    # NOW we calculate based on TAXABLE items only (not all items)
    authoritative_rate = None
    authority_source = None

    if taxable_items:
        # Calculate total taxable amount (sum of amounts for taxable items only)
        total_taxable_amount = sum(item.get("Amount", 0.0) for item in taxable_items)

        if total_taxable_amount > 0:
            # TEST SOURCE A: tax_rate (test against taxable subtotal, not total subtotal)
            if model_tax_rate > 0:
                expected_tax_from_rate = total_taxable_amount * (model_tax_rate / 100)
                expected_total_from_rate = subtotal + expected_tax_from_rate
                error_from_rate = abs(expected_total_from_rate - total_amount)
            else:
                error_from_rate = float('inf')

            # TEST SOURCE B: tax_amount (calculate rate based on taxable subtotal only)
            if model_tax_amount > 0:
                calculated_rate_from_amount = (model_tax_amount / total_taxable_amount) * 100
                expected_total_from_amount = subtotal + model_tax_amount
                error_from_amount = abs(expected_total_from_amount - total_amount)
            else:
                error_from_amount = float('inf')

            # PICK WINNER (or use whichever is available)
            if model_tax_rate > 0 or model_tax_amount > 0:
                if error_from_rate < error_from_amount:
                    authoritative_rate = round(model_tax_rate, 4)
                    authority_source = "tax_rate"
                else:
                    authoritative_rate = round(calculated_rate_from_amount, 4)
                    authority_source = "tax_amount"
            else:
                # No tax information available
                structured_data["tax_validated"] = False
                structured_data["tax_skip_reason"] = "No tax rate or amount provided"
                return structured_data
        else:
            # No taxable items with amount > 0
            structured_data["tax_validated"] = False
            structured_data["tax_skip_reason"] = "No taxable items with valid amounts"
            return structured_data
    else:
        # No taxable items found
        structured_data["tax_validated"] = False
        structured_data["tax_skip_reason"] = "No taxable items found"
        return structured_data

    # THIRD PASS: Calculate tax for taxable items using authoritative rate
    calculated_total_tax = 0.0

    if taxable_items and authoritative_rate is not None:
        # Calculate tax for each taxable item
        for item in taxable_items:
            amount = item.get("Amount", 0.0)
            # Calculate tax based on authoritative rate
            corrected_tax = round(amount * (authoritative_rate / 100), 2)
            item["Tax"] = corrected_tax
            calculated_total_tax += corrected_tax
            item["Line Total"] = round(amount + corrected_tax, 2)

    # Update summary - ENSURE BOTH FIELDS ARE FILLED
    structured_data["Tax Percentage"] = authoritative_rate
    structured_data["Total Tax"] = round(calculated_total_tax, 2)
    structured_data["Total Amount"] = round(subtotal + calculated_total_tax, 2)
    structured_data["tax_validated"] = True
    structured_data["tax_authority_source"] = authority_source
    structured_data["original_tax_rate"] = model_tax_rate
    structured_data["original_tax_amount"] = model_tax_amount

    return structured_data


# -----------------------------
# ORIGINAL (previous) mapping logic — restored verbatim
# -----------------------------
def map_prediction_to_ui(pred):
    import json, re
    from collections import defaultdict

    def safe_json_load(s):
        if s is None:
            return None
        if isinstance(s, (dict, list)):
            return s
        if isinstance(s, str):
            s = s.strip()
            if s == "":
                return None
            try:
                return json.loads(s)
            except Exception:
                subs = []
                stack = []
                start = None
                for i, ch in enumerate(s):
                    if ch == "{":
                        if not stack:
                            start = i
                        stack.append("{")
                    elif ch == "}":
                        if stack:
                            stack.pop()
                            if not stack and start is not None:
                                subs.append(s[start:i+1])
                                start = None
                for sub in subs:
                    try:
                        return json.loads(sub)
                    except Exception:
                        continue
        return None

    def clean_number(x):
        """Parse monetary amounts - use clean_float with currency awareness"""
        return clean_float(x, currency)

    def collect_keys(obj, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).strip().lower()
                out[lk].append(v)
                collect_keys(v, out)
        elif isinstance(obj, list):
            for it in obj:
                collect_keys(it, out)

    def collect_lists_of_dicts(obj, out_lists):
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    out_lists.append(v)
                else:
                    collect_lists_of_dicts(v, out_lists)
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, list) and it and isinstance(it[0], dict):
                    out_lists.append(it)
                else:
                    collect_lists_of_dicts(it, out_lists)

    def map_item_dict(it):
        if not isinstance(it, dict):
            return None
        lower = {str(k).strip().lower(): v for k, v in it.items()}
        desc = (lower.get("descriptions") or lower.get("description") or lower.get("desc") or lower.get("item") or "")
        qty = lower.get("quantity") or lower.get("qty") or lower.get("count") or ""
        unit_price = lower.get("unit_price") or lower.get("price") or ""
        amount = lower.get("amount") or lower.get("line_total") or lower.get("line total") or lower.get("total") or ""
        tax = lower.get("tax") or lower.get("tax_amount") or ""
        line_total = lower.get("line_total") or lower.get("line_total".lower()) or lower.get("line total") or amount

        return {
            "Description": str(desc).strip(),
            "Quantity": float(clean_quantity(qty)),  # ✅ USE clean_quantity for time format support
            "Unit Price": float(clean_number(unit_price)),
            "Amount": float(clean_number(amount)),
            "Tax": float(clean_number(tax)),
            "Line Total": float(clean_number(line_total))
        }

    parsed = safe_json_load(pred) if isinstance(pred, str) else pred
    if parsed is None and isinstance(pred, str):
        parsed = None

    if parsed is None and not isinstance(pred, dict):
        parsed = pred

    ui = {
        "Invoice Number": "",
        "Invoice Date": "",
        "Due Date": "",
        "Currency": "",
        "Subtotal": 0.0,
        "Tax Percentage": 0.0,
        "Total Tax": 0.0,
        "Total Amount": 0.0,
        "Sender": {"Name": "", "Address": ""},
        "Recipient": {"Name": "", "Address": ""},
        "Sender Name": "",
        "Sender Address": "",
        "Recipient Name": "",
        "Recipient Address": "",
        "Bank Details": {},
        "Itemized Data": []
    }

    key_map = defaultdict(list)
    list_candidates = []
    if isinstance(parsed, dict):
        collect_keys(parsed, key_map)
        collect_lists_of_dicts(parsed, list_candidates)
    elif isinstance(pred, dict):
        collect_keys(pred, key_map)
        collect_lists_of_dicts(pred, list_candidates)

    def pick_first(*candidate_keys):
        for k in candidate_keys:
            lk = k.strip().lower()
            if lk in key_map:
                for v in key_map[lk]:
                    if v is None:
                        continue
                    if isinstance(v, (dict, list)):
                        return v
                    s = str(v).strip()
                    if s != "":
                        return s
        return None

    # Get currency first for date parsing (USD uses MM/DD/YYYY for numeric dates)
    currency = (pick_first("currency") or "").strip()
    
    ui["Invoice Number"] = pick_first("invoice_no", "invoice_number", "invoiceid", "invoice id") or ""
    ui["Invoice Date"] = normalize_date(pick_first("invoice_date", "date", "invoice date") or "", currency)
    ui["Due Date"] = normalize_date(pick_first("due_date", "due_date", "due") or "", currency)
    ui["Sender Name"] = pick_first("sender_name", "sender") or ""
    ui["Sender Address"] = pick_first("sender_addr", "sender_address", "sender addr") or ""
    ui["Recipient Name"] = pick_first("rcpt_name", "recipient_name", "recipient", "rcpt") or ""
    ui["Recipient Address"] = pick_first("rcpt_addr", "recipient_address", "recipient addr") or ""

    bank = {}
    for bk in ("bank_name", "bank_acc_no", "bank_account_number", "bank_acc_name", "bank_account_holder", "bank_iban", "bank_swift", "bank_routing", "bank_branch", "iban"):
        val = pick_first(bk, bk.replace("bank_", ""))
        if val:
            if bk == "iban":
                bank["bank_iban"] = str(val)
            elif bk == "bank_account_holder":
                bank["bank_acc_name"] = str(val)  # Normalize to bank_acc_name
            else:
                bank[bk if bk != "bank_acc_no" else "bank_account_number"] = str(val)
    ui["Bank Details"] = bank

    ui["Subtotal"] = clean_number(pick_first("subtotal", "sub_total", "sub total") or 0.0)
    ui["Tax Percentage"] = clean_tax_percentage(pick_first("tax_rate", "tax_percentage", "tax pct", "tax percentage") or 0.0)  # ✅ USE clean_tax_percentage
    ui["Total Tax"] = clean_number(pick_first("tax_amount", "tax", "total_tax") or 0.0)
    ui["Total Amount"] = clean_number(pick_first("total_amount", "grand_total", "total", "amount") or 0.0)
    ui["Currency"] = currency

    items_rows = []

    def list_looks_like_items(lst):
        if not isinstance(lst, list) or not lst:
            return False
        if not isinstance(lst[0], dict):
            return False
        expected = {"descriptions", "description", "desc", "item", "quantity", "qty", "amount", "unit_price", "line_total", "line_total".lower(), "line_total"}
        keys0 = {str(k).strip().lower() for k in lst[0].keys()}
        return bool(expected.intersection(keys0))

    for cand in list_candidates:
        if list_looks_like_items(cand):
            for it in cand:
                row = map_item_dict(it)
                if row is not None:
                    items_rows.append(row)
            if items_rows:
                break

    if not items_rows:
        single_candidate_keys = {k.strip().lower() for k in (parsed.keys() if isinstance(parsed, dict) else [])} if isinstance(parsed, dict) else set()
        item_like_keys = {"descriptions", "description", "desc", "item", "quantity", "qty", "unit_price", "unit price", "price", "amount", "line_total", "line total", "line_total", "line_total".lower(), "sku", "tax", "tax_amount"}
        if single_candidate_keys and single_candidate_keys.intersection(item_like_keys):
            single_row = map_item_dict(parsed)
            if single_row is not None:
                items_rows.append(single_row)

    if not items_rows:
        for k, vals in key_map.items():
            for v in vals:
                if isinstance(v, dict):
                    lower_keys = {str(x).strip().lower() for x in v.keys()}
                    if lower_keys.intersection({"descriptions", "description", "desc", "amount", "line_total", "quantity", "qty", "unit_price"}):
                        row = map_item_dict(v)
                        if row is not None:
                            items_rows.append(row)

    if not items_rows:
        desc = pick_first("descriptions", "description")
        amt = pick_first("amount", "line_total")
        qty = pick_first("quantity", "qty")
        unit_price = pick_first("unit_price", "price")
        if desc or amt or qty or unit_price:
            items_rows.append({
                "Description": str(desc or ""),
                "Quantity": float(clean_quantity(qty)),  # ✅ USE clean_quantity for time format support
                "Unit Price": float(clean_number(unit_price)),
                "Amount": float(clean_number(amt)),
                "Tax": float(clean_number(pick_first("tax", "tax_amount") or 0.0)),
                "Line Total": float(clean_number(amt or 0.0))
            })

    ui["Itemized Data"] = items_rows
    ui["Sender"] = {"Name": ui["Sender Name"], "Address": ui["Sender Address"]}
    ui["Recipient"] = {"Name": ui["Recipient Name"], "Address": ui["Recipient Address"]}

    return ui

def flatten_invoice_to_rows(invoice_data) -> list:
    EXPECTED_BANK_FIELDS = [
        "bank_name",
        "bank_account_number",
        "bank_acc_name",
        "bank_iban",
        "bank_swift",
        "bank_routing",
        "bank_branch"
    ]

    # Helper to format text fields (empty -> NA)
    def format_text_field(value):
        if value is None or str(value).strip() == "":
            return "NA"
        return str(value).strip()

    # Helper to format amount fields (empty -> 0)
    def format_amount_field(value):
        if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
            return 0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    rows = []
    invoice_data = invoice_data or {}
    line_items = invoice_data.get("Itemized Data", []) or []

    bank_details = {}
    nested = invoice_data.get("Bank Details", {}) or {}
    if isinstance(nested, dict):
        for k, v in nested.items():
            key_name = k if str(k).startswith("bank_") else f"bank_{k}"
            bank_details[key_name] = v

    for k, v in invoice_data.items():
        if isinstance(k, str) and k.lower().startswith("bank_"):
            bank_details[k] = v

    for f in EXPECTED_BANK_FIELDS:
        bank_details.setdefault(f, "")

    def base_invoice_info():
        return {
            "Invoice Number": format_text_field(invoice_data.get("Invoice Number", "")),
            "Invoice Date": format_text_field(invoice_data.get("Invoice Date", "")),
            "Due Date": format_text_field(invoice_data.get("Due Date", "")),
            "Currency": format_text_field(invoice_data.get("Currency", "")),
            "Subtotal": format_amount_field(invoice_data.get("Subtotal", 0.0)),
            "Tax Percentage": format_amount_field(invoice_data.get("Tax Percentage", 0.0)),
            "Total Tax": format_amount_field(invoice_data.get("Total Tax", 0.0)),
            "Total Amount": format_amount_field(invoice_data.get("Total Amount", 0.0)),
            "Sender Name": format_text_field(invoice_data.get("Sender Name", "") or (invoice_data.get("Sender",{}) or {}).get("Name","")),
            "Sender Address": format_text_field(invoice_data.get("Sender Address", "") or (invoice_data.get("Sender",{}) or {}).get("Address","")),
            "Recipient Name": format_text_field(invoice_data.get("Recipient Name", "") or (invoice_data.get("Recipient",{}) or {}).get("Name","")),
            "Recipient Address": format_text_field(invoice_data.get("Recipient Address", "") or (invoice_data.get("Recipient",{}) or {}).get("Address","")),
        }

    if not line_items:
        row = base_invoice_info()
        for k in EXPECTED_BANK_FIELDS:
            row[k] = format_text_field(bank_details.get(k, ""))
        row.update({
            "Item Description": "NA",
            "Item Quantity": 0,
            "Item Unit Price": 0.0,
            "Item Amount": 0.0,
            "Item Tax": 0.0,
            "Item Line Total": 0.0,
            "IO Number/Cost Centre": "NA",  # Add as last column
        })
        rows.append(row)
        return rows

    for item in line_items:
        row = base_invoice_info()
        for k in EXPECTED_BANK_FIELDS:
            row[k] = format_text_field(bank_details.get(k, ""))
        row.update({
            "Item Description": format_text_field(item.get("Description", "") if isinstance(item, dict) else ""),
            "Item Quantity": format_amount_field(item.get("Quantity", 0) if isinstance(item, dict) else 0),
            "Item Unit Price": format_amount_field(item.get("Unit Price", 0.0) if isinstance(item, dict) else 0.0),
            "Item Amount": format_amount_field(item.get("Amount", 0.0) if isinstance(item, dict) else 0.0),
            "Item Tax": format_amount_field(item.get("Tax", 0.0) if isinstance(item, dict) else 0.0),
            "Item Line Total": format_amount_field(item.get("Line Total", item.get("Amount", 0.0)) if isinstance(item, dict) else 0.0),
            "IO Number/Cost Centre": format_text_field(item.get("IO Number/Cost Centre", "") if isinstance(item, dict) else ""),  # Add as last column
        })
        rows.append(row)
    return rows


# -----------------------------
# Session scaffolding
# -----------------------------
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None
if "is_processing_batch" not in st.session_state:
    st.session_state.is_processing_batch = False

# -----------------------------
# Pre-mount two-column skeleton to avoid layout jump
# -----------------------------
frame_left, frame_right = st.columns([1, 1], vertical_alignment="top")

# -----------------------------
# Upload / Process
# -----------------------------
if not st.session_state.is_processing_batch and len(st.session_state.batch_results) == 0:
    with frame_left:
        st.header("📤 Upload Invoices")
        uploaded_files = st.file_uploader(
            "Upload invoice images (png/jpg/jpeg/pdf)",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.session_state.is_processing_batch = True
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                uploaded_bytes = uploaded_file.read()
                file_hash = hashlib.sha256(uploaded_bytes).hexdigest()

                if file_hash in st.session_state.batch_results:
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    continue

                # Load image (first page for PDFs)
                image = None
                is_pdf = uploaded_file.name.lower().endswith('.pdf') or (hasattr(uploaded_file, 'type') and uploaded_file.type == 'application/pdf')
                if is_pdf:
                    if convert_from_bytes is None:
                        st.warning(f"PDF {uploaded_file.name} could not be rendered (pdf2image/poppler missing).")
                        continue
                    try:
                        pages = convert_from_bytes(uploaded_bytes, dpi=200)
                        if len(pages) > 0:
                            image = pages[0].convert("RGB")
                        else:
                            st.warning(f"PDF {uploaded_file.name} has no pages.")
                            continue
                    except Exception:
                        st.warning(f"Could not render PDF {uploaded_file.name}. Ensure 'pdf2image' and poppler are installed.")
                        continue
                else:
                    try:
                        image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
                    except Exception:
                        st.warning(f"Failed to open {uploaded_file.name}.")
                        continue

                if image is None:
                    continue

                # vLLM Inference + parsing + tax validation
                raw_json = None
                mapped = {}
                try:
                    # Call vLLM API
                    raw_json = run_inference_vllm(image)

                    if raw_json:
                        # Parse JSON response
                        parsed_data = parse_vllm_json(raw_json)

                        if parsed_data:
                            # Apply tax validation
                            mapped = validate_and_calculate_taxes(parsed_data)
                        else:
                            st.warning(f"Failed to parse JSON for {uploaded_file.name}")
                            mapped = {}
                    else:
                        st.warning(f"No response from vLLM for {uploaded_file.name}")
                        mapped = {}

                except Exception as e:
                    st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                    raw_json = None
                    mapped = {}

                safe_mapped = mapped if isinstance(mapped, dict) else {}

                # Store BOTH raw string AND parsed dict for display
                st.session_state.batch_results[file_hash] = {
                    "file_name": uploaded_file.name,
                    "image": image,
                    "raw_pred": raw_json,  # Original string from API (untouched)
                    "mapped_data": safe_mapped,
                    "edited_data": safe_mapped.copy()
                }

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("✅ All files processed!")
            st.session_state.is_processing_batch = False
            st.rerun()

    with frame_right:
        st.caption("Preview & editor will appear here after extraction.")

elif len(st.session_state.batch_results) > 0:

    # --------- Top row: All-results download + Back button ----------
    with frame_left:
        all_rows = []
        for file_hash, result in st.session_state.batch_results.items():
            rows = flatten_invoice_to_rows(result["edited_data"])
            for r in rows:
                r["Source File"] = result.get("file_name", file_hash)
            all_rows.extend(rows)

        if all_rows:
            full_df = pd.DataFrame(all_rows)
            cols = list(full_df.columns)
            if "Source File" in cols:
                cols = ["Source File"] + [c for c in cols if c != "Source File"]
            full_df = full_df[cols]
            csv_bytes = full_df.to_csv(index=False).encode("utf-8")
            st.download_button("📦 Download All Results (CSV)", csv_bytes,
                               file_name="all_extracted_invoices.csv", mime="text/csv", key="download_all_csv")

    with frame_right:
        if st.button("⬅️ Back to Upload"):
            st.session_state.batch_results.clear()
            st.session_state.current_file_hash = None
            st.session_state.is_processing_batch = False
            st.rerun()

    # --------- Selector ----------
    with frame_left:
        file_options = {f"{v['file_name']} ({k[:6]})": k for k, v in st.session_state.batch_results.items()}
        selected_display = st.selectbox("Select invoice to view/edit:", options=list(file_options.keys()), index=0, key="file_selector")
        selected_hash = file_options[selected_display]
        
        # If user switched to a different file, clear the old file's session state to load fresh data
        if st.session_state.current_file_hash != selected_hash:
            # Clear all session state keys for the old file
            if st.session_state.current_file_hash is not None:
                old_hash = st.session_state.current_file_hash
                keys_to_delete = [k for k in st.session_state.keys() if k.endswith(f"_{old_hash}")]
                for key in keys_to_delete:
                    del st.session_state[key]
            
            # Update to new file
            st.session_state.current_file_hash = selected_hash

    current = st.session_state.batch_results[selected_hash]
    image = current["image"]
    form_data = current["edited_data"]

    # --------- Initialize widget state - ONLY IF NOT EXISTS (avoid overwriting user edits) ----------
    bank = form_data.get("Bank Details", {}) if isinstance(form_data.get("Bank Details", {}), dict) else {}
    
    # Get currency for date parsing (USD uses MM/DD/YYYY for numeric dates)
    form_currency = form_data.get('Currency', '')

    # Only initialize if key doesn't exist - this preserves user edits between reruns
    if f"Invoice Number_{selected_hash}" not in st.session_state:
        st.session_state[f"Invoice Number_{selected_hash}"] = form_data.get('Invoice Number', '')

    # Parse dates to date objects for date_input widgets (pass currency for US date handling)
    if f"Invoice Date_{selected_hash}" not in st.session_state:
        invoice_date_obj = parse_date_to_object(form_data.get('Invoice Date', ''), form_currency)
        st.session_state[f"Invoice Date_{selected_hash}"] = invoice_date_obj
    
    if f"Due Date_{selected_hash}" not in st.session_state:
        due_date_obj = parse_date_to_object(form_data.get('Due Date', ''), form_currency)
        st.session_state[f"Due Date_{selected_hash}"] = due_date_obj

    if f"Currency_{selected_hash}" not in st.session_state:
        st.session_state[f"Currency_{selected_hash}"] = form_data.get('Currency', 'USD') or 'USD'
    
    if f"Currency_Custom_{selected_hash}" not in st.session_state:
        st.session_state[f"Currency_Custom_{selected_hash}"] = form_data.get('Currency', '') if form_data.get('Currency') not in ['USD','EUR','GBP','INR'] else ''
    
    if f"Subtotal_{selected_hash}" not in st.session_state:
        st.session_state[f"Subtotal_{selected_hash}"] = float(form_data.get('Subtotal', 0.0))
    
    if f"Tax Percentage_{selected_hash}" not in st.session_state:
        st.session_state[f"Tax Percentage_{selected_hash}"] = float(form_data.get('Tax Percentage', 0.0))
    
    if f"Total Tax_{selected_hash}" not in st.session_state:
        st.session_state[f"Total Tax_{selected_hash}"] = float(form_data.get('Total Tax', 0.0))
    
    if f"Total Amount_{selected_hash}" not in st.session_state:
        st.session_state[f"Total Amount_{selected_hash}"] = float(form_data.get('Total Amount', 0.0))
    
    if f"Sender Name_{selected_hash}" not in st.session_state:
        st.session_state[f"Sender Name_{selected_hash}"] = form_data.get('Sender Name', '')
    
    if f"Sender Address_{selected_hash}" not in st.session_state:
        st.session_state[f"Sender Address_{selected_hash}"] = form_data.get('Sender Address', '')
    
    if f"Recipient Name_{selected_hash}" not in st.session_state:
        st.session_state[f"Recipient Name_{selected_hash}"] = form_data.get('Recipient Name', '')
    
    if f"Recipient Address_{selected_hash}" not in st.session_state:
        st.session_state[f"Recipient Address_{selected_hash}"] = form_data.get('Recipient Address', '')
    
    if f"Bank_bank_name_{selected_hash}" not in st.session_state:
        st.session_state[f"Bank_bank_name_{selected_hash}"] = bank.get('bank_name', '')
    
    if f"Bank_bank_account_number_{selected_hash}" not in st.session_state:
        st.session_state[f"Bank_bank_account_number_{selected_hash}"] = bank.get('bank_account_number', '') or bank.get('bank_acc_no', '')
    
    if f"Bank_bank_acc_name_{selected_hash}" not in st.session_state:
        st.session_state[f"Bank_bank_acc_name_{selected_hash}"] = bank.get('bank_acc_name', '') or bank.get('bank_account_holder', '')
    
    if f"Bank_bank_iban_{selected_hash}" not in st.session_state:
        st.session_state[f"Bank_bank_iban_{selected_hash}"] = bank.get('bank_iban', '')
    
    if f"Bank_bank_swift_{selected_hash}" not in st.session_state:
        st.session_state[f"Bank_bank_swift_{selected_hash}"] = bank.get('bank_swift', '')
    
    if f"Bank_bank_routing_{selected_hash}" not in st.session_state:
        st.session_state[f"Bank_bank_routing_{selected_hash}"] = bank.get('bank_routing', '')
    
    if f"Bank_bank_branch_{selected_hash}" not in st.session_state:
        st.session_state[f"Bank_bank_branch_{selected_hash}"] = bank.get('bank_branch', '')

    # --------- Display (no wobble) ----------
    with frame_left:
        st.image(image, caption=current["file_name"], width=FIXED_IMG_WIDTH)
        st.write(f"**File Hash:** {selected_hash[:8]}...")
        
        # ============ RAW MODEL OUTPUT DISPLAY (UNTOUCHED) ============
        with st.expander("🔍 Show raw model output"):
            raw_pred = current.get('raw_pred')
            
            if raw_pred is None:
                st.warning("No raw output available (API may have returned None)")
            else:
                # Show raw output exactly as received from the model - UNTOUCHED
                st.code(str(raw_pred), language='json')
        # ==============================================================

        if st.button("🔁 Re-Run Inference", key=f"rerun_{selected_hash}"):
            with st.spinner("Re-running inference..."):
                try:
                    # Call vLLM API
                    raw_json = run_inference_vllm(image)

                    if raw_json:
                        # Parse JSON response
                        parsed_data = parse_vllm_json(raw_json)

                        if parsed_data:
                            # Apply tax validation
                            mapped = validate_and_calculate_taxes(parsed_data)
                        else:
                            st.error("Failed to parse JSON response")
                            mapped = {}
                    else:
                        st.error("No response from vLLM")
                        mapped = {}

                    safe_mapped = mapped if isinstance(mapped, dict) else {}

                    # Update stored results
                    st.session_state.batch_results[selected_hash]["raw_pred"] = raw_json
                    st.session_state.batch_results[selected_hash]["mapped_data"] = mapped
                    st.session_state.batch_results[selected_hash]["edited_data"] = safe_mapped.copy()

                    # Clear widget state for this file so defaults refresh from new mapped data
                    for key in [k for k in st.session_state.keys() if k.endswith(f"_{selected_hash}")]:
                        del st.session_state[key]

                    st.success("✅ Re-run complete")
                    st.rerun()
                except Exception as e:
                    st.error(f"Re-run failed: {e}")

    with frame_right:
        st.subheader(f"Editable Invoice: {current['file_name']}")

        # ----------------- FORM START -----------------
        with st.form(key=f"edit_form_{selected_hash}", clear_on_submit=False):
            tabs = st.tabs(["Invoice Details", "Sender/Recipient", "Bank Details", "Line Items"])

            with tabs[0]:
                st.text_input("Invoice Number", key=f"Invoice Number_{selected_hash}")
                
                # HYBRID DATE DISPLAY: Formatted display + Date picker
                st.write("**Invoice Date:**")
                invoice_date_obj = st.session_state.get(f"Invoice Date_{selected_hash}", None)
                if invoice_date_obj:
                    formatted_invoice = invoice_date_obj.strftime("%d-%b-%Y")
                    st.info(f"📅 {formatted_invoice}")  # Shows: 📅 25-Sep-2025
                st.date_input("Select date:", key=f"Invoice Date_{selected_hash}", 
                              format="DD/MM/YYYY", label_visibility="collapsed")
                
                st.write("**Due Date:**")
                due_date_obj = st.session_state.get(f"Due Date_{selected_hash}", None)
                if due_date_obj:
                    formatted_due = due_date_obj.strftime("%d-%b-%Y")
                    st.info(f"📅 {formatted_due}")  # Shows: 📅 30-Sep-2025
                st.date_input("Select date:", key=f"Due Date_{selected_hash}",
                              format="DD/MM/YYYY", label_visibility="collapsed")

                curr_options = ['USD', 'EUR', 'GBP', 'INR', 'Other']
                if st.session_state[f"Currency_{selected_hash}"] not in curr_options:
                    st.session_state[f"Currency_{selected_hash}"] = 'Other'
                st.selectbox("Currency", options=curr_options, key=f"Currency_{selected_hash}")

                if st.session_state.get(f"Currency_{selected_hash}") == 'Other':
                    st.text_input("Specify Currency", key=f"Currency_Custom_{selected_hash}")

                # ✅ FIX: Add step=None to prevent scroll interference
                st.number_input("Subtotal", key=f"Subtotal_{selected_hash}", format="%.2f", step=None)
                st.number_input("Tax %", key=f"Tax Percentage_{selected_hash}", format="%.4f", step=None)
                st.number_input("Total Tax", key=f"Total Tax_{selected_hash}", format="%.2f", step=None)
                st.number_input("Total Amount", key=f"Total Amount_{selected_hash}", format="%.2f", step=None)

            with tabs[1]:
                st.text_input("Sender Name", key=f"Sender Name_{selected_hash}")
                st.text_area("Sender Address", key=f"Sender Address_{selected_hash}", height=80)
                st.text_input("Recipient Name", key=f"Recipient Name_{selected_hash}")
                st.text_area("Recipient Address", key=f"Recipient Address_{selected_hash}", height=80)

            with tabs[2]:
                st.text_input("Bank Name", key=f"Bank_bank_name_{selected_hash}")
                st.text_input("Account Number", key=f"Bank_bank_account_number_{selected_hash}")
                st.text_input("Account Name", key=f"Bank_bank_acc_name_{selected_hash}")
                st.text_input("IBAN", key=f"Bank_bank_iban_{selected_hash}")
                st.text_input("SWIFT", key=f"Bank_bank_swift_{selected_hash}")
                st.text_input("Routing", key=f"Bank_bank_routing_{selected_hash}")
                st.text_input("Branch", key=f"Bank_bank_branch_{selected_hash}")

            with tabs[3]:
                # Initialize line items DataFrame in session state ONLY ONCE (preserves edits)
                items_state_key = f"items_df_{selected_hash}"
                
                if items_state_key not in st.session_state:
                    # Build base DF from current edited_data only on first load
                    item_rows = form_data.get('Itemized Data', []) or []
                    normalized = []
                    for it in item_rows:
                        if not isinstance(it, dict):
                            it = {}
                        normalized.append({
                            "Description": it.get("Description", it.get("Item Description", "")),
                            "Quantity": it.get("Quantity", it.get("Item Quantity", 0)),
                            "Unit Price": it.get("Unit Price", it.get("Item Unit Price", 0.0)),
                            "Amount": it.get("Amount", it.get("Item Amount", 0.0)),
                            "IO Number/Cost Centre": it.get("IO Number/Cost Centre", ""),  # New column
                            "Tax": it.get("Tax", it.get("Item Tax", 0.0)),
                            "Line Total": it.get("Line Total", it.get("Item Line Total", 0.0)),
                        })

                    st.session_state[items_state_key] = pd.DataFrame(normalized) if normalized else pd.DataFrame(
                        columns=["Description", "Quantity", "Unit Price", "Amount", "IO Number/Cost Centre", "Tax", "Line Total"]
                    )
                
                # Use the stored DataFrame
                items_df = st.session_state[items_state_key]

                # Configure column widths to make Description wider and avoid horizontal scrolling
                column_config = {
                    "Description": st.column_config.TextColumn(
                        "Description",
                        width="large",  # Make description column wider
                    ),
                    "Quantity": st.column_config.NumberColumn(
                        "Quantity",
                        width="small",
                    ),
                    "Unit Price": st.column_config.NumberColumn(
                        "Unit Price",
                        width="small",
                        format="%.2f"
                    ),
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        width="small",
                        format="%.2f"
                    ),
                    "IO Number/Cost Centre": st.column_config.TextColumn(
                        "IO Number/Cost Centre",
                        width="medium",
                    ),
                    "Tax": st.column_config.NumberColumn(
                        "Tax",
                        width="small",
                        format="%.2f"
                    ),
                    "Line Total": st.column_config.NumberColumn(
                        "Line Total",
                        width="small",
                        format="%.2f"
                    ),
                }

                # Show editor without totals - no horizontal scrolling
                edited_df = st.data_editor(
                    items_df,
                    num_rows="dynamic",
                    key=f"items_editor_{selected_hash}",
                    use_container_width=True,
                    height=DATA_EDITOR_HEIGHT - 50,  # Reduce height slightly for totals below
                    column_config=column_config,
                )
                
                # Update session state with edited DataFrame (preserves changes across reruns)
                st.session_state[items_state_key] = edited_df

                # Display non-editable totals row immediately below (looks integrated)
                if len(edited_df) > 0:
                    # Use clean_float so weird types (lists, strings, etc.) don't break things
                    total_amount = sum(clean_float(v) for v in edited_df["Amount"])
                    total_tax = sum(clean_float(v) for v in edited_df["Tax"])
                    total_line_total = sum(clean_float(v) for v in edited_df["Line Total"])
                
                    totals_df = pd.DataFrame([{
                        "Description": "──── TOTAL ────",
                        "Quantity": "",
                        "Unit Price": "",
                        "Amount": f"{total_amount:,.2f}",
                        "IO Number/Cost Centre": "",
                        "Tax": f"{total_tax:,.2f}",
                        "Line Total": f"{total_line_total:,.2f}"
                    }])
                
                    st.dataframe(
                        totals_df,
                        use_container_width=True,
                        hide_index=True,
                        height=38
                    )
                    
                    # Add spacing before save button to prevent overlap
                    st.write("")  # Empty line for spacing

            saved = st.form_submit_button("💾 Save All Edits")
        # ----------------- FORM END -----------------

        # Calculate current values for display and download (always, not just on save)
        currency = st.session_state.get(f"Currency_{selected_hash}", 'USD')
        if currency == 'Other':
            currency = st.session_state.get(f"Currency_Custom_{selected_hash}", '')

        # Convert date objects to normalized strings (dd-MMM-yyyy format)
        invoice_date = st.session_state.get(f"Invoice Date_{selected_hash}", None)
        due_date = st.session_state.get(f"Due Date_{selected_hash}", None)

        invoice_date_str = ""
        if invoice_date is not None:
            try:
                invoice_date_str = invoice_date.strftime("%d-%b-%Y")
            except (AttributeError, ValueError):
                invoice_date_str = ""

        due_date_str = ""
        if due_date is not None:
            try:
                due_date_str = due_date.strftime("%d-%b-%Y")
            except (AttributeError, ValueError):
                due_date_str = ""

        # Get the current line items from session state (not the form-scoped edited_df)
        items_state_key = f"items_df_{selected_hash}"
        current_items_df = st.session_state.get(items_state_key, pd.DataFrame())
        
        # Calculate totals from line items (always, for both save and download)
        line_items_list = current_items_df.to_dict('records')
        calculated_subtotal = sum(clean_float(item.get('Amount', 0)) for item in line_items_list)
        calculated_total_tax = sum(clean_float(item.get('Tax', 0)) for item in line_items_list)
        calculated_total = sum(clean_float(item.get('Line Total', 0)) for item in line_items_list)

        # Calculate tax percentage if possible
        calculated_tax_pct = 0.0
        if calculated_subtotal > 0 and calculated_total_tax > 0:
            calculated_tax_pct = round((calculated_total_tax / calculated_subtotal) * 100, 4)

        if saved:
            # Build updated data structure using ACTUAL user-entered values from form
            updated = {
                'Invoice Number': st.session_state.get(f"Invoice Number_{selected_hash}", ''),
                'Invoice Date': invoice_date_str,
                'Due Date': due_date_str,
                'Currency': currency,
                'Subtotal': st.session_state.get(f"Subtotal_{selected_hash}", 0.0),
                'Tax Percentage': st.session_state.get(f"Tax Percentage_{selected_hash}", 0.0),
                'Total Tax': st.session_state.get(f"Total Tax_{selected_hash}", 0.0),
                'Total Amount': st.session_state.get(f"Total Amount_{selected_hash}", 0.0),
                'Sender Name': st.session_state.get(f"Sender Name_{selected_hash}", ''),
                'Sender Address': st.session_state.get(f"Sender Address_{selected_hash}", ''),
                'Recipient Name': st.session_state.get(f"Recipient Name_{selected_hash}", ''),
                'Recipient Address': st.session_state.get(f"Recipient Address_{selected_hash}", ''),
                'Bank Details': {
                    'bank_name': st.session_state.get(f"Bank_bank_name_{selected_hash}", ''),
                    'bank_account_number': st.session_state.get(f"Bank_bank_account_number_{selected_hash}", ''),
                    'bank_acc_name': st.session_state.get(f"Bank_bank_acc_name_{selected_hash}", ''),
                    'bank_iban': st.session_state.get(f"Bank_bank_iban_{selected_hash}", ''),
                    'bank_swift': st.session_state.get(f"Bank_bank_swift_{selected_hash}", ''),
                    'bank_routing': st.session_state.get(f"Bank_bank_routing_{selected_hash}", ''),
                    'bank_branch': st.session_state.get(f"Bank_bank_branch_{selected_hash}", '')
                },
                'Itemized Data': line_items_list,
                'Sender': {"Name": st.session_state.get(f"Sender Name_{selected_hash}", ''),
                           "Address": st.session_state.get(f"Sender Address_{selected_hash}", '')},
                'Recipient': {"Name": st.session_state.get(f"Recipient Name_{selected_hash}", ''),
                              "Address": st.session_state.get(f"Recipient Address_{selected_hash}", '')},
            }

            # Save to batch_results (this persists the data)
            st.session_state.batch_results[selected_hash]["edited_data"] = updated
            
            # CRITICAL: Clear ALL session state keys for this file so they reload from saved edited_data
            keys_to_delete = [k for k in list(st.session_state.keys()) if k.endswith(f"_{selected_hash}")]
            for key in keys_to_delete:
                del st.session_state[key]
            
            # Show success message
            st.success("✅ Saved")
            
            # Rerun to reload the form with saved data
            st.rerun()

        # Per-file CSV download (ALWAYS visible, uses current form values)
        download_data = {
            'Invoice Number': st.session_state.get(f"Invoice Number_{selected_hash}", ''),
            'Invoice Date': invoice_date_str,
            'Due Date': due_date_str,
            'Currency': currency,
            'Subtotal': st.session_state.get(f"Subtotal_{selected_hash}", 0.0),
            'Tax Percentage': st.session_state.get(f"Tax Percentage_{selected_hash}", 0.0),
            'Total Tax': st.session_state.get(f"Total Tax_{selected_hash}", 0.0),
            'Total Amount': st.session_state.get(f"Total Amount_{selected_hash}", 0.0),
            'Sender Name': st.session_state.get(f"Sender Name_{selected_hash}", ''),
            'Sender Address': st.session_state.get(f"Sender Address_{selected_hash}", ''),
            'Recipient Name': st.session_state.get(f"Recipient Name_{selected_hash}", ''),
            'Recipient Address': st.session_state.get(f"Recipient Address_{selected_hash}", ''),
            'Bank Details': {
                'bank_name': st.session_state.get(f"Bank_bank_name_{selected_hash}", ''),
                'bank_account_number': st.session_state.get(f"Bank_bank_account_number_{selected_hash}", ''),
                'bank_acc_name': st.session_state.get(f"Bank_bank_acc_name_{selected_hash}", ''),
                'bank_iban': st.session_state.get(f"Bank_bank_iban_{selected_hash}", ''),
                'bank_swift': st.session_state.get(f"Bank_bank_swift_{selected_hash}", ''),
                'bank_routing': st.session_state.get(f"Bank_bank_routing_{selected_hash}", ''),
                'bank_branch': st.session_state.get(f"Bank_bank_branch_{selected_hash}", '')
            },
            'Itemized Data': line_items_list
        }
        rows = flatten_invoice_to_rows(download_data)
        full_df = pd.DataFrame(rows)
        csv_bytes_one = full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download This Invoice (CSV)",
            csv_bytes_one,
            file_name=f"{Path(current['file_name']).stem}_full.csv",
            mime="text/csv",
            key=f"dl_{selected_hash}"
        )

elif st.session_state.is_processing_batch:
    with frame_left:
        st.info("⏳ Processing batch... Please wait.")
        st.progress(0)
    with frame_right:
        st.caption("Preview & editor will appear here after extraction.")

else:
    # Shouldn't happen, but keeps skeleton steady
    with frame_left:
        st.caption("Ready when you are.")
    with frame_right:
        st.caption("Preview & editor will appear here after extraction.")
