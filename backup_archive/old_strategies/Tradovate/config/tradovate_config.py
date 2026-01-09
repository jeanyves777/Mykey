"""
Tradovate API Configuration

IMPORTANT: Fill in your credentials after creating Tradovate account
"""

# ==================== TRADOVATE CREDENTIALS ====================

# Demo Account (for testing)
TRADOVATE_DEMO_USERNAME = ""  # Your Tradovate demo username
TRADOVATE_DEMO_PASSWORD = ""  # Your Tradovate demo password
TRADOVATE_DEMO_API_KEY = ""   # Get this from Tradovate Settings > API

# Live Account (for real trading / FundedNext)
TRADOVATE_LIVE_USERNAME = ""  # Your Tradovate live username
TRADOVATE_LIVE_PASSWORD = ""  # Your Tradovate live password
TRADOVATE_LIVE_API_KEY = ""   # Get this from Tradovate Settings > API

# ==================== ACCOUNT SETTINGS ====================

# Use demo account by default
USE_DEMO = True  # Set to False for live trading

# Account size (for FundedNext)
INITIAL_BALANCE = 25000  # $25K challenge

# ==================== RISK MANAGEMENT ====================

# FundedNext Challenge Rules
FUNDEDNEXT_RULES = {
    'max_loss_limit': 1000,     # $1,000 max loss
    'profit_target': 1250,      # $1,250 profit target
    'consistency_limit': 0.40,  # 40% max per day
    'daily_loss_limit': -500,   # Self-imposed daily loss limit
    'max_concurrent': 5,        # Max 5 positions
    'max_trades_per_day': 50,   # Max 50 trades/day
}

# Position Sizing
CONTRACTS_PER_TRADE = 1  # Fixed 1 contract per trade

# ==================== TRADING HOURS ====================

# Forex futures trade nearly 24/5
TRADING_HOURS = {
    'start': '18:00',  # Sunday 6 PM ET (futures open)
    'end': '17:00',    # Friday 5 PM ET (futures close)
    'timezone': 'America/New_York'
}

# ==================== HOW TO GET API CREDENTIALS ====================

"""
1. Create Tradovate Account:
   - Demo: https://trader.tradovate.com/#/signup
   - Live: https://www.tradovate.com/

2. Get API Key:
   - Log in to Tradovate
   - Go to Settings > API
   - Click "Generate API Key"
   - Copy the API key

3. Fill in credentials above:
   TRADOVATE_DEMO_USERNAME = "your_username"
   TRADOVATE_DEMO_PASSWORD = "your_password"
   TRADOVATE_DEMO_API_KEY = "your_api_key"

4. Test connection:
   python run_tradovate_test_connection.py
"""
