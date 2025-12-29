# OANDA Forex Trading - Quick Setup Guide

## Step 1: Get OANDA Practice Account (5 minutes)

1. Go to https://www.oanda.com/
2. Click "Sign Up" or "Practice Account"
3. Fill in basic info (name, email, password)
4. Verify email
5. Login to your account

## Step 2: Get API Credentials (2 minutes)

1. Once logged in, go to "Manage API Access" or "My Account" → "API Access"
2. Click "Generate" or "Create Token"
3. Copy your **API Token** (looks like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6-a1b2c3d4e5f6g7h8`)
4. Copy your **Account ID** (looks like: `123-456-7890123-001`)

## Step 3: Configure .env File (1 minute)

Open your `.env` file in the project root and add:

```bash
# OANDA Practice Account
OANDA_PRACTICE_API_KEY=paste_your_api_token_here
OANDA_PRACTICE_ACCOUNT_ID=paste_your_account_id_here
```

**Example:**
```bash
OANDA_PRACTICE_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6-a1b2c3d4e5f6g7h8
OANDA_PRACTICE_ACCOUNT_ID=123-456-7890123-001
```

Save the file.

## Step 4: Test Connection (30 seconds)

```bash
python trading_system/Forex_Trading/engine/oanda_client.py
```

You should see:
```
[OANDA] Initialized practice client
[OANDA] Account ID: 123-456-7890123-001

Account Balance: $100,000.00
NAV: $100,000.00
Margin Available: $100,000.00

EUR/USD: Bid=1.08450, Ask=1.08452, Spread=0.00002
```

## Step 5: Run Your First Backtest (2 minutes)

```bash
python trading_system/Forex_Trading/run_backtest.py --days 7 -y
```

This will:
- Download 7 days of EUR/USD and GBP/USD data
- Run the multi-timeframe momentum strategy
- Show you complete results

## Step 6: Run Paper Trading (Optional)

```bash
# Run for 1 hour to test
python trading_system/Forex_Trading/run_paper_trading.py --hours 1 -y

# Run for full trading day
python trading_system/Forex_Trading/run_paper_trading.py --hours 8 -y
```

## Troubleshooting

### Error: "OANDA practice credentials not found"
- Check your `.env` file is in the project root (same folder as `trading_system/`)
- Verify variable names are exactly: `OANDA_PRACTICE_API_KEY` and `OANDA_PRACTICE_ACCOUNT_ID`
- No quotes needed around values
- No extra spaces

### Error: "401 Unauthorized"
- Your API token is incorrect - regenerate it in OANDA dashboard
- Make sure you're using the practice token, not live token

### Error: "Account not found"
- Your Account ID is incorrect
- Check it matches exactly what's shown in OANDA dashboard

### No data returned
- Practice account might need to be "activated" by logging into OANDA web interface first
- Some regions may have restrictions - try accessing OANDA website first

## What's Next?

1. **Review Results**: Check `trading_system/Forex_Trading/logs/` for detailed trade logs
2. **Adjust Strategy**: Edit `config/forex_trading_config.py` to tune parameters
3. **Test More Pairs**: Run backtest with `--instruments EUR_USD GBP_USD USD_JPY`
4. **Longer Backtest**: Try `--days 90` to see performance over 3 months
5. **Paper Trade Live**: Run paper trading during London/NY session for best signals

## Best Practices

- **Start Small**: Begin with 2-3 currency pairs
- **Test First**: Always backtest before paper trading
- **London/NY Hours**: Best signals during 13:00-17:00 UTC (8 AM - 12 PM EST)
- **Review Logs**: Check trade logs to understand strategy decisions
- **Be Patient**: Strategy is strict - it will skip many signals (that's good!)

## Currency Pair Notation

OANDA uses underscore notation:
- EUR/USD → `EUR_USD`
- GBP/USD → `GBP_USD`
- USD/JPY → `USD_JPY`

## Practice Account Limits

OANDA practice accounts:
- ✅ Free forever
- ✅ $100,000 virtual money (can be reset)
- ✅ Real-time market data
- ✅ Full API access
- ✅ No credit card required
- ❌ Profits are not real
- ❌ Cannot withdraw

Perfect for testing strategies!

## Getting Help

If you see errors, check:
1. `.env` file is configured correctly
2. OANDA account is active (login to web interface)
3. Internet connection is stable
4. Running during market hours (Sunday 5 PM - Friday 5 PM EST)

Market is closed on weekends - you'll get limited data if testing Saturday/Sunday!
