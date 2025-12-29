# Match-Trader Trading System for FundedNext

Automated trading system for FundedNext prop firm challenge using Match-Trader platform.

## Files

| File | Description |
|------|-------------|
| `match_trader_client.py` | Match-Trader REST API client |
| `match_trader_engine.py` | Trading engine with risk management |
| `config.json` | Configuration file (edit with your credentials) |
| `run_fundednext.py` | Main entry point |

## Quick Start

### 1. Get Your Credentials from FundedNext

After purchasing a challenge:
1. Login to FundedNext dashboard
2. Go to your Match-Trader account
3. Note down:
   - Server URL (base_url)
   - Broker ID
   - Your email and password

### 2. Configure

Edit `config.json`:

```json
{
    "match_trader": {
        "base_url": "https://your-server.match-trader.com",
        "broker_id": "fundednext",
        "email": "your@email.com",
        "password": "your_password"
    }
}
```

### 3. Run

```bash
cd trading_system/MatchTrader
python run_fundednext.py
```

## FundedNext Stellar 2-Step Rules

| Rule | Phase 1 | Phase 2 |
|------|---------|---------|
| Profit Target | 8% | 5% |
| Daily Loss Limit | 5% | 5% |
| Max Drawdown | 10% | 10% |
| Min Trading Days | 5 | 5 |

## Risk Management

The system enforces:
- **Daily loss limit**: Stops trading if daily loss reaches 5%
- **Max drawdown protection**: Stops if total loss reaches 10%
- **Position sizing**: Risk 1% per trade
- **Max positions**: 2 concurrent positions
- **Max lot size**: 0.3 lots per position

## API Endpoints Used

| Action | Endpoint |
|--------|----------|
| Login | `POST /manager/mtr-login` |
| Get Balance | `GET /mtr-api/{uuid}/balance` |
| Get Quote | `GET /mtr-api/{uuid}/quotations` |
| Open Position | `POST /mtr-api/{uuid}/position/open` |
| Close Position | `POST /mtr-api/{uuid}/position/close` |
| Edit Position | `POST /mtr-api/{uuid}/position/edit` |

## Safety Features

1. **Token refresh**: Automatically refreshes auth token before expiry
2. **Rate limiting**: Respects 500 requests/minute limit
3. **Error handling**: Graceful handling of API errors
4. **Daily reset**: Resets daily P&L tracking at midnight
5. **Breakeven stops**: Moves SL to entry when in profit
6. **Trailing stops**: Locks in profits as price moves

## Disclaimer

Trading involves risk. Use at your own risk. Always test with demo account first.
