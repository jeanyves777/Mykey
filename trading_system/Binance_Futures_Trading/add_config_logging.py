#!/usr/bin/env python3
"""Add config logging to live trading engine"""

config_path = '/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py'

with open(config_path, 'r') as f:
    content = f.read()

# Add config logging method before the run method
config_log_method = '''
    def log_trading_config(self):
        """Log all trading configuration parameters at startup"""
        self.log("="*70)
        self.log("TRADING CONFIGURATION PARAMETERS")
        self.log("="*70)

        # General settings
        self.log(f"Leverage: {STRATEGY_CONFIG.get('leverage', 20)}x")
        self.log(f"ADX Threshold (Strong Trend): {self.adx_threshold}")
        self.log(f"Strong Trend DCA Block: ALL DCA on loser side")

        # Default DCA Levels
        self.log("DEFAULT DCA LEVELS:")
        self.log(f"  TP ROI: {DCA_CONFIG.get('take_profit_roi', 0.08)*100:.0f}%")
        self.log(f"  SL ROI: {DCA_CONFIG.get('stop_loss_roi', 0.90)*100:.0f}%")
        for i, lvl in enumerate(DCA_CONFIG.get('levels', []), 1):
            self.log(f"  DCA {i}: Trigger={lvl.get('trigger_roi', 0)*100:.0f}% | Mult={lvl.get('multiplier', 1.0)}x | TP={lvl.get('tp_roi', 0.08)*100:.0f}%")

        # Per-symbol settings
        self.log("PER-SYMBOL DCA SETTINGS:")
        for symbol in self.symbols:
            settings = SYMBOL_SETTINGS.get(symbol, {})
            symbol_dca = settings.get('dca_levels', None)
            vol_mult = settings.get('dca_volatility_mult', 1.0)
            tp_roi = settings.get('tp_roi', DCA_CONFIG.get('take_profit_roi', 0.08))

            self.log(f"  {symbol}:")
            self.log(f"    TP ROI: {tp_roi*100:.0f}%")

            if symbol_dca:
                self.log(f"    Custom DCA Levels (no volatility multiplier):")
                for i, lvl in enumerate(symbol_dca, 1):
                    trend_filter = "+ Trend Filter" if lvl.get('require_trend_filter', False) else ""
                    self.log(f"      DCA {i}: Trigger={lvl.get('trigger_roi', 0)*100:.0f}% | Mult={lvl.get('multiplier', 1.0)}x {trend_filter}")
            else:
                self.log(f"    Using default levels with volatility_mult={vol_mult}x")
                for i, lvl in enumerate(DCA_CONFIG.get('levels', []), 1):
                    effective_trigger = lvl.get('trigger_roi', 0) * vol_mult
                    self.log(f"      DCA {i}: Trigger={effective_trigger*100:.0f}% (base {lvl.get('trigger_roi', 0)*100:.0f}% x {vol_mult})")

        # Boost Mode settings
        self.log("BOOST MODE:")
        self.log("  Trigger at DCA: 3")
        self.log("  Boost multiplier: 1.5x")
        self.log("  Half-close at TP: Yes")
        self.log("  Trailing after half-close: Yes")

        self.log("="*70)

'''

# Insert before def run
insert_point = '    def run(self, duration_hours: float = 0):'
if insert_point in content:
    content = content.replace(insert_point, config_log_method + insert_point)
    print('SUCCESS: Added log_trading_config method')
else:
    print('ERROR: Could not find run method')

# Now add the call to log_trading_config after starting balance log
old_balance_log = '        self.log(f"Starting balance: ${self.starting_balance:,.2f}")'
new_balance_log = '''        self.log(f"Starting balance: ${self.starting_balance:,.2f}")

        # Log all trading configuration
        self.log_trading_config()'''

if old_balance_log in content:
    content = content.replace(old_balance_log, new_balance_log)
    print('SUCCESS: Added config logging call after starting balance')
else:
    print('ERROR: Could not find starting balance log')

with open(config_path, 'w') as f:
    f.write(content)
