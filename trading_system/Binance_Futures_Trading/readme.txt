https://demo.binance.com/fr/my/settings/api-management

Gestion des API
1. Chaque compte peut créer 30 clés API.
2. Ne divulguez pas votre clé API, votre clé secrète (HMAC) ou votre clé privée (Ed25519, RSA) à qui que ce soit, afin d’éviter de perdre des actifs. Vous devez considérer votre clé API et votre clé secrète (HMAC) ou votre clé privée (Ed25519, RSA) comme vos mots de passe.
3. Il est recommandé de restreindre l’accès uniquement aux adresses IP auxquelles vous avez confiance, afin de renforcer la sécurité de votre compte.
4. Vous ne pourrez pas créer de clé API tant que le KYC n’aura pas été effectué.
Documentation API Spot :https://developers.binance.com/docs/binance-spot-api-docs/
Documentation API Futures :https://developers.binance.com/docs/derivatives/


demo.binance.com API Trading


Clé API
X4YoWNWVwGzoT3WayKn1cVzExUoN2jCWQ8uXlLiMW0geuPdWIxUFv0Ce6YW2xq8x

Clé secrète
yWH6NGkOb3R0I9tGnB04od24xX42Wk9WPFtb70e0Gwg4RZtqzPMANo8QzS40Izgy


To run Demo Trading (visible on demo.binance.com):

cd c:\Users\Jean-Yves\thevolumeainative\trading_system\Binance_Futures_Trading
py run_demo_trading.py -y
This will:
Run continuously (24/7)
Place real orders on the demo account
Positions will be visible at https://demo.binance.com/en/futures/
No real money at risk
To run Paper Trading (console only):

py run_paper_trading.py -y
To run REAL MONEY (DANGEROUS):

py run_live_trading.py --live
(Requires typing "YES" in caps and safety confirmations)
Use run_demo_trading.py -y to start Demo trading now - you'll see positions appear on demo.binance.com!