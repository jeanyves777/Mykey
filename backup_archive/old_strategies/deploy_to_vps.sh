#!/bin/bash
# Deploy trading system changes to VPS and restart live trading

VPS_HOST="root@72.62.3.184"
VPS_PATH="/root/thevolumeainative/trading_system"
SSH_KEY="~/.ssh/id_rsa_vps"
LOG_FILE="/tmp/htf_engine.log"

echo "🚀 Deploying to VPS at 72.62.3.184..."

# Check if specific files are provided
if [ $# -eq 0 ]; then
    echo "📦 Syncing entire trading_system directory..."
    rsync -avz --progress -e "ssh -i $SSH_KEY" \
        ./trading_system/ \
        $VPS_HOST:$VPS_PATH/
else
    echo "📦 Syncing specific files: $@"
    for file in "$@"; do
        # Remove 'trading_system/' prefix from file path if present
        relative_path="${file#trading_system/}"
        remote_path="$VPS_PATH/$relative_path"
        
        # Create remote directory if needed
        remote_dir=$(dirname "$remote_path")
        ssh -i $SSH_KEY $VPS_HOST "mkdir -p $remote_dir"
        
        # Sync the file
        rsync -avz --progress -e "ssh -i $SSH_KEY" \
            "$file" \
            "$VPS_HOST:$remote_path"
    done
fi

echo ""
echo "🔄 Checking live trading status..."
ssh -i $SSH_KEY $VPS_HOST << 'EOF'
    echo "Current screen sessions:"
    screen -ls
    
    echo ""
    echo "Current processes:"
    ps aux | grep -E "live_trading|binance.*bot|python.*htf" | grep -v grep
    
    echo ""
    read -p "Do you want to restart the live trading engine? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping live trading in screen session 'htf_binance'..."
        
        # Kill any existing screen session
        screen -S htf_binance -X quit 2>/dev/null || true
        sleep 2
        
        echo "✅ Starting live trading in new screen session..."
        cd /root/thevolumeainative/trading_system/Binance_Futures_Trading
        screen -dmS htf_binance bash -c "cd /root/thevolumeainative/trading_system/Binance_Futures_Trading && echo CONFIRM | python3 engine/htf_confluence_live_engine.py --live 2>&1 | tee /tmp/htf_engine.log"
        
        sleep 3
        echo ""
        echo "📊 Screen sessions:"
        screen -ls
        
        echo ""
        echo "📊 Running processes:"
        ps aux | grep -E "live_trading|binance.*bot|python.*htf" | grep -v grep
        
        echo ""
        echo "📜 Last 30 lines of log:"
        tail -30 /tmp/htf_engine.log 2>/dev/null || echo "No log file yet, waiting..."
        sleep 2
        tail -30 /tmp/htf_engine.log 2>/dev/null || echo "Still starting up..."
    else
        echo "❌ Restart cancelled"
    fi
EOF

echo ""
echo "✅ Deployment complete!"
