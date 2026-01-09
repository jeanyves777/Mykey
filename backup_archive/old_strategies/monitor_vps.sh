#!/bin/bash
# Monitor live trading on VPS

VPS_HOST="root@72.62.3.184"
SSH_KEY="~/.ssh/id_rsa_vps"

echo "📊 Monitoring VPS Trading System..."
echo "=================================="
echo ""

ssh -i $SSH_KEY $VPS_HOST << 'EOF'
    echo "📺 Screen sessions:"
    screen -ls
    echo ""
    
    echo "🔄 Running processes:"
    ps aux | grep -E "python.*htf|live_trading|binance" | grep -v grep
    echo ""
    
    echo "💰 Recent trades (last 50 lines of log):"
    tail -50 /tmp/htf_engine.log 2>/dev/null || echo "No log file found"
    echo ""
    
    echo "📈 System resources:"
    echo "CPU & Memory:"
    ps aux | grep python | grep -v grep | awk '{print $2, $3, $4, $11}' | head -5
    echo ""
    
    echo "💾 Disk usage:"
    df -h / | tail -1
    echo ""
    
    echo "⏰ System uptime:"
    uptime
EOF

echo ""
echo "=================================="
echo "To attach to screen: ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 -t 'screen -r htf_binance'"
echo "To view live log: ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 'tail -f /tmp/htf_engine.log'"
