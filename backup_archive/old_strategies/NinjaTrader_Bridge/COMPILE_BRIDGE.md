# How to Compile NinjaTrader Bridge

## Quick Compile Command (Copy & Paste)

```powershell
powershell -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\Roslyn\csc.exe' /target:exe /out:'C:\Users\Jean-Yves\thevolumeainative\trading_system\NinjaTrader_Bridge\NinjaTraderBridge.exe' /reference:'C:\Users\Jean-Yves\thevolumeainative\trading_system\NinjaTrader_Bridge\NinjaTrader.Client.dll' /reference:'C:\Users\Jean-Yves\thevolumeainative\trading_system\NinjaTrader_Bridge\Newtonsoft.Json.dll' 'C:\Users\Jean-Yves\thevolumeainative\trading_system\NinjaTrader_Bridge\NinjaTraderBridge.cs'"
```

## Details

### Compiler Location
```
C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\Roslyn\csc.exe
```

### Required DLLs (in NinjaTrader_Bridge folder)
- `NinjaTrader.Client.dll` - NinjaTrader API
- `Newtonsoft.Json.dll` - JSON serialization

### Source File
- `NinjaTraderBridge.cs`

### Output
- `NinjaTraderBridge.exe`

## Notes
- DO NOT use the old .NET Framework 4.0 compiler (`C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe`) - it only supports C# 5 and doesn't support string interpolation (`$""`)
- Must use Visual Studio 2022 Roslyn compiler for C# 6+ features
