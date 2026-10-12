import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.fred_tools import get_macro_snapshot, get_macro_history
from agents.quant_research import macro_score
from agents.market_intelligence import gather_market_intelligence
from agents.quant_research import research_symbol

def test_full_macro_integration():
    print("Fetching FRED macro snapshot & history...")
    snapshot = get_macro_snapshot()
    history = get_macro_history(years=1)
    print("Snapshot:", snapshot)
    print("History points - T10Y2Y:", len(history.get("t10y2y", {})), "VIXCLS:", len(history.get("vixcls", {})))
    
    print("Running research_symbol on AAPL with macro_history...")
    try:
        intel = gather_market_intelligence("AAPL")
        res = research_symbol(intel, snapshot, history)
        print("Research result successfully computed!")
        print("Direction:", res["direction"], "Score:", res["score"], "Macro Score Used:", res["macro_score"])
        print("ML Diagnostics:", res["ml_diagnostics"])
    except Exception as e:
        print("Error during research execution:", e)

if __name__ == "__main__":
    test_full_macro_integration()
