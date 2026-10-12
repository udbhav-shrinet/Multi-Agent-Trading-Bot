import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.fred_tools import get_macro_snapshot, get_macro_history
from agents.quant_research import macro_score
from agents.market_intelligence import gather_market_intelligence
from agents.quant_research import research_symbol
from agents.portfolio_council import build_portfolio_plan

def test_macro_and_pipeline():
    print("Testing FRED macro snapshot...")
    snapshot = get_macro_snapshot()
    print("Snapshot:", snapshot)
    
    history = get_macro_history(years=1)
    print("History keys count:", len(history.get("t10y2y", {})), len(history.get("vixcls", {})))
    
    score = macro_score(snapshot.get("t10y2y"), snapshot.get("vixcls"))
    print("Macro score:", score)
    
    print("Testing single symbol research (AAPL)...")
    try:
        intel = gather_market_intelligence("AAPL")
        research = research_symbol(intel, snapshot)
        print("Research result for AAPL:", research["direction"], research["score"], research["probability_up"])
    except Exception as e:
        print("Market intelligence/research failed (expected if network restricted):", e)

if __name__ == "__main__":
    test_macro_and_pipeline()
