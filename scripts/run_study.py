import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumo_study.workflow import Study
s = Study("results", profile=sys.argv[1], workers=int(sys.argv[2]))
s.run_all(until=sys.argv[3] if len(sys.argv) > 3 else None)
