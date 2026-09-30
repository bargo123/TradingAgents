import json
from datetime import datetime, timedelta, timezone

from cli.forex_hft_replay import main


UTC = timezone.utc


def test_replay_cli_writes_scalar_report(tmp_path):
    start = datetime.now(UTC) - timedelta(hours=1)
    lines = ["timestamp,symbol,bid,ask,point,sequence"]
    for index in range(10):
        stamp = start + timedelta(seconds=index)
        lines.append(f"{stamp.isoformat().replace('+00:00','Z')},EURUSD,{1.1 + index * 0.00001},{1.1001 + index * 0.00001},0.00001,{index}")
    source = tmp_path / "ticks.csv"
    source.write_text("\n".join(lines), encoding="utf-8")
    output = tmp_path / "report.json"
    assert main(["--ticks", str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["executed"] is False
    assert report["ticks_processed"] == 10

