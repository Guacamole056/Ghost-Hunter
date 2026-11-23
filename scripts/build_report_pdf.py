"""Export the editable course report, or print the current evaluation with --summary."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
REPORT = DOCS / "Ghost_Hunter_Project_Report.docx"


def make() -> None:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        locations = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "LibreOffice/program/soffice.exe",
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        ]
        executable = next((str(p) for p in locations if p.is_file()), None)
    if not executable:
        raise SystemExit("Open the report DOCX in Word and export it as PDF, or install LibreOffice.")
    output = REPORT.with_suffix(".pdf")
    with TemporaryDirectory(dir=DOCS) as temp:
        folder = Path(temp)
        profile = (folder / "profile").as_uri()
        subprocess.run([executable, f"-env:UserInstallation={profile}", "--headless",
                        "--convert-to", "pdf", "--outdir", str(folder), str(REPORT)], check=True)
        generated = folder / output.name
        if not generated.is_file() or generated.stat().st_size == 0:
            raise SystemExit("LibreOffice did not produce the PDF. Check the Word file and try again.")
        generated.replace(output)
    print(f"Saved {output}")


def evaluation_summary() -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as error:
        raise SystemExit("Install the optional summary dependency with python -m pip install -r requirements-report.txt") from error

    data = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    config = data["configuration"]
    styles = getSampleStyleSheet()
    rows = [["Metric", "HMM-guided", "Sensor-only"]]
    for label, key, unit in (
        ("Episodes", "episodes", ""), ("Captures", "captures", ""),
        ("Capture rate", "capture_rate_pct", "%"), ("Mean turns at capture", "mean_turns_to_capture", ""),
        ("Mean moves at capture", "mean_movement_to_capture", ""), ("Mean location error", "mean_localization_error", " cells"),
        ("Mean time per episode", "mean_runtime_ms", " ms"),
    ):
        values = [data[mode][key] for mode in ("ai", "baseline")]
        rows.append([label, *["N/A" if value is None else f"{value}{unit}" for value in values]])
    table = Table(rows, colWidths=[72*mm, 43*mm, 43*mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story = [
        Paragraph("Ghost Hunter evaluation summary", styles["Title"]),
        Paragraph(f"{config['episodes_per_strategy']} paired scenarios, seed {config['master_seed']}, "
                  f"{config['grid_size']} by {config['grid_size']} board, {config['max_turns']} turns, "
                  f"zero-noise probability {config['sensor_accuracy']:.2f}.", styles["BodyText"]),
        Spacer(1, 6*mm), table, Spacer(1, 4*mm),
        Paragraph("Turns and moves to capture include successful games only. Localization error and runtime include all episodes.", styles["BodyText"]),
        Spacer(1, 6*mm), Image(str(RESULTS / "01_capture_rate.png"), width=132*mm, height=79*mm),
        PageBreak(), Image(str(RESULTS / "02_mean_turns.png"), width=151*mm, height=90*mm),
        Spacer(1, 6*mm), Image(str(RESULTS / "03_localization_error.png"), width=156*mm, height=91*mm),
    ]
    output = RESULTS / "Evaluation_Summary.pdf"
    with TemporaryDirectory(dir=RESULTS) as temp:
        generated = Path(temp) / output.name
        SimpleDocTemplate(str(generated), pagesize=A4, title="Ghost Hunter evaluation summary").build(story)
        generated.replace(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="print the current experimental results")
    if parser.parse_args().summary:
        evaluation_summary()
    else:
        make()
