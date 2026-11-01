"""Generate the synthetic sample PDF and its evaluation cases used by the Kaggle notebook.

    python scripts/make_sample_pdf.py

Everything in the document is fictional. Facts are specific and invented on purpose: if the
model answers correctly, it must have read the PDF rather than recalled training data.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.pdf_factory import build_pdf  # noqa: E402

OUT_DIR = ROOT / "notebooks" / "sample_data"
DISCLAIMER = (
    "SYNTHETIC SAMPLE DOCUMENT created for the ResearchPilot demo. Lakeside Technical "
    "University, its people, and all figures in this report are fictional."
)

PAGES = [
    (
        "1. Project Overview",
        f"""{DISCLAIMER}
The Lakeside Technical University Campus Microgrid Pilot ran from March 2024 to February 2025 on the university's north campus. The pilot connected three buildings, the Engineering Hall, the Library Annex, and the Student Services Center, to a shared solar and battery system that can operate either connected to the regional grid or in island mode during outages.
The project had three goals: reduce electricity costs for the connected buildings, keep critical services running during grid outages, and give engineering students a working energy system to study.
The solar array has a peak capacity of 1.2 megawatts and is installed on the rooftops of the three buildings and on a covered parking structure next to the Library Annex. The pilot was led by the university Facilities Office with support from the Department of Electrical Engineering.""",
    ),
    (
        "2. Battery Storage Design",
        """The microgrid uses a lithium iron phosphate (LFP) battery system with a usable capacity of 2.4 megawatt-hours, housed in two outdoor containers beside the Student Services Center.
The project team selected LFP chemistry over nickel manganese cobalt (NMC) for two main reasons. First, LFP cells have a lower risk of thermal runaway, which mattered because the containers sit close to occupied buildings. Second, LFP cells offered a longer expected cycle life, rated at about 6,000 full cycles, which suits a system that charges and discharges every day.
The trade-off is lower energy density: the LFP system needed roughly 20 percent more floor space than a comparable NMC system would have required. The team judged the extra space acceptable because land next to the containers was available.
The battery management system limits daily depth of discharge to 85 percent to extend battery life.""",
    ),
    (
        "3. Costs and Results",
        """The total installed cost of the pilot was 3.1 million US dollars, of which 1.3 million was covered by a regional clean-energy grant. The remaining 1.8 million came from the university capital budget.
During the twelve-month pilot, electricity bills for the three connected buildings fell by 18 percent compared with the previous year, after adjusting for weather. The university estimates a simple payback period of about 9 years for its share of the investment.
Reliability results were also positive. The regional grid had four outages during the pilot, the longest lasting 3 hours and 40 minutes. In every case the microgrid switched to island mode and kept the Library Annex and Student Services Center powered without interruption.
Students used operating data from the microgrid in two courses, Power Systems Analysis and Energy Data Science.""",
    ),
    (
        "4. Risks, Limitations, and Next Steps",
        """The pilot identified three main risks. Inverter firmware updates caused two short unplanned shutdowns in the first quarter, which led the team to schedule updates only during weekend maintenance windows. Snow cover reduced solar output by more than half on several winter days. Finally, the evaluation covered only one year, so long-term battery degradation is still uncertain.
The Facilities Office recommends expanding the microgrid to the Sports Center and the Chemistry Building in 2026, subject to funding approval. It also recommends installing a weather-based forecasting tool to plan battery charging ahead of cloudy days.
This report does not include an environmental life-cycle assessment of the batteries, and it does not compare the pilot against buying renewable energy certificates.""",
    ),
]

EVAL_CASES = [
    {
        "id": "battery-chemistry",
        "question": "Which battery chemistry did the microgrid pilot use, and why was it chosen?",
        "answerable": True,
        "expected_pages": [2],
    },
    {
        "id": "cost-and-payback",
        "question": "What was the total installed cost of the pilot and its payback period?",
        "answerable": True,
        "expected_pages": [3],
    },
    {
        "id": "solar-capacity",
        "question": "What is the peak capacity of the solar array?",
        "answerable": True,
        "expected_pages": [1],
    },
    {
        "id": "outage-performance",
        "question": "How did the microgrid perform during grid outages?",
        "answerable": True,
        "expected_pages": [3],
    },
    {
        "id": "expansion-plans",
        "question": "Which buildings are recommended for the next expansion?",
        "answerable": True,
        "expected_pages": [4],
    },
    {
        "id": "unanswerable-president",
        "question": "Who is the president of Lakeside Technical University?",
        "answerable": False,
        "expected_pages": [],
    },
    {
        "id": "unanswerable-wind",
        "question": "How many wind turbines does the microgrid use and what is their capacity?",
        "answerable": False,
        "expected_pages": [],
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = build_pdf(
        [text for _, text in PAGES],
        page_titles=[title for title, _ in PAGES],
        metadata={
            "Title": "Campus Microgrid Pilot - Technical Summary (synthetic sample)",
            "Author": "ResearchPilot demo",
        },
    )
    (OUT_DIR / "researchpilot_sample_report.pdf").write_bytes(pdf)
    (OUT_DIR / "sample_eval_cases.json").write_text(json.dumps(EVAL_CASES, indent=2) + "\n")
    print(f"Wrote {OUT_DIR / 'researchpilot_sample_report.pdf'} ({len(pdf)} bytes)")


if __name__ == "__main__":
    main()
