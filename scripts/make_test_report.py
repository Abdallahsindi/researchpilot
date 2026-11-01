"""Generate the synthetic test report used for the web UI, the API demo, and tests.

    uv run python scripts/make_test_report.py

Writes to sample_data/pdfs/:
    researchpilot_test_report.pdf              5-page synthetic report (with a spec table)
    researchpilot_test_report_cases.json       questions + expected pages (machine-readable)
    researchpilot_test_questions.md            the same, for people

Every figure is fictional and internally consistent (e.g. 1,000 kWh x $420/kWh = $420,000).
Before writing, the script parses the generated PDF and checks that each question's expected
facts really are on the expected page, so the page references cannot drift from the PDF.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.pdf import parse_pdf  # noqa: E402
from tests.pdf_factory import build_pdf  # noqa: E402

OUT_DIR = ROOT / "sample_data" / "pdfs"
PDF_NAME = "researchpilot_test_report.pdf"
DISCLAIMER = (
    "SYNTHETIC TEST DATA. This report was generated to test ResearchPilot. Harbor Point "
    "Community College, its staff, and every figure below are fictional and must not be "
    "cited as real research."
)

PAGES = [
    (
        "1. Introduction and Project Objectives",
        f"""{DISCLAIMER}
The Harbor Point Community College Energy Storage Pilot added a battery energy storage system to the campus microgrid in January 2025. The microgrid already had an 800 kW rooftop solar array serving the Science Building and the Library. Without storage, most of the midday solar output was exported to the grid at a low price, while electricity was bought back at a high price during the evening peak.
The pilot had three objectives:
Objective 1: Store midday solar energy and use it during the evening peak, reducing grid purchases.
Objective 2: Keep the Library's emergency lighting and network equipment running for at least four hours during a grid outage.
Objective 3: Measure the real efficiency, capacity, safety, and costs of the system, to decide whether to install storage at the college's second campus.
The pilot ran for twelve months, from January to December 2025. Section 2 compares the battery options that were considered, Section 3 reports measured efficiency and capacity, Section 4 covers safety and cost, and Section 5 lists the limitations of this report.""",
    ),
    (
        "2. Battery Options and Specifications",
        """Three battery technologies were compared before purchase. Table 1 lists the example specifications supplied for each option. All three options were sized for 1,000 kWh of usable capacity.
Table 1. Example battery specifications (synthetic test data). Capacity is usable capacity, efficiency is round-trip efficiency, and price is in USD per kWh of capacity.
    Option  Chemistry                Capacity   Efficiency  Cycle life  Price
    A       Lithium iron phosphate   1,000 kWh  92%         6,000       USD 420
    B       Nickel manganese cobalt  1,000 kWh  94%         3,500       USD 390
    C       Vanadium redox flow      1,000 kWh  75%         15,000      USD 610

Option A, lithium iron phosphate (LFP), was selected. It was not the cheapest option, but it combined a cycle life of 6,000 cycles with a lower fire risk than Option B. Option B, nickel manganese cobalt (NMC), had the highest efficiency and the lowest price, but its cycle life of 3,500 cycles was much shorter and it was rejected mainly for safety reasons (see Section 4). Option C, a vanadium redox flow battery, had the longest cycle life, but its 75% round-trip efficiency would have wasted too much of the stored solar energy.""",
    ),
    (
        "3. Efficiency and Capacity Results",
        """Measured results for the selected LFP system (Option A) over the twelve-month pilot:
Round-trip efficiency: the average measured round-trip efficiency was 90.5%, slightly below the rated 92%. Most of the difference came from inverter losses and the energy used by the cooling system.
Energy throughput: the battery charged 361 MWh of midday solar energy and discharged 327 MWh during the evening peak (327 / 361 = 90.5%). This corresponds to 330 full equivalent cycles, an average of 0.9 cycles per day.
Capacity: the usable capacity was 1,000 kWh at the start of the pilot and 982 kWh after twelve months, a capacity fade of 1.8% in the first year.
Power: the maximum discharge power is 250 kW, so a full discharge takes about four hours.
Outage performance: during the only grid outage of the year, on 14 October 2025, lasting 2 hours and 30 minutes, the battery kept the Library's emergency lighting and network equipment running without interruption, as required by Objective 2.""",
    ),
    (
        "4. Safety and Cost Considerations",
        """Safety: lithium iron phosphate cells have higher thermal stability than nickel manganese cobalt cells, which was the main reason for choosing Option A over Option B. The battery is housed in a separate container placed 3 metres from the Library, with gas detection and a clean-agent fire suppression system. Two minor alarms occurred during the pilot: a cooling-fan fault in March and a door-sensor fault in August. Neither involved overheating, and there were no safety incidents.
Cost: Table 2 shows the installed cost of the system.
Table 2. Installed cost (synthetic test data)
    Cost item                                   USD
    Battery modules (1,000 kWh x USD 420/kWh)   420,000
    Installation, inverter, grid integration    100,000
    Total installed cost                        520,000

The measured savings in 2025 were USD 61,000, from lower evening electricity purchases and lower peak-demand charges. The simple payback period is therefore about 8.5 years (520,000 / 61,000). These figures do not include the annual maintenance contract (see Section 5).""",
    ),
    (
        "5. Limitations and Next Steps",
        """This report has several limitations that should be considered before using its results:
Short period: only twelve months of data were collected, so long-term capacity fade is uncertain.
Single site and mild year: the results come from one campus, and the 2025 winter was milder than average, so they may not transfer to the second campus.
Maintenance cost: the cost figures exclude the maintenance contract of USD 8,000 per year. Including it, the simple payback period would be about 9.8 years (520,000 / 53,000).
No environmental assessment: no life-cycle environmental assessment of the batteries was carried out.
Supplier figures: the specifications in Table 1 are example values from supplier quotes in this synthetic scenario, not independent laboratory tests.
Next steps: the college will collect a second year of data in 2026 before deciding whether to install the same system at its second campus.""",
    ),
]

CASES = [
    {
        "id": "efficiency",
        "question": "What round-trip efficiency did the battery system achieve during the pilot?",
        "answerable": True,
        "expected_pages": [3],
        "expected_facts": ["90.5%", "rated 92%"],
        "section": "3. Efficiency and Capacity Results",
    },
    {
        "id": "selected-option",
        "question": "Which battery option was selected and why?",
        "answerable": True,
        "expected_pages": [2, 4],
        "expected_facts": [
            "Option A, lithium iron phosphate (LFP), was selected",
            "thermal stability",
        ],
        "section": "2. Battery Options and Specifications; 4. Safety and Cost Considerations",
    },
    {
        "id": "cycle-life",
        "question": "How does the cycle life of the NMC battery compare with lithium iron phosphate?",
        "answerable": True,
        "expected_pages": [2],
        "expected_facts": ["3,500", "6,000"],
        "section": "2. Battery Options and Specifications (Table 1)",
    },
    {
        "id": "cost-payback",
        "question": "What was the total installed cost and the simple payback period?",
        "answerable": True,
        "expected_pages": [4],
        "expected_facts": ["520,000", "8.5 years"],
        "section": "4. Safety and Cost Considerations (Table 2)",
    },
    {
        "id": "capacity-fade",
        "question": "How much usable capacity was left after the first year?",
        "answerable": True,
        "expected_pages": [3],
        "expected_facts": ["982 kWh", "1.8%"],
        "section": "3. Efficiency and Capacity Results",
    },
    {
        "id": "limitations",
        "question": "What are the main limitations of this report?",
        "answerable": True,
        "expected_pages": [5],
        "expected_facts": ["only twelve months of data", "USD 8,000 per year"],
        "section": "5. Limitations and Next Steps",
    },
    {
        "id": "unanswerable-ev",
        "question": "How many electric vehicles does the campus charge each day?",
        "answerable": False,
        "expected_pages": [],
        "expected_facts": [],
        "section": None,
    },
    {
        "id": "unanswerable-manufacturer",
        "question": "Which company manufactured the battery cells?",
        "answerable": False,
        "expected_pages": [],
        "expected_facts": [],
        "section": None,
    },
]

UI_DEMO_REQUEST = {
    "question": "Based on the pilot results, should the college reuse the lithium iron "
    "phosphate battery at its second campus?",
    "criteria": ["efficiency", "safety", "cost"],
    "constraints": ["limited capital budget"],
}


def _normalise(text: str) -> str:
    return " ".join(text.split())


def verify(pdf: bytes) -> None:
    """Fail loudly if any expected fact is not on its expected page in the produced PDF."""
    document = parse_pdf(pdf, PDF_NAME)
    pages = {p.page_number: _normalise(p.text) for p in document.pages}
    assert document.page_count == len(PAGES) and not document.warnings
    for case in CASES:
        for fact in case["expected_facts"]:
            found = [n for n, text in pages.items() if _normalise(fact) in text]
            assert set(found) & set(case["expected_pages"]), (case["id"], fact, found)


def questions_markdown() -> str:
    lines = [
        "# Test questions for `researchpilot_test_report.pdf`",
        "",
        "The PDF is **synthetic test data** about a fictional college's battery storage pilot.",
        "Page numbers were verified against the generated PDF by `scripts/make_test_report.py`.",
        "",
        "## Answerable questions",
        "",
        "| Question | Expected evidence | Page(s) | Key facts |",
        "|---|---|---|---|",
    ]
    for c in CASES:
        if c["answerable"]:
            facts = "; ".join(c["expected_facts"])
            pages = ", ".join(map(str, c["expected_pages"]))
            lines.append(f"| {c['question']} | {c['section']} | {pages} | {facts} |")
    lines += [
        "",
        "## Questions the PDF cannot answer",
        "",
        "The PDF says nothing about these. Expected behaviour: the analysis status is",
        "`insufficient_evidence` (the Analyst abstains, or retrieval finds nothing relevant), the",
        "report presents no verified findings. Any off-topic claim the model makes is shown only",
        "as an unverified observation.",
        "",
    ]
    lines += [f"- {c['question']}" for c in CASES if not c["answerable"]]
    lines += [
        "",
        "## Suggested web-UI demo",
        "",
        f"- **Question:** {UI_DEMO_REQUEST['question']}",
        f"- **Criteria (one per line):** {', '.join(UI_DEMO_REQUEST['criteria'])}",
        f"- **Constraint:** {UI_DEMO_REQUEST['constraints'][0]}",
        "",
        "Findings, statuses, and pages depend on the model run. The report marks each claim's",
        "verification status, lists criteria with no verified evidence, and withholds the",
        "model's recommendation when a requested criterion has no verified evidence. When all",
        "criteria are covered, the recommendation is labelled as a judgement, not a fact.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    pdf = build_pdf(
        [text for _, text in PAGES],
        page_titles=[title for title, _ in PAGES],
        metadata={
            "Title": "Campus Energy Storage Pilot - Synthetic Test Report",
            "Author": "ResearchPilot test data",
        },
    )
    verify(pdf)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / PDF_NAME).write_bytes(pdf)
    cases = {"pdf": PDF_NAME, "ui_demo_request": UI_DEMO_REQUEST, "cases": CASES}
    (OUT_DIR / "researchpilot_test_report_cases.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )
    (OUT_DIR / "researchpilot_test_questions.md").write_text(questions_markdown())
    print(f"Wrote {OUT_DIR / PDF_NAME} ({len(pdf)} bytes, {len(PAGES)} pages) and companion files")


if __name__ == "__main__":
    main()
