# Test questions for `researchpilot_test_report.pdf`

The PDF is **synthetic test data** about a fictional college's battery storage pilot.
Page numbers were verified against the generated PDF by `scripts/make_test_report.py`.

## Answerable questions

| Question | Expected evidence | Page(s) | Key facts |
|---|---|---|---|
| What round-trip efficiency did the battery system achieve during the pilot? | 3. Efficiency and Capacity Results | 3 | 90.5%; rated 92% |
| Which battery option was selected and why? | 2. Battery Options and Specifications; 4. Safety and Cost Considerations | 2, 4 | Option A, lithium iron phosphate (LFP), was selected; thermal stability |
| How does the cycle life of the NMC battery compare with lithium iron phosphate? | 2. Battery Options and Specifications (Table 1) | 2 | 3,500; 6,000 |
| What was the total installed cost and the simple payback period? | 4. Safety and Cost Considerations (Table 2) | 4 | 520,000; 8.5 years |
| How much usable capacity was left after the first year? | 3. Efficiency and Capacity Results | 3 | 982 kWh; 1.8% |
| What are the main limitations of this report? | 5. Limitations and Next Steps | 5 | only twelve months of data; USD 8,000 per year |

## Questions the PDF cannot answer

The PDF says nothing about these. Expected behaviour: the analysis status is
`insufficient_evidence` (the Analyst abstains, or retrieval finds nothing relevant), the
report presents no verified findings. Any off-topic claim the model makes is shown only
as an unverified observation.

- How many electric vehicles does the campus charge each day?
- Which company manufactured the battery cells?

## Suggested web-UI demo

- **Question:** Based on the pilot results, should the college reuse the lithium iron phosphate battery at its second campus?
- **Criteria (one per line):** efficiency, safety, cost
- **Constraint:** limited capital budget

Findings, statuses, and pages depend on the model run. The report marks each claim's
verification status, lists criteria with no verified evidence, and withholds the
model's recommendation when a requested criterion has no verified evidence. When all
criteria are covered, the recommendation is labelled as a judgement, not a fact.
