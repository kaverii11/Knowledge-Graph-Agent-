# Data

Three real SEBI circulars (public, from sebi.gov.in → Legal → Circulars), used for the demo graph and the evals.
The PDFs are not committed; download them into this folder with these names:

| File | Circular | Date | Source |
|---|---|---|---|
| `fpi_kyc_review_2026.pdf` | HO/19/34/14(10)2026-AFD-POD3/I/24814/2026, Rationalisation of periodicity of KYC review for FPIs | Oct 30, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/oct-2026/1793358139677.pdf |
| `fpi_digital_poa_2026.pdf` | HO/19/34/14(8)2026-AFD-POD2/I/19251/2026, Acceptance of digitally signed Power of Attorney from FPIs | Aug 20, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/aug-2026/1787224027885.pdf |
| `dt_timeline_extension_2026.pdf` | HO/(201)2026-DDHS-POD1/I/24467/2026, Extension of timeline for Debenture Trustees (SBU framework) | Oct 27, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/oct-2026/1793167964839.pdf |

`*.json` are the extraction outputs (`main.py`), `*.meta.json` the detected circular number, subject and date,
and `baseline/` the outputs before the extraction fixes (kept so the before/after eval can be re-run).
