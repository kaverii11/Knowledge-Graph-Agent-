# Data

Seven real SEBI circulars (public, from sebi.gov.in → Legal → Circulars), used for the demo graph and the evals.
The PDFs are not committed; download them into this folder with these names:

| File | Circular | Date | Source |
|---|---|---|---|
| `fpi_kyc_review_2026.pdf` | HO/19/34/14(10)2026-AFD-POD3/I/24814/2026, Rationalisation of periodicity of KYC review for FPIs | Oct 30, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/oct-2026/1793358139677.pdf |
| `fpi_digital_poa_2026.pdf` | HO/19/34/14(8)2026-AFD-POD2/I/19251/2026, Acceptance of digitally signed Power of Attorney from FPIs | Aug 20, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/aug-2026/1787224027885.pdf |
| `dt_timeline_extension_2026.pdf` | HO/(201)2026-DDHS-POD1/I/24467/2026, Extension of timeline for Debenture Trustees (SBU framework) | Oct 27, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/oct-2026/1793167964839.pdf |
| `fpi_gsec_compliance_2026.pdf` | HO/(485)2026-AFD-POD2/I/20296/2026, Ease of regulatory compliances for FPIs investing only in Government Securities | Sep 7, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/sep-2026/1788779580343.PDF |
| `debt_isin_review_2026.pdf` | HO/17/11/24(8)2026-DDHS-POD1/I/23125/2026, Review of provisions related to ISIN for debt securities issued on private placement basis | Oct 7, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/oct-2026/1791374056536.pdf |
| `debt_mb_exemption_2026.pdf` | HO/17/11/24(7)2026-DDHS-POD1/I/23122/2026, Exemption from mandatory merchant banker appointment for privately placed debt by certain listed issuers | Oct 7, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/oct-2026/1791374560264.pdf |
| `fatca_crs_reporting_2026.pdf` | HO/38/14/(18)2026-MIRSD-POD2/I/25560/2026, Reporting requirement under FATCA and CRS revised through Income-Tax Act, 2025 | Nov 11, 2026 | https://www.sebi.gov.in/sebi_data/attachdocs/nov-2026/1794392501415.pdf |

`*.json` are the extraction outputs (`main.py`), `*.meta.json` the detected circular number, subject and date,
and `baseline/` the outputs before the extraction fixes (kept so the before/after eval can be re-run).

## Hand corrections to the four circulars added in Nov 2026

The last four circulars were added so the agent eval has multi-hop questions (documents shared by several
circulars). They have no hand-labelled answer key, so after extraction every reference was checked against
the PDF text. The raw model output is kept unchanged in `raw_extraction/`; the corrected files are the ones
loaded into the graph. Corrections (`extraction_method: manual_correction` marks added references):

| Circular | Problem in the model output | Correction |
|---|---|---|
| FPI G-Sec | Page 1 labelled *modifies* the FPI Master Circular, but page 1 only describes an older amendment | relation set to *references* (the real modification, page 2, was right) |
| FPI G-Sec | Page 2 reference titled just "Master Circular" | title set to the FPI Master Circular it means (defined on page 1) |
| FPI G-Sec | "SEBI circular dated September 10, 2025" missed | added (page 1) |
| ISIN | Missed that para 3 replaces paras 1.1-1.3 of the NCS Master Circular | added a *modifies* reference (page 1) |
| ISIN | "Income Tax Act, 1961" also put on page 2; it appears only on page 1 | removed the page-2 entry |
| FATCA/CRS | Circular no. CIR/MIRSD/2/2015 missed (the regex only knows numbers starting with `SEBI/`) | added (page 1) |
| FATCA/CRS | Income-tax Rules, 1962 missed | added (page 1) |
| FATCA/CRS | "IT Rules, 2026" and "Rules 238 to 240 of the IT Rules, 2026" as separate documents | titled "Income-tax Rules, 2026" (the circular defines "IT Rules, 2026" as its short name) |
| FATCA/CRS | One Guidance Note under two titles | one title |

The merchant-banker circular needed no corrections. Problems that came from entity resolution rather than
extraction were fixed in code instead (see the commit history): the graph now comes out the same whatever
order the files are loaded in, "NCS Master Circular" is recognised as the short name of the long title,
"SEBI (...)" and "Securities and Exchange Board of India (...)" are one document, and new spellings of an
act or regulation join the node already in the graph.

Load order used for the eval graph (by date): `fpi_digital_poa_2026`, `fpi_gsec_compliance_2026`,
`debt_isin_review_2026`, `debt_mb_exemption_2026`, `dt_timeline_extension_2026`, `fpi_kyc_review_2026`,
`fatca_crs_reporting_2026`.
