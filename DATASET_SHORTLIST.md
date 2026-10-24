
## Expanded tabular shortlist — 6 October 2026

This screening uses publisher metadata and visible records; files have not been downloaded or audited. Proposed ML targets remain subject to checking missingness, class balance, leakage and access terms.

| Dataset | Catalogue size | Proposed task | Assessment |
| --- | --- | --- | --- |
| Global Findex 2025: Sri Lanka, observations from 2024 | 1,000 respondents; 199 variables | Classify digital-payment use or account ownership from demographic/access variables | Best new classification candidate. Research-use terms on retrieval page. Exclude target-derived fields and target-revealing skip patterns. |
| Global Findex 2021: Sri Lanka | 1,005 respondents; 114 variables; Stata file | Financial inclusion classification | Older alternative; free account required. Separate rounds are not the same-person panel. |
| World Bank Enterprise Survey 2025: Sri Lanka | 607 establishments; 336 variables | Predict sales or classify export participation | Strong business option. Download routed to Enterprise Surveys; check portal access terms and missing-value codes. Select a modest predictor set. |
| FAO Paddy Farmers Survey 2024 | Plot file: 7,182 cases; 254 variables | Predict harvest quantity or standardized yield | Strong agriculture regression option. Plot records are not independent farmers, and include multiple crops/seasons. Filter eligible paddy observations; split by household. Research access conditions apply. |
| Ikman island-wide land CSV | 9 columns; exact row count not audited; posted March 2019 | Predict advertised per-perch land price | Public CSV with area, location, listing attributes, land size and price. Historical asking prices; duplicate URLs and missing listing age visible. Few predictors; reuse licence unverified. |
| Smallholder farmer resilience survey | 300 respondents from 12 Galgamuwa villages | Predict a documented resilience outcome or strategy adoption | CC BY 4.0. Small sample; target and schema need inspection. |
| Colombo A/L student gaming-disorder survey | 395 respondents from four schools | Classify survey-defined gaming disorder | CC BY-NC 3.0. Check positive-class count; exclude questionnaire items/scores defining the label. |
| E-waste technology adoption | 279 employees | Predict adoption score | CC BY 4.0; small survey. Exclude target constituent items. |
| Consumer values and patronage intention | 312 consumers | Predict patronage-intention score | CC BY 4.0; small self-report survey. |
| Kolonnawa Marsh water quality | CSV of 102.3 kB; row count not verified | Predict a measured water-quality variable | Backup option; inspect site/time grouping and measurement schema first. |

Updated tabular-only recommendation: investigate Global Findex 2025 first for classification, Paddy Farmers Survey 2024 for agriculture regression, or Enterprise Survey 2025 for a business topic. Land listings offer a simpler public-CSV backup.

Sources:
- https://microdata.worldbank.org/catalog/7979/data-dictionary
- https://microdata.worldbank.org/catalog/7979/get-microdata
- https://microdata.worldbank.org/catalog/4710/data-dictionary
- https://microdata.worldbank.org/catalog/8264/data-dictionary/F1/?offset=300
- https://microdata.fao.org/index.php/catalog/2724/data-dictionary
- https://microdata.worldbank.org/catalog/6788
- https://gist.github.com/kaveenr/2e9a9c7417a13eedc3d9dbd6bbd05447
- https://data.mendeley.com/datasets/rynj38rnsd/1
- https://data.mendeley.com/datasets/8r2jgm6ygh/3
- https://data.mendeley.com/datasets/5g5x7dhfzj/1
- https://data.mendeley.com/datasets/zyprwprry7/1
- https://zenodo.org/records/19019893
