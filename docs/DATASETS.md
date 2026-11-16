# Extra datasets to extend the project

**Status:** Dunnhumby (`src/campaign_uplift.py`) and the multi-category store (`src/generalization_check.py`) are downloaded and analysed. Online Retail (the 12-month 2010-2011 version) is downloaded and analysed (`src/online_retail.py`).

None of these can be merged with the cosmetics store (different customers, different brands). Their value is to **test whether the methods hold elsewhere** and to cover things this dataset cannot (a longer time window, returns, promotions). Put downloads in `data_external/` (git-ignored).

Check each page's licence/terms and file names before downloading; versions on Kaggle change.

| Priority | Dataset | Why | Size | Ready to run |
| --- | --- | --- | --- | --- |
| 1 | **Online Retail / Online Retail II** (UCI Machine Learning Repository, also on Kaggle) | 12 months (Dec 2010 – Dec 2011; the II version has 24) of invoices with customer IDs and returns. Fixes the "only five months" limit: real retention, CLV validated on a 6-month holdout, return rates. | about 45 MB | `python src/online_retail.py data_external/online_retail_II.csv` |
| 2 | **eCommerce behavior data from multi-category store** (Kaggle, by Michael Kechinov / REES46) | Same schema as the cosmetics data, so the whole pipeline runs unchanged. Seven months of a different store; many products have `category_code`, which answers the missing-categories limit. | several GB per month; start with ONE month file | `python src/pipeline.py --data-dir data_external/multicat --out-dir outputs/multicat` |
| 3 | **Dunnhumby "The Complete Journey"** (Kaggle) | Contains coupons, campaigns and demographics, i.e. real marketing-treatment data. This is the only way to replace the *assumed* campaign uplift with measured uplift. | a few hundred MB | needs a small adapter (tell me when you have it) |
| optional | **Olist Brazilian E-Commerce** (Kaggle) | Orders, payments, product categories (English), reviews and delivery times. Useful for review/delivery analysis and CLV with categories. | about 120 MB | needs an adapter |

## Steps

1. Download from Kaggle (needs a free account): open the dataset page, click **Download**, unzip into `data_external/<name>/`.
2. Run the command in the table. Results go to `outputs/online_retail/` or `outputs/multicat/`.
3. Compare with the cosmetics results: do repeat rate, segment shape, retention and model lift look similar? State differences in the README rather than hiding them.

## What each will let you claim

- **Online Retail:** "validated lifetime-value model on a 6-month holdout across 12 months of data (24 with the II version)".
- **Multi-category store:** "pipeline and models generalise to a second store without code changes".
- **Dunnhumby:** "estimated campaign uplift from coupon data instead of assuming it".
