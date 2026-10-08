# Customer Segmentation & Personalization Platform

*A case study in evidence-driven recommendation: when the simple baseline wins*

Prepared by Ashok

**Stack:** Python · scikit-learn · implicit ALS · FastAPI · Google Cloud Run · Streamlit · MLflow · Airflow · Evidently

---

## Executive Summary

This project builds an end-to-end personalization platform on the public Instacart grocery dataset. It groups customers by shopping behavior, ranks products for each customer's next basket, and serves both through a token-protected API with a browser demo. It is a portfolio project built to production-style standards, and its most useful finding is not a model that won, but an evaluation that was honest enough to say which model should.

Five baselines and ten tuned alternating-least-squares (ALS) collaborative-filtering trials were compared on a leakage-free validation split. A transparent baseline that ranks a customer's own most-purchased products, filled from their segment's popularity, scored Recall@10 of 0.333 and NDCG@10 of 0.395. The best ALS trial reached 0.191 and 0.208. The simpler method was selected and shipped; on a final holdout consumed exactly once it scored 0.323 and 0.391. The system is deployed with the API on Google Cloud Run and the interface on Streamlit Community Cloud.

## 1. Background and Problem

In grocery shopping, a large share of what a customer buys next is something they have bought before. That changes what a good recommender looks like: novelty is not the goal, and a sophisticated latent-factor model has to beat a simple count of what the customer already buys. The project asked two questions. Which approach ranks the products in a customer's next completed basket most accurately? And how can the winner be shipped as a service without leaking future information into evaluation or overstating what the evidence supports?

The dataset also constrains the work. It has no prices, no calendar dates, and no demographics. Monetary RFM (recency, frequency, monetary value) and calendar recency are therefore impossible, and the project uses behavioral features instead and says so, rather than relabeling proxies as spend or churn.

## 2. Design Philosophy

- **Evidence over complexity.** Model choice follows a declared validation rule, not the appeal of a more advanced algorithm. ALS was built, tuned, and kept as a documented comparison, but it was not deployed because the evidence did not favor it.
- **No leakage by construction.** Each customer's last two baskets are reserved: earlier baskets train the population models, the second-last selects among methods, and the last is a one-time final holdout. The split is sequential per customer rather than random, because a random row split would let a customer's future purchases into training while testing on their past.
- **Claims bounded by evidence.** Weak cluster separation, a tiny gain over the nearest baseline, and the absence of any measured business uplift are all stated in the documentation rather than left for a reader to discover.
- **Learning separated from serving.** At request time the API uses frozen population artifacts. A new basket can change that customer's features, segment, and ranking, but it never refits cluster centroids, popularity tables, or item factors.
- **Reproducible, integrity-checked releases.** Segmentation and recommendation models ship as paired, checksummed bundles, and the runtime refuses to start on a mismatched or modified release.

## 3. Architecture

### Data preparation

Six raw Instacart tables are validated and reduced to a seeded sample of 10,000 customers with complete order histories. A sequence manifest assigns each order a role (train, validation, or final). For a customer with six baskets, baskets 1–4 train the population models, basket 5 selects the method, and basket 6 is reserved for final evaluation.

### Features and segmentation

Each customer is described by 32 raw features (30 used by the model): order count, basket size, repeat share, gaps between orders, 21 department shares, and cyclical hour-of-day and day-of-week encodings. Skewed counts are log-transformed and feature groups are balanced so that 21 department columns do not dominate the distance calculation. K-Means was fitted for K = 2 to 8, and K = 4 was chosen explicitly for its interpretable profiles. Customers with fewer than three orders (1,160 of 10,000) are returned as insufficient history rather than assigned an unreliable persona.

| Segment (reviewed name) | Customers | Evidence from measured profile |
|---|---:|---|
| Frequent repeat shoppers | 2,460 | Highest mean order count (36.4) and repeat share (0.648) |
| Morning beverage/snack shoppers | 1,905 | Beverages and snacks over-indexed; peak ordering at 10:00–11:00 |
| Produce-focused shoppers | 2,123 | Produce share 0.485, about 20 points above the population mean |
| Later-day shoppers | 2,352 | Highest order shares at 16:00–17:00 |

*Names were assigned after inspecting cluster profiles, not before. Cluster IDs are arbitrary and carry no ranking.*

Separation is weak: the K = 4 silhouette score is 0.090, below the 0.102 of K = 2. Results are highly stable across random seeds (pairwise agreement 0.976–0.998) but sensitive to how feature groups are weighted (0.35–0.52). The segments are therefore descriptive groupings of shopping tendencies, not ground-truth customer classes.

### Recommendation

Customer-product purchase counts form a sparse matrix of 10,000 by 34,636 products with 606,674 nonzero entries. Five baselines were evaluated: global popularity, segment popularity, most recent basket, personal frequency, and personal frequency with segment fill. Ten bounded ALS trials varied factors, regularization, iterations, and confidence scaling, with purchases treated as implicit feedback and repeat counts converted to a log-scaled confidence weight. Repeat purchases remain eligible, since repeating is the behavior being predicted.

### Serving and deployment

- **API:** FastAPI with strict Pydantic contracts, Bearer-token authentication that fails closed when no token is configured, separate liveness and readiness checks, and a supplied-history route that scores any complete history without writing it.
- **Cloud Run:** scale-to-zero container running as a non-root user with pinned dependencies. Model artifacts live in a private Cloud Storage archive pinned to an exact object generation and verified by SHA-256 at startup, and synthetic demo histories persist in Firestore, separate from benchmark data.
- **CI/CD:** GitHub Actions builds and tests the images, and deployment uses Workload Identity Federation so no long-lived cloud key is stored.
- **Interface:** a Streamlit app on Community Cloud that only calls the API, with no model loading or data access of its own.

### MLOps tooling

MLflow records experiment runs and registers complete release bundles. Airflow orders a five-task manual batch workflow that defaults to auditing the saved release. Evidently produces feature-drift reports, and a promotion gate checks whether a candidate meets documented floors and required reviews. These tools run on demand; none is presented as a continuously operating production service, and a completed training run never deploys itself.

## 4. Results

Validation comparison across all 10,000 customers at K = 10, with equal weight per customer:

| Method | Recall@10 | NDCG@10 |
|---|---:|---:|
| Global popularity | 0.0697 | 0.0963 |
| Segment popularity | 0.0734 | 0.0985 |
| Most recent basket + global fill | 0.2606 | 0.3191 |
| Best ALS trial of ten | 0.1906 | 0.2081 |
| Personal frequency + global fill | 0.3332 | 0.3953 |
| **Personal frequency + segment fill (selected)** | **0.3334** | **0.3954** |

The selected method's edge over plain personal frequency is 0.0002 in Recall@10, which is not a demonstrated gain. It was selected because it scored highest under the declared rule (validation NDCG@10), and its segment fill supplies items when a customer's personal list is short. The substantive result is the gap above ALS. A plausible reading is that in repeat-heavy grocery data a customer's own purchase counts are a stronger signal than shared latent structure, though the project did not run an experiment to confirm why.

Final holdout, selected method only, scored once: Recall@10 0.323, NDCG@10 0.391, candidate coverage 0.352, with 12,187 unique products recommended and no short lists. The final targets were not reopened afterward, and further model selection would require a new reviewed evaluation protocol.

## 5. Key Engineering Decisions

| Decision | Alternative considered | Why this one |
|---|---|---|
| Sequential per-customer split with a one-time final holdout | Random row split | Prevents future purchases leaking into training; calendar split impossible without dates |
| Behavioral features instead of RFM | Monetary RFM, calendar recency | No prices or dates in the data; avoids implying spend or churn |
| Explicit K = 4 with a minimum of three orders | Highest silhouette (K = 2); assign every customer | Interpretable profiles, with weak separation disclosed; unsupported customers stay unassigned |
| Ship the baseline, keep ALS as comparison | Deploy ALS as the more advanced model | Validation evidence favored the baseline |
| Frozen population state, history-based inference | Refit on every basket | Personalizes without request-time training or leakage |
| API separate from the UI | UI owns models and data | Independently callable contracts and one consistent inference path |
| Local on-demand MLflow, Airflow, and Evidently | Hosted always-on MLOps stack | Fits portfolio cost and scope, and avoids calling historical replay a live pipeline |

## 6. Technology Stack

| Layer | Technologies |
|---|---|
| Data and modeling | Python, pandas, scikit-learn (K-Means), SciPy sparse matrices, implicit (ALS) |
| Serving | FastAPI, Pydantic, Uvicorn, Bearer-token authentication |
| Cloud | Google Cloud Run, Artifact Registry, Cloud Storage, Firestore, Secret Manager, Workload Identity Federation |
| Interface | Streamlit, Streamlit Community Cloud |
| MLOps | MLflow (tracking and registry), Apache Airflow, Evidently |
| Delivery | Docker (non-root, pinned dependencies), GitHub Actions |

## 7. Validation and Operational Evidence

The recorded acceptance run executed 65 tests, and local API and UI acceptance checks passed. Deployed checks against the live API covered seven groups, including authentication rejection, repeatability of predictions, and two concurrent clients. Ten recommendation requests had a median client latency of 519 ms and a maximum of 809 ms. These are smoke measurements over a wide-area network, not a capacity claim. A hosted walkthrough of the published interface confirmed readiness, both model versions, an existing customer's ten recommendations and segment, and saved demo history.

One failure is worth recording. The first CI-built deployment was rejected at startup because a Linux checkout normalized the historical line endings of the model source, which broke the byte-level provenance check. Rather than weakening the check, the fix preserved exact source bytes through Git attributes and added a build-time guard, keeping the accepted release identities intact.

## 8. What This Project Demonstrates

- **Evaluation rigor:** leakage-free sequential splitting, a consumed final holdout, and model selection by a declared rule, including selecting the simpler model when it won.
- **End-to-end ownership:** from data validation and feature design through a deployed, authenticated, observable service with keyless CI/CD.
- **Honest reporting:** weak separation, negligible gains, and unbuilt components are stated alongside the results.
- **Production habits from a QA background:** contract validation, integrity checks, failure-boundary testing, and reproducible releases applied to an ML system.


