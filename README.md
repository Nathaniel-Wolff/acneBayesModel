# acneBayesModel

# Acne Severity Modeling

This project explores how acne severity evolves in response to treatment over time. Using daily-level data on treatment regimens and acne severity normalized to patient baseline, I modeled each day’s severity as a categorical state. I generated metadata representing each patient’s treatment history, binned severity scores using kernel density estimation (KDE), and computed Dirichlet posterior distributions for severity given history. I then tracked change over time using cumulative KL divergence, modeling its slope with piecewise linear regression to identify diminishing returns in treatment effect.

## Overview

- **Goal:** To model and interpret changes in acne severity across treatment timelines with a probabilistic approach.
- **Core Question:** Does repeated use of the same treatment eventually lead to diminishing returns in acne severity improvement? Can we quantify those diminishing returns? 

## Approach

1. **Normalization & KDE Binning:**
   - Normalized acne severity scores by baseline.
   - Used kernel density estimation (KDE) to identify a natural 3-state acne severity system: low, medium, and high severity.
   - Identified these states based on the local maxima and saddle points in the distribution.

2. **Dirichlet Posterior Modeling:**
   - Computed the posterior distribution over severity states, conditioned on treatment history.
   - Visualized uncertainty using 95% credible intervals of each corresponding Beta marginal distribution. 
   - Compared posteriors across histories using KL divergence to quantify the “information gain” from each treatment update. 
   - Consolidated posteriors across histories using cumulative KL divergence to quantify the cumulative “information gain” over all treatment updates. 

3. **Piecewise Linear Regression:**
   - Modeled trends in cumulative KL Divergence across treatment history stretches using piecewise linear regression.
   - Approximated slope of cumulative KL divergence over the entire curve piecewise.  
   - Found diminishing slopes over consecutive segments, suggesting diminishing treatment effect with time.
   - Observed that improvement does not entirely plateau, but marginal gains reduce with successive treatments.

## Key Takeaways

- **Trimodal Severity States:** The KDE of normalized acne severity is bimodal, justifying a 3 category classification (low severity, medium severity, high severity).
- **Tightly Clustered Distributions:** Severity tends to cluster tightly around the two peaks (~60% and ~100% below baseline).
- **Diminishing Returns:** Consecutive treatments provide reduced improvements over time, pointing toward a steady state acne severity.
- **Steady-State Hypothesis:** Eventually, acne severity may converge toward a steady-state distribution determined by non-treatment biological factors like hormones, skin type, or diet.

## Applications

- **Clinical Utility:**
  - Helps clinicians identify when further treatment will have minimal benefit.
  - Assists in advising patients and insurers about the cost-effectiveness of treatment regimens.

- **Automation & Scale:**
  - ChatGPT or other LLMs could be used to:
    - Automate patient reports.
    - Annotate datasets.
    - Summarizing statistical trends across cohorts.

## Next Steps

- **Simulation Mode:**
  - Build synthetic patient models to simulate long-term treatment strategies.
  - Build a patient-accessible interface to explore probabilities of improvements over time, maintaining motivation to continue treatment. 
  - Use a stochastic framework (e.g., Markov models) to incorporate biological fluctuations.

- **Demographic Analysis:**
  - Investigate how treatment responses differ across patient ages, backgrounds, and area of residence.

- **LLM Integration:**
  - Incorporate ChatGPT into the pipeline for automating annotations, planning, and insight generation for patients and clinicians separately.



**Author:** Nathaniel Wolff
**Contact:** [nathanielwolff1818@gmail.com; https://github.com/Nathaniel-Wolff]  
**Status:** Draft complete, polishing in progress  


