# acneBayesModel

# Acne Severity Modeling

This project explores how acne severity evolves in response to treatment over time. Using daily-level data on treatment regimens and acne severity normalized to patient baseline, I modeled each day’s severity as a categorical state. I generated metadata representing each patient’s treatment history, binned severity scores using kernel density estimation (KDE), and computed Dirichlet posterior distributions for severity given history. I then tracked change over time using cumulative KL divergence, modeling its slope with piecewise linear regression to identify diminishing returns in treatment effect.

## Overview

- **Goal:** To model and interpret changes in acne severity across treatment timelines with a probabilistic approach.
- **Core Questions:** Does repeated use of the same treatment eventually lead to diminishing returns in acne severity improvement? Can we quantify those diminishing returns? How do we model acne severity

## Approach

1. **Severity Score Normalization, KDE Binning, & Bootstrapping Validation:**
   - Normalized acne severity scores by baseline.
   - Used kernel density estimation (KDE) to identify a natural 3-state acne severity change system: low, medium, and high severity change.
   - Identified these states based on the local maxima and saddle points in the distribution. Computed quantiles of KDE corresponding to each. 
   - Verified robustness of hard binning with bootstrapping around each corresponding quantiles, inspecting fraction of observed severities changing bins. 
   

2. **Dirichlet Posterior Modeling:**
   - Computed the posterior distribution over severity states, conditioned on treatment history (also known as "Treatment Series to Date"/ttsd).
   - Visualized uncertainty using 95% credible intervals of each corresponding Beta marginal distribution. 
   - Compared posteriors across histories using KL divergence to quantify the “information gain” from each treatment update. 
   - Consolidated posteriors across histories using cumulative KL divergence to quantify the cumulative “information gain” over all treatment updates. 

3. **Cumulative KL Divergence Curve Fitting, Piecewise Linear Regression:**
   - Determined slope of cumulative KL divergence cruve by fitting a univariate spline curve, validated with bootstrapping.  
   - Found diminishing slopes over consecutive segments, suggesting diminishing treatment effect with time.
   - Determined stretches of treatment with non-diminishing, constant slopes subject to threshold (known as "p value threshold"); fit linear regression models to each stretch piecewise. 
   - Observed that improvement does not entirely plateau, but marginal gains reduce with certain successive treatments, while others maintain consistent gains with consecutive use.
   - Plotted piecewise linear regression models to inspect which treatments are responsible for consistent gains, inferring biochemical mechanism of diminishing returns. 

4. **Fitting of State-Space/Statistical Mechanistic Model, Optimization of Parameters with Expectation-Maximization/Kalman Filter Algorithm**
   - Results of item 3 suggested the following SMM Model:
   - \(\begin{pmatrix} B_t \\ I_t \\ S_t \end{pmatrix}\)



     Where $B_t$ refers to bacterial facial load at time t, $I_t$ refers to inflammatory activity load at time t, $S_t$ refers to sebum production at time t. 

   > The Acne System State's components are defined thusly - 
   $$B_t = B_{t-1} + r_{growth} \cdot B_{t-1}\frac{1-B_{t-1}}{K_{CC}}-k_{antibiotics} \cdot days_{antibiotics} \cdot B_{t-1} + k_{sebum} \cdot B_{t-1} \cdot S_{t-1} + noise$$
   $$I_t = I_{t-1} + I_{bacterial \, induction} \cdot B_{t-1} - I_{decay}/T(tstd)\cdot T(tstd) - I_{baseline decay} \cdot I_{t-1} + noise$$
   $$S_t = S_{t-1} + r_{I production} \cdot I_{t-1} -r_{cream \, clean} \cdot cream \, used + noise$$

   > And acne severity evolves according to the following: 
   $$severity_{t} = C \cdot acne \, state_{t} + m_{t-1}$$

   - Implemented Expectation-Maximization/Kalman Filter to determine model parameters.
   - Mapped inferrred latent acne system states to probabilities of acne severity state being realized with softmax of score. 
   - Used latent state-severity state mapping to maximize non-linear model parameters with gradient ascent. 
   - Maximized linear parameters with Moore-Penrose psuedoinversion of associated matrices.  



## Key Takeaways

- **Three Severity States:** The KDE of normalized acne severity is bimodal, justifying a 3 category classification (low severity change, medium severity change, high severity change) (now validated statistically).
- **Tightly Clustered Distributions:** Severity tends to cluster tightly around the two peaks (~60% and ~100% below baseline).
- **Diminishing Returns:** Consecutive treatments provide reduced improvements over within certain consecutive treatment stretches, pointing toward a feedback-like mechanism govering acne severity based on bacterial load, inflammatory activity, and sebum concentration.
- **Steady-State Hypothesis:** Eventually, acne severity may converge toward a steady-state distribution determined by non-treatment biological factors like hormones, skin type, or diet.

## Applications

- **Clinical Utility:**
  - Helps clinicians identify when further treatment will have minimal benefit.
  - Helps clinicians determine when other interventions (like hormonal therapies, antibiotic switches, etc) are justified. 
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
- **Omics Pipeline Integration:**
  - Integrate omics pipelines to pass biochemical signatures of imflammation into SSM model, refining that component.  

- **Demographic Analysis:**
  - Investigate how treatment responses differ across patient ages, backgrounds, and area of residence.

- **LLM Integration:**
  - Incorporate ChatGPT into the pipeline for automating annotations, planning, and insight generation for patients and clinicians separately.



**Author:** Nathaniel Wolff
**Contact:** [nathanielwolff1818@gmail.com; https://github.com/Nathaniel-Wolff]  
**Status:** 3rd draft complete, polishing in progress
**Date Updated:** 11-26-2025  


