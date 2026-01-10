# acneBayesModel
**A probabilistic modeling framework for evaluating longitudinal treatment effects and intervention efficacy.**

This project demonstrates how a probabilistic modeling framework can be used to evaluate clinical intervention strategies in noisy, longitudinal biological data. 

In this system, outcome severity is modeled as a discrete state, enabling inference about how repeated treatments and treatment switches affect future responses. 

The framework produces treatment-history-aware probabilistic predictions by using kernel-density-based discretization in conjunction with Bayesian updating of Dirichlet posteriors. 

The framework tracks the evolution of sequential posteriors using cumulative Kullback-Leibler divergence, and fits piecewise linear models to identify treatment regimes with non-diminishing returns. 

This allows evaluation of the efficacy of different treatment strategies through an information-theoretic lens, instead of relying on point estimates alone. 


![Cumulative KL divergence over sequential treatments](AcneProjectPic.png)
*Cumulative KL divergence over sequential treatments. Piecewise spline fit highlights regions of diminishing returns, demonstrating how treatment history informs probabilistic prediction of future outcomes.*



## Approach

1. **Severity Score Normalization, KDE Binning, & Bootstrapping:**
   - Normalized longitudinal severity measurements and discretized them into 3 interpretable states using probability-density based binning.
   - Verified binning robustness with bootstrapping. 

    *This step ensured that subsequent probabilistic modeling is based on identifiable state representations.* 

2. **Dirichlet Posterior Modeling:**
   - Computed the posterior distribution over severity states, conditioned on treatment history. 
   - Quantified the information gained from each treatment update via cumulative KL divergence across all treatment updates.
   
   *This step enabled history-aware, probabilistic prediction of future outcomes and comparison of treatment strategies over time.* 

3. **Cumulative KL Divergence Curve Fitting, Piecewise Linear Regression:**
   - Modeled evolution of cumulative KL divergence to detect sustained or diminishing treatment effects via piecewise linear regression.
  
   *This step highlighted which treatment regimes resulted in consistent improvement, revealing insights into intervention efficacy and mechanistic drivers of diminishing returns.* 

4. **Fitting of State-Space/Statistical Mechanistic Model, Optimization of Parameters with Expectation-Maximization/Generalized EM Algorithm**
   - Developed a latent-state mechanistic model of outcome severity, incorporating bacterial load, inflammation, and sebum production.
   - Optimized parameters with an Expectation-Maximization / Generalized EM (EM/GEM) approach.
   
   *This model enabled direct testing of potential treatment regimes for outcome efficacy and cost-optimization.
   Model equations and EM/GEM implementation can be found in the corresponding Jupyter Notebook.*

## Applications

1. **Clinical Decision Support:**
  - Identifies when further treatment is likely to have minimal benefit.
  - Aids in clinician decision-making, highlighting when alternative interventions (like hormonal therapies) may be justified. 
  - Supports cost-effectiveness assessments for clinicians, patients, and insurers.

2. **Scalable Data Insights & Automation:**
   - Enables dataset annotation and patient report generation.
   - Can be used to summarize trends across patient cohorts to guide treatment strategy.
   - Integrates with LLM tools for scalable data analysis and reporting. 

## Next Steps

1. **Simulation Mode:**
  - Build synthetic patient models to simulate long-term treatment strategies.
  - Create a patient-facing interface to explore probabilities of improvement over time, supporting adherence and motivation. 

2. **Omics Integration:**
  - Incorporate biochemical signatures of inflammation into the state-space model to refine predictions.  

3. **Demographic Analysis:**
  - Examine treatment responses across age, background, and geographic cohorts.

4. **LLM Integration:**
  - Integrate LLM tools to automate annotations, planning, and report generation for patients and clinicians.



**Author:** Nathaniel Wolff
**Contact:** [nathanielwolff1818@gmail.com; https://github.com/Nathaniel-Wolff]  
**Status:** 6th draft complete, Simulation Mode in progress. 
**Date Updated:** 01-09-2026  


