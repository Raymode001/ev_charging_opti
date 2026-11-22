# Related work

Summaries come from each paper's abstract, record page or search snippets, **not from a full read**, unless stated. Status tells you what was checked (2026-11-22) and what to confirm before citing in the report.

## A. The dataset and the system it comes from

**[1] Z. J. Lee, T. Li, S. H. Low. "ACN-Data: Analysis and Applications of an Open EV Charging Dataset." *Proc. 10th ACM e-Energy*, 2019, pp. 139–149. doi:10.1145/3307772.3328313.**
- What it is:
  - Introduces ACN-Data, workplace charging sessions from Caltech, JPL and an office site, with user IDs, arrival and departure, requested and delivered energy, and fine-grained charging data.
  - Over 30,000 sessions at the time of writing.
  - The dataset site asks users to cite this paper.
- Relevance: **our data source.** Cite it for the dataset.
- Status: venue, pages and DOI confirmed via the [Caltech dataset page](https://ev.caltech.edu/dataset) and [researchr](https://researchr.org/publication/LeeLL19-2). PDF: [ev.caltech.edu](https://ev.caltech.edu/assets/pub/ACN_Data_Analysis_and_Applications.pdf).

**[2] Z. J. Lee, D. Chang, C. Jin, G. S. Lee, R. Lee, T. Lee, S. H. Low. "Large-Scale Adaptive Electric Vehicle Charging." *IEEE SmartGridComm*, 2018.**
- What it is:
  - Argues that large garages should *oversubscribe* electrical infrastructure (more chargers than the transformer can feed at full power) and control charging to stay within limits.
  - Proposes an online scheduling framework based on **model predictive control and convex optimization**, with practical constraints such as unbalanced three-phase wiring and non-ideal battery behavior.
- Relevance:
  - The motivation for peak minimization.
  - The closest prior version of our rolling-horizon controller (`acn/online.py`).
- Status: authors and venue confirmed via [CaltechAUTHORS](https://authors.library.caltech.edu/records/yvb0d-sqe61). [PDF (NSF PAR)](https://par.nsf.gov/servlets/purl/10129549).

**[3] Z. J. Lee et al. (incl. S. H. Low). "Adaptive Charging Networks: A Framework for Smart Electric Vehicle Charging." *IEEE Trans. Smart Grid* 12(5):4339–4350, 2021. arXiv:2012.02636.**
- What it is:
  - Journal-length description of the ACN: deployed at Caltech in early 2016 and at over 100 US sites.
  - Covers unbalanced three-phase infrastructure and quantized control signals.
  - Its scheduler raised operator profit about 3.4× vs uncontrolled charging in Caltech-workload simulations.
- Relevance:
  - Cite for the 32 A / 208 V EVSE model, the discrete pilot signals (our MIP extension) and the transformer limits.
  - The simulator in `acnportal` implements this framework.
- Status: venue confirmed via [researchr](https://researchr.org/publication/LeeLLJLLCOL21) and [IEEE Xplore](https://ieeexplore.ieee.org/document/9409126/metrics). The full author list has 9 names; copy it from the IEEE record. [arXiv](https://arxiv.org/abs/2012.02636v1).

## B. Online scheduling and departure uncertainty

**[4] "Optimal Online Adaptive Electric Vehicle Charging." 2017 (Caltech Netlab; L. Guo, K. F. Erliksson, S. H. Low, per a 2016 Netlab presentation).**
- What it is:
  - An **online linear program** that schedules only the cars currently present, without forecasts of future arrivals.
  - Proves it matches the offline optimum when cost coefficients are uniformly monotone; otherwise the competitive ratio is bounded by the variability of the cost coefficients.
  - Tested on Google and Caltech ACN data.
- Relevance:
  - Theory behind comparing our offline LP with the online controller.
  - Our "true departures" online policy is this setting: arrivals unknown, departures known.
- Status: **title and abstract confirmed** ([CaltechAUTHORS](https://authors.library.caltech.edu/records/15bpb-0ct28)). **Authors and venue (believed IEEE PES General Meeting 2017) NOT confirmed.** Check IEEE Xplore before citing.

**[5] N. Tucker, G. Cezar, M. Alizadeh. "Real-Time Electric Vehicle Smart Charging at Workplaces: A Real-World Case Study." arXiv:2203.06847, 2022.**
- What it is:
  - Real-time workplace charging that minimizes time-of-use energy cost plus **monthly demand charges**.
  - Because departure times are unknown, it generates several candidate departure times per EV as scenarios and drops those that become infeasible.
  - Evaluated on two weeks at a Google parking lot.
- Relevance: the same problem as our departure-uncertainty extension, handled with scenarios instead of our point estimates plus minimum progress.
- Status: authors and date confirmed via [arXiv](https://arxiv.org/abs/2203.06847v1).

**[6] A. Puech, T. Rigaut, W. Templier, M. Tournoud. "Controlling Large Electric Vehicle Charging Stations via User Behavior Modeling and Stochastic Programming." IEEE ISGT Europe 2024. arXiv:2402.13224.**
- What it is:
  - Models uncertain start and end times and energy demand.
  - A **two-stage** variant uses charging-session scenarios instead of a single forecast and is more robust to early disconnections.
- Relevance: the closest match to combining our two-stage model with departure uncertainty. Supports our finding that hedging against early departures matters.
- Status: authors and venue confirmed via [arXiv](https://arxiv.org/abs/2402.13224v4) and Crossref (doi:10.1109/ISGTEUROPE62998.2024.10863074).

## C. Demand charges and stochastic programming

**[7] Z. J. Lee, J. Z. F. Pang, S. H. Low. "Pricing EV Charging Service with Demand Charge." *Electric Power Systems Research* 189:106694, 2020. doi:10.1016/j.epsr.2020.106694.**
- What it is:
  - Allocates a site's costs (time-of-use energy, **demand charge**, congestion) to individual sessions.
  - Prices are computed offline at the end of the billing period; an MPC scheduler sets rates in real time.
  - Simulated on Caltech and JPL data.
- Relevance:
  - Shows why the monthly peak, not the daily peak, is what's billed.
  - Basis for the prices / demand-charge extension.
- Status: confirmed via [CaltechAUTHORS](https://authors.library.caltech.edu/records/8hkmq-h5806). Disclosure in the record: Low co-founded PowerFlex.

**[8] L. Chen, L. He, Y. (H.) Zhou. "An Exponential Cone Programming Approach for Managing Electric Vehicle Charging." *Operations Research* 72(5):2215–2240, 2024. doi:10.1287/opre.2023.2460.**
- What it is:
  - A charging provider with random arrivals and **demand charges** minimizes expected total cost, formulated as a stochastic program.
  - Solved approximately via exponential cone programming, with robust-optimization ideas.
- Relevance:
  - The OR-methods anchor for our two-stage model (`acn/stochastic.py`).
  - A good contrast: we solve exactly by exploiting a single first-stage variable; they need approximation for richer first-stage decisions.
- Status: title, authors and venue confirmed via [RePEc](https://econpapers.repec.org/RePEc:inm:oropre:v:72:y:2024:i:5:p:2215-2240). Whether it is strictly two-stage was not confirmed from snippets.

## How our work relates (for the report's related-work paragraph)
- **Offline peak LP:** standard as a benchmark ([2], [4]). Ours adds per-car rates estimated from data and reports the gap to two simple causal baselines.
- **Online control:** our rolling-horizon controller follows the MPC idea of [2]/[3] with a tie-break toward earlier charging. The new piece is measuring how *driver-stated* departures (first estimate, as known at plug-in) degrade it, and a minimum-progress hedge ([5] and [6] use scenarios instead).
- **Two-stage cap:** a stage-1 demand-charge commitment with daily recourse, related to [7]/[8]. Ours is solved exactly over 260 historical days because the first stage is a single number.

**Before submitting:** confirm [4]'s venue and authors, and pull [3]'s full author list from IEEE Xplore.
