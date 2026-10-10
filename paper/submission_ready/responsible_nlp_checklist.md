# Responsible NLP answers

Prepared October 10, 2026 for the current manuscript. Labels paraphrase the
[official questions](https://aclrollingreview.org/responsibleNLPresearch/); enter the
answers in the live form after author review. Appendix letters follow the current PDF.

| Item | Answer and evidence |
| --- | --- |
| A1 Limitations | Yes. Limitations follows Section 6; covers task scope, competence, probe interpretation and patch specificity. |
| A2 Potential risks | No separate societal-risk discussion. This is a diagnostic study of synthetic arithmetic, with no personal records or deployed system. Limitations bounds generalization; compute use is reported in Appendix A. |
| B Artifacts | Yes. Generated problems, pretrained models, probes and scientific software. |
| B1 Attribution | Yes. Kudo et al. in Sections 1–2 and Appendix A; model-family and software citations in Appendix A. |
| B2 Terms | Yes for the models used. Appendix A names Llama Community Licenses, Gemma Terms, Apache 2.0 and NVIDIA Open Model License. We reimplement the task and do not redistribute model weights or upstream task code. A distribution license for our research code/data has not been assigned; no such release is included in this submission package. |
| B3 Intended use | Yes, with scope. Appendix A states research inference; Limitations restricts generalization. Generated tasks are intended for controlled model analysis. |
| B4 Identifying content | Not applicable to personal-data processing. Problems contain generated digits, operators and a fixed English word pool; no human records are collected. Appendix A describes the construction. |
| B5 Documentation | Yes. Section 2 and Appendix A document English arithmetic, naming conditions, task levels and tokenizer checks. |
| B6 Data sizes and splits | Yes. Appendix A reports matched-set and probe-training sizes; Appendix F documents computation-disjoint generated-chain training/validation; Appendix H gives independent calibration and control-cohort counts. |
| C Experiments | Yes. |
| C1 Compute | Yes. Model sizes are stated in names; Appendix A reports hardware and the shared allocation upper bound for both studies, excluding separate adapter training. |
| C2 Setup and selection | Yes. Section 2 and Appendices A, C, F and H report probe recipes, layer selection, patch scope and calibration. Appendix H reports failed matching rather than selecting a favorable held-out format. |
| C3 Statistics | Yes. Effects, equivalence bounds, demonstration repeats and cluster intervals are explained in Section 2 and Appendices B, F–H. Appendix G distinguishes ten consistent paired reductions from 43 small-effect cells. Exploratory control intervals are pointwise; zero-event intervals are not population bounds. |
| C4 Software | Yes. Appendix A gives package citations and installed versions, including PyTorch 2.11.0/CUDA 12.9 and Transformers 5.14.1. |
| D Human participants | No. D1–D5 are not applicable; cited human studies collected no new participant data for this paper. |
| E AI assistants | Yes. |
| E1 Disclosure | Claude/Claude Code and OpenAI Codex assisted with experiment and analysis code, job orchestration, and manuscript drafting/editing. Results come from executable scripts; automated checks compare calculations, paired cohorts, raw outputs, summaries and manuscript quantities. The human authors retain responsibility for the final content and must review this disclosure before submission. |
