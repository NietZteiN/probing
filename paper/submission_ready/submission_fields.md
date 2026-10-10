# Submission fields

Title: Chain of Thought Reduces Errors from Misleading Names in Arithmetic

Type: Short paper

Suggested area: Interpretability and Analysis of Models for NLP

TL;DR: Writing arithmetic calculations makes answers less sensitive to misleading variable names.

Abstract:

Asking language models to write out arithmetic calculations reduces answers that follow a misleading variable name. Building on Kudo et al.'s arithmetic study, we test thirteen models on pairs of problems with the same equations. In one version, a variable has an ordinary name; in the other, its name is a word for an incorrect digit. When asked only for the final answer, models more often return the digit suggested by the misleading name. Chain-of-thought prompting asks them to write calculations before answering and reduces this tendency. The reduction is reliable in 10 comparisons; with calculations, 43 of 44 added-error intervals lie within 2 percentage points of zero. Writing the calculation therefore makes answers more robust to misleading wording while the underlying math remains unchanged.

Authors, submission history, preferred venue, preprint declaration and service contributor require author completion in OpenReview.
