.. _recipe_training:

Training XGBoost Ranking
========================

**Use this route for:** fitting a ranking model to your completed BKG/SIM study.
Supply background training catalogs, matched simulation training catalogs and
feature/model settings. Reserve separate background for FAR evaluation and
independent simulations for sensitivity evaluation.

1. Prepare and split the samples with :doc:`postproduction_trainingset`.
2. Follow :doc:`postproduction_study` to adapt and run the maintained study template.
3. Use :doc:`postproduction_xgboost` for feature/ranking interpretation and
   :doc:`postproduction_actions` for exact action arguments.

**Completion check:** keep the model, feature settings and training record;
verify interval separation and inspect ranking behavior on held-out samples.
Sample size and feature stability must be assessed for your population.

**Worked precursor:** :doc:`tutorial_custom_postproduction` teaches the workflow
and report mechanics using prepared catalogs. Training requires your study data.

See :doc:`analysis_recipes` for the other analysis tasks.
