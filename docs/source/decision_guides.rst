.. _decision_guides:

Decision Guides
===============

Quick flowcharts for common choices. Each guide starts with your situation and
leads to a concrete recommendation.

.. contents:: Table of Contents
   :depth: 1
   :local:


Which Clustering Settings Should I Use?
---------------------------------------

.. mermaid::

   flowchart TD
     Q[Your situation?] --> A[Small test search]
     Q --> B[Large production]
     Q --> C[Unknown source type]
     Q --> D[Targeted / follow-up]
     Q --> E[Glitch rejection]
     Q --> F[Short-duration signals]

     A --> A1[Use defaults<br/>TFgap=6, Tgap=3, Fgap=130]
     B --> B1[Same defaults<br/>Tune only if bkg too high]
     C --> C1[Lower subnet to 0.5–0.6<br/>Increase TFgap to 8–10]
     D --> D1[Apply sky mask<br/>Restricts false clusters]
     E --> E1[Increase subnet to 0.8+<br/>Decrease Tgap / Fgap]
     F --> F1[Decrease TFgap<br/>Avoid merging distinct signals]

See :ref:`clustering_algorithm` for full details.


Which Recipe Should I Follow?
-----------------------------

.. mermaid::

   flowchart TD
     Q[What do you need?] --> A[First time]
     Q --> B[Real data search]
     Q --> C[GRB follow-up]
     Q --> D[Sensitivity study]
     Q --> E[Production bkg]
     Q --> F[Ranking model]
     Q --> G[hrss50 / hrss90]
     Q --> H[Debug failure]

     A --> A1[Start Here → Learning Path]
     B --> B1[All-Sky Burst Search]
     C --> C1[Targeted External-Trigger]
     D --> D1[Injection Campaign]
     E --> E1[Background-Only Production]
     F --> F1[Training XGBoost Ranking]
     G --> G1[Efficiency Study]
     H --> H1[Debugging Failed Production]

See :ref:`analysis_recipes` for all recipes.


Which Background Split Strategy?
--------------------------------

.. mermaid::

   flowchart TD
     Q[How many chunks?] --> A[Quick test]
     Q --> B[Single chunk]
     Q --> C[Multi-chunk]
     Q --> D[Blind analysis]

     A --> A1[fraction<br/>Random split, fast]
     B --> B1[job<br/>Only if jobs don't share time]
     C --> C1[interval_livetime ✓<br/>Guarantees train/FAR independence]
     D --> D1[interval_livetime<br/>+ fake_openbox]

See :ref:`postproduction_background` for full details.


Which Likelihood Parameters Should I Change?
--------------------------------------------

.. mermaid::

   flowchart TD
     Q[Your search type?] --> A[Standard burst]
     Q --> B[2-detector network]
     Q --> C[XGBoost ranking]
     Q --> D[CBC / BBH]
     Q --> E[Large clusters]
     Q --> F[Poor localization]

     A --> A1[netRHO, netCC, healpix ONLY]
     B --> B1[+ delta = 0.5<br/>Tune if sky looks biased]
     C --> C1[xgb_rho_mode: true<br/>Uses ρ₀ instead of ρ]
     D --> D1[Search: BBH or IMBHB<br/>Enables chirp mass]
     E --> E1[Adjust precision<br/>Big-cluster optimization]
     F --> F1[Increase healpix<br/>7 → 8 = 4× more pixels]

.. warning::

   Do not tune ``delta``, ``cfg_gamma``, or ``precision`` without
   understanding the impact. Incorrect values can suppress real signals
   or inflate the background.

See :ref:`likelihood_guide` for full details.


Which Training Strategy?
------------------------

.. mermaid::

   flowchart TD
     Q[Your setup?] --> A[Single run]
     Q --> B[Multiple chunks]
     Q --> C[Limited sims]
     Q --> D[New waveform]
     Q --> E[Stability test]

     A --> A1[Train on same-run<br/>10% split]
     B --> B1[Train on K chunks<br/>FAR on target chunk]
     C --> C1[scale_pos_weight: auto<br/>Balances BKG/SIM ratio]
     D --> D1[Separate model per family<br/>or family as feature]
     E --> E1[Train on K-1, test on K<br/>Cross-validation]

See :ref:`postproduction_xgboost` and :ref:`postproduction_trainingset` for full details.
