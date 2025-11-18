# DSC_211_KarateClub
DSC212 – Spectral Modularity Community Detection on the Karate Club Graph
Research Assignment — Modularity & Community Detection

Course: DSC212: Graph Theory
Dataset: Zachary’s Karate Club Network
Method: Spectral Modularity + Recursive Bisection
Submission Format: Jupyter Notebook (.ipynb) + repository documentation

This project implements spectral modularity maximization to detect communities in the classic Zachary’s Karate Club graph.
The goal is to reproduce, using only network structure, the natural community split that occurred in the real karate club.

Steps:
  Construct the modularity matrix

  Perform spectral bipartition using the leading eigenvector

  Apply recursive bisection to detect multiple communities

  Visualize the graph after every split

  Compute and track centrality + cohesion metrics across iterations

  Discuss how community structure influences node importance

The final deliverable is a Jupyter Notebook that runs top-to-bottom without edits, including visualizations, plots, and analysis.
