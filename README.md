# ReGrid-
ReGrid is a Python toolkit for the parametric modelling, structural analysis and reuse-based stock-constrained structural optimisation of gridshell structures. The project uses OpenSeesPy to perform elastic static analysis, linear buckling analysis, nonlinear load-path analysis and mechanical optimisation. 


<img width="800" height="400" alt="Screenshot 2026-11-18 195154" src="https://github.com/user-attachments/assets/b9807cc1-483b-41dc-96d9-257850b67047" />

Features
Parametric geometry: Generate barrel vaults, parabolic domes, hypar surfaces and Kiewitt-8 domes by specifying dimensions, rise and mesh subdivisions.
Mesh generation: Create quadrangular or triangular grids and identify structural nodes, members and boundary sets.
Structural analysis: Use OpenSeesPy to run elastic static analyses, linear buckling analyses and nonlinear load-path analyses. Nonlinear runs support force, displacement and arc-length control.
Design checks and section sizing: Assign steel sections to members, evaluate structural demand and perform Eurocode 3 checks. Mechanicl optimisation (section-sizing) routines can search a section catalogue against selected design criteria (maximum demand-to-capacity-ratio, maximum displacement, linear buckling factor)
Optimization: Use genetic algorithms to explore geometry and section-design alternatives. Multi-objective runs can compare criteria such as mass, cost, environmental impact and reuse-related measures, with Pareto-front tools for examining trade-offs.
Reuse and corrosion: Optionally match structural members to reclaimed steel stock, account for cutting and leftover material, and model corrosion-related section loss.
Results and exports: Produce plots, reports and CSV summaries, export geometry to DXF, and save optimization results for later review.
Workflow
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/128ee709-e52b-4a18-bd08-96c02acd4639" />



The analysis workflow starts with user-defined geometry and mesh parameters. The generated structure is then assigned material and section properties, boundary conditions and loads. Depending on the chosen settings, the project runs structural analyses and design checks, or evaluates candidate designs through optimization. Optional reuse and corrosion workflows can be included in this process.
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/5f85896d-4669-420b-960c-e93e90e8016a" />


The code is organized into modules for geometry, meshing, analysis, optimization, visualization and data export. Configuration options and batch tools support running different cases without manually rebuilding each model.

The project is intended as a configurable research environment: available analyses and optional workflows depend on the selected settings and input data.



If you use this code in your research, please cite the following papers
1. Ascione, F., Esposito, F., Faiella, D., Mele, E. (2026). Reuse Based Design of Steel Gridshells Through Multi-Objective Optimization. In: Zanon, R., Nastri, E., Simoncelli, M., Paolacci, F. (eds) Proceedings of the XXX Congress CTA 2026. CTA 2026. Lecture Notes in Civil Engineering, vol 942. Springer, Cham. https://doi.org/10.1007/978-3-032-39505-4_10
2. Ascione, F., Esposito, F., Faiella, F., Mele E., Steel reuse for sustainable design of large-span structures: The case of gridshells, Journal of Constructional Steel Research, Volume 234,
2025, 109743, ISSN 0143-974X, https://doi.org/10.1016/j.jcsr.2025.109743.
