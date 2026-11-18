# ReGrid-
ReGrid is a Python toolkit for the parametric modelling, structural analysis and reuse-based stock-constrained structural optimisation of gridshell structures. The project uses OpenSeesPy to perform elastic static analysis, linear buckling analysis, nonlinear load-path analysis and mechanical optimisation. 

Features
Parametric geometry: Generate barrel vaults, parabolic domes, hypar surfaces and Kiewitt-8 domes by specifying dimensions, rise and mesh subdivisions.
Mesh generation: Create quadrangular or triangular grids and identify structural nodes, members and boundary sets.
Structural analysis: Use OpenSeesPy to run elastic static analyses, linear buckling analyses and nonlinear load-path analyses. Nonlinear runs support force, displacement and arc-length control.
Design checks and section sizing: Assign steel sections to members, evaluate structural demand and perform Eurocode 3 checks. Mechanicl optimisation (section-sizing) routines can search a section catalogue against selected design criteria (maximum demand-to-capacity-ratio, maximum displacement, linear buckling factor)
Optimization: Use genetic algorithms to explore geometry and section-design alternatives. Multi-objective runs can compare criteria such as mass, cost, environmental impact and reuse-related measures, with Pareto-front tools for examining trade-offs.
Reuse and corrosion: Optionally match structural members to reclaimed steel stock, account for cutting and leftover material, and model corrosion-related section loss.
Results and exports: Produce plots, reports and CSV summaries, export geometry to DXF, and save optimization results for later review.
Workflow

The analysis workflow starts with user-defined geometry and mesh parameters. The generated structure is then assigned material and section properties, boundary conditions and loads. Depending on the chosen settings, the project runs structural analyses and design checks, or evaluates candidate designs through optimization. Optional reuse and corrosion workflows can be included in this process.

The code is organized into modules for geometry, meshing, analysis, optimization, visualization and data export. Configuration options and batch tools support running different cases without manually rebuilding each model.

The project is intended as a configurable research environment: available analyses and optional workflows depend on the selected settings and input data.
