import optuna

# Point to your database and study name
storage_url = r"sqlite:///C:\VideoTools\MTB-Video-Tools\data\db\hpo_study.db"
study_name = "study_12"  # Update with your Study #12 name

study = optuna.load_study(study_name=study_name, storage=storage_url)

# 1. Optimization History (shows the sawtooth climbing over trials)
fig_history = optuna.visualization.plot_optimization_history(study)
fig_history.show()

# 2. Slice Plot (shows how parameters like hidden_dim and c2_threshold impact the score)
fig_slice = optuna.visualization.plot_slice(study)
fig_slice.show()

# 3. Parameter Importances (shows which hyperparameter drove the score the most)
fig_importances = optuna.visualization.plot_param_importances(study)
fig_importances.show()