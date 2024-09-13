# SHAPR

Configurations for training: ```config/sfd_test21.json (dim=0,1,2), config/sfd_test25.json (dim=0,1,2,3), config/cell_sfd_test26.json (dim=0,1,2,3), config/cell_sfd_test27.json (dim=0,1,2)```.

Configurations for evaluation: ```scripts/evaluation/sfd_test21.json, scripts/evaluation/sfd_test25.json, scripts/evaluation/cell_sfd_test26.json, scripts/evaluation/cell_sfd_test27.json```.

Training:
```
python run_train_script.py -p config/sfd_test21.json
```
Evaluation:
```
python -m scripts.evaluation scripts/evaluation/sfd_test21.json
```
