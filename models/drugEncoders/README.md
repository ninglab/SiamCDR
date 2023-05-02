Below is a list of models available in this directory and their significance:

- `DrugFewShot_Layers2_Hidden64_DO0-1_AFrelu_LR0-01_DR0-99_DS1000`: This model achieves the best performance (precision@k=5) of all drug encoders when concatenated with raw RNA data and used as input to a logistic regression model. This model is used in future iterations along side raw RNA: fusion+logistic, concat+dnn, and fusion+dnn.

- `DrugFewShot_Layers1_Hidden64_DO0-1_AFrelu_LR0-001_DR0-99_DS1000`: This model achieves the best performance with respect to its training objective. That is, descriminating between mechanisms of action (MOA). This model also achieves the best performance (precision@k=5) of all drug encoders when concatenated with the best rna encoder (`CellLineFewShot_Layers2_Hidden64_DO0-1_AFsigmoid_LR0-001_DR0-99_DS1000`) and used as input to a logistic regression model. This model is used in future iterations alongside encoded RNA: fusion+logistic, concat+dnn, and fusion+dnn.

