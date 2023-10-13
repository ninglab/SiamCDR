Below is a list of models available in this directory and their significance:

`DrugFewShot_Layers1_Hidden64_DO0-1_AFrelu_LR0-001_DR0-99_DS1000`: This model achieves the best performance with respect to its training objective. That is, descriminating between mechanisms of action (MOA). This model also achieves the best performance (precision@k=5) of all drug encoders when concatenated with the best rna encoder (`CellLineFewShot_Layers2_Hidden64_DO0-1_AFsigmoid_LR0-001_DR0-99_DS1000`) and used as input to a logistic regression classifier.

