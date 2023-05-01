Below is a list of models available in this directory and their significance:

- `CellLineFewShot_Layers2_Hidden64_DO0-1_AFsigmoid_LR0-001_DR0-99_DS1000`: This model achieves the best performance (precision@k=5) when concatenated with both raw drug fingerprints and embedded drug features and used as input to a logistic regression model. This model is used in future iterations: fusion+logistic, concat+dnn, and fusion+dnn.


- `CellLineFewShot_Layers1_Hidden64_DO0-1_AFsigmoid_LR0-0001_DR0-99_DS1000`: This model achieves the best performance with respect to its training objective. That is, descriminating between cancer types.
