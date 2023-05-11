Below is a list of models available in this directory and their significance:

- `Fused-FewShotCDR_NL64_16_DO0-1_AFsigmoid_LR0-01_DR0-99_DS500`: This model achieves the best performance using the best fusion model that fuses the RNA and Drug encoders that perform the best at descriminating by cancer type and MOA, respectively. See their respective readme files.

- `Fused-FewShotCDRRawDrugRawCell_NL64_32_16_DO0-3_AFrelu_LR0-01_DR0-99_DS500`: This model achieves the best performance using the best fusion model of raw RNA and Drug features

- `Fused-FewShotCDREmbedDrugRawCell_NL32_16_8_DO0-1_AFrelu_LR0-001_DR0-99_DS500`: This model achieves the best performance using the best fusion model of raw RNA and embeded Drug features

- `Fused-FewShotCDRRawDrugEmbedCell_NL64_32_DO0-1_AFrelu_LR0-01_DR0-99_DS50`: This model achieves the best performance using the best fusion model of embedded RNA and raw Drug features

- `Fused-FewShotCDREmbedDrugEmbedCell_NL64_32_DO0-3_AFrelu_LR0-001_DR0-99_DS50`: This model achieves the best performance using the best fusion model of embedded RNA and Drug features

- `Unfused-FewShotCDRRawDrugRawCell_NL64_16_DO0-1_AFrelu_LR0-001_DR0-99_DS50`: This model achieves the best performance using the concat of raw RNA and Drug features

- `Unfused-FewShotCDREmbedDrugRawCell_NL64_32_16_DO0-1_AFrelu_LR0-01_DR0-99_DS50`: This model achieves the best performance using the concat of raw RNA and embedded Drug features

- `Unfused-FewShotCDRRawDrugEmbedCell_NL64_64_DO0-1_AFrelu_LR0-01_DR0-99_DS500`: This model achieves the best performance using the concat of embedded RNA and raw Drug features

- `Unfused-FewShotCDREmbedDrugEmbedCell_NL64_16_DO0-3_AFrelu_LR0-001_DR0-99_DS500`: This model achieves the best performance using the concat of embedded RNA and Drug features
