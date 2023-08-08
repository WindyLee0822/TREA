# TREA
Source code of “TREA: Tree-structure Reasoning Schema for Conversational Recommendation (ACL 2023)”
If you encounter problems, feel free to contact me (wendili@hust.edu.cn). I will reply to you as soon as possible.


## Run
To run the recommendation part.
`python run_publish.py -is_finetune mov`

To run the generation part.
`python run_publish.py -is_finetune gen`


## Dataset
We publish the preprocessed dataset  `train_publish.json` and  `test_publish.json`

if you wanna use raw datasets from previous works  `train.json` and `test.json`, you just need to set `process_raw_data=True` during the dataset initialization.






