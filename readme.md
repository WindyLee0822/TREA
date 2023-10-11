

# TREA
Source code of “TREA: Tree-structure Reasoning Schema for Conversational Recommendation (ACL 2023)”

If you encounter problems, feel free to contact me (wendili@hust.edu.cn). I will reply to you as soon as possible.


## Run
To run the recommendation part.
`python run_publish.py -is_finetune mov`

To run the generation part. (The generation part of tgredial is under organization, we will publish this part as soon as possible)
`python run_publish.py -is_finetune gen`


## Dataset

You need to download dataset from [https://drive.google.com/file/d/1cFSSlMUBmyWRpoqqAtqYKbrQ6fVMqApY/view?usp=drive_link](url) and unzip it to a directory named "tgredial"


## Citation

If you find this repo helpful, please cite the following:

```bibtex
@inproceedings{li-etal-2023-trea,
    title = "{TREA}: Tree-Structure Reasoning Schema for Conversational Recommendation",
    author = "Li, Wendi  and
      Wei, Wei  and
      Qu, Xiaoye  and
      Mao, Xian-Ling  and
      Yuan, Ye  and
      Xie, Wenfeng  and
      Chen, Dangyang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.167",
    doi = "10.18653/v1/2023.acl-long.167",
    pages = "2970--2982",
}
```






