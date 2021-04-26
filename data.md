# Dataset Description
We are using [the Offensive Language Identification Dataset (OLID)](https://sites.google.com/site/offensevalsharedtask/olid), which contains a collection of 14,200 annotated English tweets using an annotation model. 

## Files
We extracted columns from the provided training and testing dataset. Then, we splited the extracted training set into train dataset and development dataset by 9:1. The number of entries in each file is shown below:
- `train.csv`: 11916 entries
- `dev.csv`: 1324 entries
- `test.csv`: 860 entries

## Columns
- `tweet`: Original tweets
- `class`: Label of a tweet. There are 2 categories: 
	- 1 (NOT) Not Offensive - This post does not contain offense or profanity.
	- 0 (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense.

## An Example of the data

| tweet | class 
| ------------- | ------------- 
|@USER She should ask a few native Americans what their take on this is. | 0
|@USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 👊🇺🇸👊 URL|	0
|Amazon is investigating Chinese employees who are selling internal data to third-party sellers looking for an edge in the competitive marketplace. URL #Amazon #MAGA #KAG #CHINA #TCOT| 1

## Dataset Reference
```
@inproceedings{zampierietal2019, 
title={{Predicting the Type and Target of Offensive Posts in Social Media}}, 
author={Zampieri, Marcos and Malmasi, Shervin and Nakov, Preslav and Rosenthal, Sara and Farra, Noura and Kumar, Ritesh}, 
booktitle={Proceedings of NAACL}, 
year={2019}, 
} 
```
