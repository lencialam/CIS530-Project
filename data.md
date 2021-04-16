# Dataset Description
The dataset we are using for this project is collected and compiled by Tom Davidson, which contains 24k tweets that are labeled as hate speech, offensive language, or neither.  The original dataset can be accessed from [here](https://data.world/thomasrdavidson/hate-speech-and-offensive-language).

## Files
We splited the original dataset into train dataset, development dataset, and test dataset:
- `train.csv`: 0.8
- `dev.csv`: 0.1
- `test.csv`: 0.1

## Columns
- `count`: number of CrowdFlower users who coded each tweet. 
- `hate_speech`: number of CF users who judged the tweet to be hate speech.
- `offensive_language`: umber of CF users who judged the tweet to be offensive.
- `neither`: number of CF users who judged the tweet to be neither offensive nor non-offensive.
- `class`: class label for majority of CF users.
- `tweet`: Original tweets

## Dataset Reference
```
@inproceedings{hateoffensive, 
    title={Automated Hate Speech Detection and the Problem of Offensive Language}, 
    author={Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
    booktitle={Proceedings of the 11th International AAAI Conference on Weblogs and Social Media}, 
    series={ICWSM '17}, 
    year={2017}, 
    location = {Montreal, Canada} 
}
```
