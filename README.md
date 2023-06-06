# Aspect-Driven Structuring of Historical Dutch Newspaper Archives
This paper belongs to our TPDL2023 submission. 
Thank you for being here.
However, due to legal restrictions we cannot provide news articles or produced intermediate data.


## Setup
We used a virtual Python environment.
### Conda
Setup Python 3.8 environment
```
conda create -c news -python=3.8
```
Activate environment
```
conda activate news
```
### Requirements
Install the required requirements via pip:
```
pip install -r requirements.txt
```
### Run setup script
```
python src/nnsummary/setup.py
```


### Python Path
In order to make our scripts find the correct path, you must set your Python path:
```
export PYTHONPATH="/home/USER/AspectDrivenNewsStructuring/src/:"
```
If you use an IDE, you can simply mark the src directory as "sources root".



# Code Documentation
Our code is located in the src directory. 
Our pipeline comes with different components:
1. [Wikipedia Processing](src/nnsummary/wikipedia) (all scripts to process Wikipedia categories and XML Dumps)
2. [Wikipedia Section Title Clustering](src/nnsummary/wikipedia/clustering.py)
3. [Wikipedia Aspect Mining](src/nnsummary/wikipedia/aspect_mining.py)
4. [Training + Applying Text Classification Models](src/nnsummary/classification)
5. [Translation from Dutch to English](src/nnsummary/translation)
6. [Multi-Document Summarization](src/nnsummary/summary)
7. [News Article Parsing and Snippet Generation](src/nnsummary/news)
8. [Component-wise Evaluation Scripts](src/nnsummary/evaluation)


There is a dedicated [config](src/nnsummary/config.py) with all parameters and paths. 
If you want to run our scripts, please set the paths accordingly to your structure. 

# Resources
We provided our [crawled Wikipedia category file](resources/categories.json) as an additional resource.
In order to reproduce our findings, you need to crawl the Dutch Wikipedia Dumps by yourself. 
However, due to legal restrictions we cannot provide news articles or produced intermediate data.