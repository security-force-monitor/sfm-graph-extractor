# NLP in Human Rights Research - Extracting Knowledge Graphs About Police and Army Units and Their Commanders

This repository hosts the code of an NLP system developed during a research collaboration between Security Force Monitor, a project of the Human Rights Institute at Columbia Law School, and [Dr Daniel Bauer](http://www.cs.columbia.edu/~bauer/) of the Computer Science Department at Columbia University, and Yueen Ma, a post-graduate student at the same.

Our resulting working paper "[NLP in Human Rights Research - Extracting Knowledge Graphs About Police and Army Units and Their Commanders](https://arxiv.org/abs/2201.05230)", published January 2021, discusses the system's purpose, development, outcomes and performance.

## Training data

The training data used to build the model is hosted in the our [nlp_starter_dataset repository](https://github.com/security-force-monitor/nlp_starter_dataset). 

## Knowledge Graph Extraction

We designed a pipeline that can extract a special kind of knowledge graphs where a person's name will be recognized and his/her rank, role, title and organization will be related to him/her. It is not expected to perform perfectly so that all relevant persons will be recognized and all irrelevant persons will be excluded. Rather, it is seen as a first step to reduce the workload that is involved to manually extract such knowledge by combing through a large amount of documents.

This pipeline consists of two major components: Name Entity Recognition and Relation Extraction. Name Entity Recognition uses a BiLSTM-CNNs-CRF model. It recognizes names, ranks, roles, titles and organizations from raw text files. Then the Relation Extraction relates names to his/her corresponding rank, role, title or organization.

Example:
![Example](images/brat_stn.png)

## Dependencies
Tensorflow 2.2.0 <br>
Tensorflow-addons <br>
SpaCy <br>
NumPy <br>
DyNet <br>
Pathlib <br>

## Install
Package: https://pypi.org/project/extract-sfm/
```shell
$ pip install extract_sfm
```


## Usage

### Method 1

Create a python file and write:
```python
import extract_sfm

extract_sfm.extract("/PATH/TO/DIRECTORY/OF/INPUT/FILES")
```
Then run the python file. This may take a while to finish.

### Method 2

Download this Github repository
Under the project root directory, run the python script

```shell
$ python pipeline.py /PATH/TO/DIRECTORY/OF/INPUT/FILES
```
> Note 1: Use absolute path.<br>
> Note 2: Using time_pipeline.py instead of pipeline.py will produce an additional "time.txt" file, which includes how much time each component of the pipeline takes to run.


## Website
1. Copy NER_v2, RE, pipeline.py into the "SERVER/KGE" directory
2. Install npm dependencies under the "SERVER" directory: express, path, multer
```
  $ npm install <package name>
```
3. Run the server by typing in:
```
  $ node server.js
```

![Example](images/website.png)


## Documentation
The documentation for NER and RE is stored in: [doc.txt](doc.txt)
