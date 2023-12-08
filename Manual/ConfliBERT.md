---
title: ConfliBERT Documentation and Usage
author: "Patrick T. Brandt, Vito D'Orazio, Yibo Hu, Latifur Khan, Shreyas Meher, Javier Osorio, Marcus Sianan"
description: What ConfliBERT is and how to use it.
date: November 2023
version:  0.1
output:
  html_document:
    anchor_sections: true
    css: styles.css
    number_sections: true
    toc: true
    toc_float: true
    toc_depth: 2
    collapsed: true
    smooth_scroll: true
bibliography: ConfliBERT.bib  
editor_options: 
  markdown: 
    wrap: 72
---

\> If you are looking for event data - you should not be here. If you
want to make your own event data - you are in the right place. If you
want datasets that are already event-coded using ML. Go to GTD, Uppsala,
etc. Not in the intro specifically.

Whether you are a seasoned data scientist or new to machine learning,
this guide is structured to help you navigate through the varied
functionalities of ConfliBERT with ease. For a hands-on introduction and
demonstration, jump straight to our [Demo usage]. If you're interested
in applying ConfliBERT for **Named Entity Recognition**, see the section
on [NER (Named Entity Recognition) with
ConfliBERT](#ner-named-entity-recognition-with-conflibert). To dive into
**text classification**, visit the section on [Classification with
ConfliBERT]. For **masking and coding tasks**, refer to the section on
[Masking and Coding Tasks with ConfliBERT]. The section on
[Computational Considerations and Benchmarks for ConfliBERT] covers
vital aspects of performance optimization.To explore the various
adaptations of ConfliBERT, see the section on [ConfliBERT Variants]. At
any point, you can refer to the [References / Citations] for further
reading and information sources that have informed the development and
use of ConfliBERT.

# Demo usage

For a hands-on example of ConfliBERT in action, check out the [Google
Colab
Demo](https://colab.research.google.com/drive/1d4557lxoDWKTx0FWcmSPlLx9UEn2BdcA?usp=sharing).

This Google Colab document is a comprehensive guide for setting up and
running experiments with the ConfliBERT language model, which is
specifically tailored for analyzing political conflict and violence
events. Here's a breakdown of its functions:

1.  **Environment Setup**:

    -   The document begins by installing necessary Python packages like
        `torch`, `transformers`, `numpy`, `scikit-learn`, and `pandas`,
        which are essential for machine learning tasks.
    -   It then installs `simpletransformers`, a library that simplifies
        training and using transformer models.

2.  **Cloning the ConfliBERT Source Code**:

    -   The script clones the ConfliBERT repository from GitHub,
        ensuring that the latest version of the code and models are
        used.

3.  **Importing Required Modules and Setting Arguments**:

    -   Necessary Python modules like `pandas` for data manipulation,
        and `argparse` for handling command line arguments, are
        imported.
    -   A namespace for arguments (`args`) is created to manage various
        settings and configurations.

4.  **Selecting a Dataset**:

    -   ConfliBERT allows you to choose from a diverse range of
        processed datasets, each suited for specific analytical tasks.
        Below is a table showcasing the available datasets and their
        corresponding tasks. Depending on your research needs or project
        requirements, you can select the most appropriate dataset:

| Dataset                 | Task       |
|-------------------------|------------|
| 20news                  | binary     |
| BBC_News                | binary     |
| IndiaPoliceEvents_doc   | multilabel |
| IndiaPoliceEvents_sents | multilabel |
| cameo_class             | multiclass |
| cameo_ner               | ner        |
| insightCrime            | multilabel |
| re3d                    | ner        |
| satp_relevant           | multilabel |

-   Each dataset is tailored for different types of classification
    tasks, such as binary classification, multilabel classification,
    multiclass classification, and named entity recognition (NER). This
    versatility allows for a wide range of analyses, from simple binary
    decisions to complex entity extraction and multilabel
    classifications.

For more detailed information about each dataset and to understand how
to align them with your specific needs, please visit the [ConfliBERT
GitHub
repository](https://github.com/eventdata/ConfliBERT/tree/main/configs).

5.  **Training Configuration**:
    -   It sets up training configurations including the task type
        (`multilabel`), the number of seeds for training, initial seed
        value, number of epochs, batch size, and maximum sequence
        length.
    -   Two models are included in this example
        (`ConfliBERT-scr-uncased` and `bert-base-uncased`), with their
        respective paths and configurations.
6.  **Custom Configuration and Data Loading**:
    -   The custom training configuration is saved to the `args` object.
    -   The script then loads training, evaluation, and testing
        datasets, along with the number of labels, and label list for
        Named Entity Recognition (NER) tasks, if applicable.
7.  **Model Training**:
    -   The document executes a loop to run experiments for all the
        models listed in the configuration.
    -   For each model, it sets up an output directory and calls the
        `train_multi_seed` function, passing the training, evaluation,
        and test datasets along with the model configurations.
8.  **Results Compilation**:
    -   Finally, it reads and displays the full report of the training
        results from a CSV file. This file includes detailed performance
        metrics for each model and configuration.

Overall, this Colab document is an all-in-one toolkit for users looking
to leverage the ConfliBERT model for their research or projects related
to political events analysis.

# Introduction to ConfliBERT:

ConfliBERT @hu2022conflibert is a
domain-specific pre-trained language model tailored for the analysis of
political conflict and violence that was developed through a
collaboration between conflict scholars and computer scientists. It was
introduced in a paper titled "ConfliBERT: A Pre-trained Language Model
for Political Conflict and Violence," which was presented at the North
American Chapter of the Association for Computational Linguistics
(NAACL) in 2022. ConfliBERT's design and architecture cater to a wide
range of interactions, from material and verbal conflicts to cooperative
endeavors, encompassing events like protests, insurgencies, civil wars,
and diplomatic disputes.

The study and monitoring of political violence and conflict have long
been vital domains within the political science and policy communities.
The traditional methods used by many event data systems to understanding
these dynamics, such as manual coding and pattern-matching techniques,
though valuable, have limitations in terms of scale and adaptability.
Also, many of these legacy systems are dictionary-based, which makes
them difficult to maintain and update @halterman2023plover. ConfliBERT helps to
overcome these issues with its machine learning approach to data
analysis.

ConfliBERT offers a promising avenue for advancing research in political
science by enabling the efficient and comprehensive analysis of conflict
processes. By bridging the expertise of political science and advanced
natural language processing, it has the potential to redefine
traditional methods, facilitating more nuanced and expansive research
outcomes. It also represents a significant advancement in
domain-specific language models for political conflict and violence,
offering researchers and practitioners a tool that has been pre-trained
on relevant data, thus ensuring better performance on domain-related
tasks.

> Imagine a specific political science usecase that an user can have - why would they use machine-coded event data creation techniques? 

## What does ConfliBERT do and why?:

1.  **Domain-Specific Pre-training:** While many language models are
    trained on general corpora, ConfliBERT's training is rooted in
    domain-specific corpora. This focus allows for enhanced performance
    in performing classification tasks related to political violence,
    conflict, cooperation, and diplomacy.

2.  **Utility for Various Stakeholders:** ConfliBERT is beneficial for
    academics, policymakers, and security analysts. It provides an
    efficient tool for monitoring and understanding the multifaceted
    dynamics of social unrest, political upheavals, and other
    conflict-related events on a global scale.

3.  **Automation and Efficiency:** Traditional conflict analysis methods
    have their limitations in terms of scale and speed. ConfliBERT, with
    its automation capabilities, can parse vast datasets, significantly
    alleviating the challenges of manual annotations.

4.  **Broad Application Potential:** Beyond mere data categorization,
    ConfliBERT can be employed for a diverse range of tasks related to
    conflict research, including event classification, entity
    recognition, relationship extraction, and data augmentation.

## Running Examples

In the following sections, we will walk through illustrative examples of
ConfliBERT's capabilities in action. These examples are designed to help
researchers and analysts better understand how ConfliBERT automates the
coding of textual data into structured conflict and event data. Grasping
these capabilities is essential for a myriad of applications such as
real-time monitoring of political events, in-depth conflict analysis,
and the accumulation of data for international relations and peace
studies.

*Make a Statement* President Biden said that the U.S. alliance with
Japan is stronger than ever.

*Verbal Cooperation* Saudi Arabia expressed intent to restore diplomatic
relations with Iran.

*Material Cooperation* China provided humanitarian aid to Turkey.

*Verbal Conflict* President Zelenskyy accused Russia of war crimes.

*Material Conflict* Israeli forces attacked Hamas in Gaza City.

## Key Features and Components:

1.  **Platform and Requirements:**

    -   ConfliBERT's code requires a Python installation on your
        computer.
    -   Necessary packages include torch, transformers, numpy,
        scikit-learn, pandas, and simpletransformers.
    -   CUDA 10.2 support is included for GPU acceleration.

2.  **Model Versions:**

    -   Four distinct versions of ConfliBERT are available:
        1.  ConfliBERT-scr-uncased: Pre-trained from scratch using an
            uncased vocabulary.
        2.  ConfliBERT-scr-cased: Pre-trained from scratch using a cased
            vocabulary.
        3.  ConfliBERT-cont-uncased: Continual pre-training using the
            original BERT's uncased vocabulary.
        4.  ConfliBERT-cont-cased: Continual pre-training using BERT's
            cased vocabulary.

    These models are available on Huggingface and can be imported
    directly using its API.

3.  **Evaluation and Usage:**

    -   Using ConfliBERT is analogous to other BERT models within the
        Huggingface ecosystem.
    -   Examples are provided for fine-tuning using the Simple
        Transformers library.
    -   A Google Colab demo is available for hands-on evaluation and
        experimentation.

4.  **Evaluation Datasets:**

    -   Several datasets related to news, global events, and political
        conflicts are mentioned for evaluation. These datasets include:
        -   20Newsgroups, BBCnews, EventStatusCorpus, GlobalContention,
            GlobalTerrorismDatabase, Gun Violence Database,
            IndiaPoliceEvents, InsightCrime, MUC-4, re3d, SATP, and
            CAMEO.
    -   Custom datasets can be integrated after preprocessing into the
        required formats.

5.  **Pre-Training Corpus:**

    -   ConfliBERT was pre-trained on an extensive corpus from the
        politics and conflict domain (33 GB in size).
    -   Due to copyright constraints, only sample scripts and a few
        samples from the pre-training corpus are provided. The details
        of the pre-training corpus are documented in Section 2 and the
        Appendix of the paper.

6.  **Pre-Training Process:**

    -   ConfliBERT utilizes the same pre-training scripts (`run_mlm.py`)
        from Huggingface. An example is provided for pre-training on 8
        GPUs, though parameters should be adapted based on the user's
        available resources.

7.  **Citation:**

    -   If used for research purposes, it is recommended to cite the
        original paper. The citation details are provided in the
        repository.

## Main Tasks

-   NER

-   Classification of events

-   Masking and coding tasks

# NER (Named Entity Recognition) with ConfliBERT {#ner-named-entity-recognition-with-conflibert}

Named Entity Recognition (NER) is a fundamental task in the field of
natural language processing (NLP) that involves identifying and
classifying key information (entities) in text into predefined
categories. These categories can include names of persons,
organizations, locations, expressions of times, quantities, monetary
values, percentages, etc.

ConfliBERT, like BERT, is inherently capable of recognizing and
categorizing entities. It uses the context within which words appear to
determine their meaning and classify them accordingly. The standard
model can identify a variety of entities such as:

-   `PERSON`: People, including fictional persons.
-   `GPE`: Geopolitical entities, like countries, cities, and states.
-   `ORG`: Organizations, including governmental, companies, and
    agencies.

ConfliBERT enhances BERT's capabilities by being specifically fine-tuned
on conflict and event data. This specialized focus enables it to be more
effective in contexts related to political science, international
relations, and conflict studies.

> What are you trying to get out of this? Actors or statements (verbs)? Sentence has been NER'd vs 

### Fine-Tuning for Custom Entities

However, the scope of entities that can be recognized is not limited to
the defaults. Users can fine-tune ConfliBERT on domain-specific corpora
that contain unique entities of relevance to their research. For
instance, one might train it to recognize military equipment, political
parties, or specific event-related terminology.

To integrate more entities and classifications, researchers can proceed
with the following steps:

1.  **Data Collection**: Compile a dataset where the additional entities
    are labeled in the context of their use.
2.  **Annotation**: Use a consistent annotation schema to mark up the
    entities in the dataset.
3.  **Model Training**: Fine-tune ConfliBERT on this annotated dataset.
    This involves training the model to adjust its weights and biases to
    better predict your custom labels.
4.  **Evaluation**: Assess the model's performance on a separate
    validation dataset and iterate on your approach as needed.

Fine-tuning allows ConfliBERT to expand its understanding and
recognition capabilities to align with specific research needs or
domain-specific requirements.

<p>[President Biden]{style="color: blue;"}
[PERSON]{style="font-weight: bold;"} said that the
[U.S.]{style="color: green;"} [GPE]{style="font-weight: bold;"} alliance
with [Japan]{style="color: green;"} [GPE]{style="font-weight: bold;"} is
stronger than ever.</p>

<p>[Saudi Arabia]{style="color: green;"}
[GPE]{style="font-weight: bold;"} expressed intent to restore diplomatic
relations with [Iran]{style="color: green;"}
[GPE]{style="font-weight: bold;"}.</p>

<p>[China]{style="color: green;"} [GPE]{style="font-weight: bold;"}
provided humanitarian aid to [Turkey]{style="color: green;"}
[GPE]{style="font-weight: bold;"}.</p>

<p>[President Zelenskyy]{style="color: blue;"}
[PERSON]{style="font-weight: bold;"} accused
[Russia]{style="color: green;"} [GPE]{style="font-weight: bold;"} of war
crimes.</p>

<p>[Israeli forces]{style="color: orange;"}
[ORG]{style="font-weight: bold;"} attacked
[Hamas]{style="color: orange;"} [ORG]{style="font-weight: bold;"} in
[Gaza City]{style="color: green;"}
[GPE]{style="font-weight: bold;"}.</p>

Event data, in conflict research, refers to the systematic and
chronological cataloging of political interactions, actions, and
conflicts. It is instrumental in detailing when and where an incident
happened, who was involved, and the nature of the event. The granularity
of event data can range from large-scale political changes to minor
altercations. Thus, the swift and accurate extraction of these data
points from voluminous and diverse sources is paramount.

Here is how NER, when integrated with ConfliBERT, aids in the enrichment
of event data:

1.  **Contextual Identification**: With the specialized pre-training of
    ConfliBERT on conflict and political violence texts, the NER process
    becomes adept at distinguishing entities that are of significance in
    this domain. For instance, while general NLP models might identify
    'Aleppo' merely as a location, ConfliBERT can contextualize it
    within the Syrian conflict, recognizing it as a key site of unrest.

2.  **Actor Recognition and Classification**: Political events often
    involve a myriad of actors - from state agents and rebel groups to
    international entities. ConfliBERT's NER can discern and categorize
    these actors, providing clarity on the roles they play in specific
    incidents. This facilitates a richer and more layered analysis of
    events.

3.  **Event Dynamics and Temporal Analysis**: By associating entities
    with timestamps and specific actions, ConfliBERT can provide a
    chronological sequence of events. This temporal dimension is vital
    in understanding the progression of conflicts and predicting
    potential future escalations.

4.  **Enhancing Manual Coding**: Manual coding of event data, a
    traditional method, can be significantly enhanced by automating the
    extraction process. ConfliBERT can quickly scan through vast amounts
    of data, flagging key entities and events, thereby reducing the
    manual labor and errors inherent in such a task.

5.  **Real-time Analysis**: Given the swift nature of NER in ConfliBERT,
    real-time sources like news articles or social media feeds can be
    processed promptly. This ensures that event datasets remain
    up-to-date, capturing the dynamism of political conflicts.

6.  **Event Characterization**: Beyond just identifying entities, NER in
    ConfliBERT can also help in characterizing the event itself. By
    recognizing and associating various entities and actions, the model
    can provide insights into the nature of the event, be it a peaceful
    protest, armed insurgency, or a diplomatic negotiation.

In essence, the synergy between ConfliBERT's NER capabilities and event
data extraction offers a transformative approach to conflict research.
It streamlines the data gathering process, enhances the depth and
breadth of analysis, and empowers researchers with a tool that resonates
with the nuances of political violence and upheaval.

## Inputs for NER

Basic NER terminology:

-   **B** (Beginning) denotes the beginning of an entity.
-   **I** (Inside) marks the subsequent words of a multi-word entity.
-   **O** (Outside) is used for non-entity words.

### Inputs for NER using the ConfliBERT model:

#### 1. **Text Pre-processing**:

Before feeding the data to ConfliBERT (or any BERT-like model), one
needs to ensure the text data is pre-processed:

-   **Tokenization**: Convert sentences into tokens. Depending on the
    language and context, tokenization might split words into subwords
    or characters.

-   **Lowercasing**: This step is optional and based on the pre-trained
    ConfliBERT model. Some BERT models are case-sensitive.

-   **Special Tokens**: BERT models usually expect [CLS] and [SEP]
    tokens to mark the beginning and end of a sentence, respectively.

    For example:
    `[CLS] U.S. military chief General Colin Powell said ... [SEP]`

-   **Padding and Truncating**: Make sure each input sequence has the
    same length by either padding short sequences or truncating long
    ones.

#### 2. **Setting Options for Classification**:

-   **Entity Tags**: Define a fixed set of entity tags, such as `B-S`,
    `I-S`, `B-T`, `I-T`, and `O`. This is crucial for the classification
    layer's output size.

-   **Loss Function**: Since NER is a multi-class classification
    problem, use a categorical cross-entropy loss.

-   **Model Architecture**: Depending on the version and customization
    of ConfliBERT, ensure the last layer is a dense layer with an output
    size equal to the number of unique entity tags.

#### 3. **Input Files**:

The files provided seem to be in a CoNLL-like format, which is standard
for NER tasks. A breakdown of the structure:

-   Each word is on a new line with its respective IOB-tag.
-   Sentences or sequences are separated by blank lines.

Example:

```         
U.S. B-S
military I-S
...
. O

NATO B-S
...
. O
```

**Note**: The input file's structure needs to be consistent for
effective model training and evaluation.

#### 4. **Additional Metadata (Optional)**:

Depending on the specificities of the ConfliBERT model or the task
requirements:

-   **Attention Masks**: Used to tell the model to pay attention to
    specific tokens and ignore others (like padding tokens).
-   **Segment IDs**: If handling multiple sentences in a sequence,
    segment IDs can distinguish them.

#### 5. **Training and Evaluation Data**:

-   **Training Data**: Pre-processed sentences along with their entity
    tags for training the model.
-   **Evaluation Data**: A separate set of pre-processed sentences and
    their tags to validate the model's performance.

### Conclusion:

Users should prepare their data in a CoNLL-like format, ensure text
pre-processing aligns with ConfliBERT's requirements, and define a set
of entity tags for classification. The more consistent and clean the
input data, the better the model's performance.

Based on the given example outputs, the sections for "Outputs from NER,"
"Metrics for NER," and "Evaluation of NER" are provided next.

------------------------------------------------------------------------

## Outputs from NER

**Sentence:** "For the first time in decades, there is at least the
potential of an armed clash with America's largest adversaries, Russia
and China."

**Predictions:** The named entity recognition model has identified three
entities in the given sentence: - `decades`: Temporal Entity -
`America’s`: Government - `Russia`: Government - `China`: Government

**Raw Outputs:** There also is likely the raw logits or scores
associated with each token for different potential named entity classes
given with the outputs. For brevity, they are not all listed here, but
they can be useful for diving deeper into model decisions or for model
calibration.

------------------------------------------------------------------------

## Metrics for NER

Here is a general format you might encounter:

-   **Precision:** This is a measure of the accuracy provided that a
    specific class (or label) has been predicted. It is the number of
    true positives divided by the sum of true positives and false
    positives.

-   **Recall:** This is a measure of the ability of the model to find
    all the relevant cases within a dataset. It is the number of true
    positives divided by the sum of true positives and false negatives.

-   **F1-Score:** The F1 score is the harmonic mean of precision and
    recall. It provides a balance between the two. When it is closer to
    1, it indicates better performance, and 0 indicates poorer
    performance.

-   **Accuracy:** This metric calculates the ratio of correctly
    predicted observation to the total observations.

(Note: For a complete evaluation, one would typically calculate these
metrics for each entity type, as well as overall.)

------------------------------------------------------------------------

## Evaluation of NER

The model seems to correctly identify the governmental entities
`America’s`, `Russia`, and `China`. Additionally, the model successfully
picked up on the temporal entity `decades`.

However, the evaluation of the model's performance would ideally require
a much larger test dataset, encompassing a diverse range of sentences
and contexts. Furthermore, an in-depth evaluation would involve
comparing the model's predictions against a ground truth or a labeled
dataset to compute metrics like precision, recall, and F1-score.

One key aspect to consider in such models is their confidence scores (or
probabilities) associated with predictions. From the raw outputs, we can
extract these values to potentially set thresholds or make informed
decisions.

------------------------------------------------------------------------

Remember, while metrics provide a quantitative measure of performance,
qualitative analysis (like manually examining where the model goes right
or wrong) is invaluable, especially when deploying in critical
applications.

## [SATP_12.20 Pipeline](https://github.com/javierosorio/SATP/tree/master)

Here is a detailed breakdown of the preprocessing and training steps for
Named Entity Recognition (NER):

### 1. Annotation Pipeline

-   **File**: `1_annotation_pipeline-12.21.ipynb`
-   **Description**: This notebook likely contains the process for
    manually annotating or labelling the data. The annotations refer to
    the entities within the text that the model will later be trained to
    recognize.

### 2. Preparing the Training and Testing Data

-   **File**: `2_Preprocess-Data-12.27.ipynb`
-   **Description**: After annotation, the data needs to be prepared for
    model training. This might include splitting the data into training,
    validation, and test sets; tokenizing the data; and converting it
    into a format suitable for deep learning frameworks.

### 3. BiLSTM Baseline Deep Learning Model

-   **File**: `3_BiLSTM.ipynb`
-   **Description**: This notebook contains the implementation and
    training process of a BiLSTM (Bidirectional Long Short-Term Memory)
    model. BiLSTMs are a type of recurrent neural network that can
    capture context from both past and future input sequences.

### 4. BERT Model

-   **File**: `4_transformer_reduced_bert.ipynb`
-   **Description**: Here, the BERT (Bidirectional Encoder
    Representations from Transformers) model is implemented and trained.
    BERT is a popular transformer-based model known for its
    effectiveness in various NLP tasks, including NER.

### 5. Traditional Machine Learning Model Baselines

-   **File**: `5_Baselines_reduced-2-step.ipynb`
-   **Description**: This notebook provides baseline results using
    traditional machine learning models like Support Vector Machines
    (SVM) and Logistic Regression (LR). A two-step pipeline might refer
    to a two-phase process: feature extraction and then modeling.

### 6. Hierarchical Attention Networks (HAN)

-   **Steps**:
    1.  **Creating Word Vectors and DataLoader from train.csv**:
        -   **Command**: `python create_input_files.py`
        -   **Description**: This script prepares the data and creates
            word embeddings, possibly using pre-trained models. The
            embeddings will be used to represent words or tokens in the
            data. It also sets up data loaders which are essential for
            training deep learning models in batches.
    2.  **Training and Evaluating the HAN Model**:
        -   **Command**: `python HAN_end2end.py`
        -   **Description**: Hierarchical Attention Networks are neural
            models that can pay differentiated attention to various
            parts of the input data, making them suitable for tasks
            where the importance of different parts of the input varies.

### Results:

The results section showcases the performance of various models on the
task:

-   Metrics used include:
    -   **Accuracy**: The ratio of correctly predicted entities to the
        total entities.
    -   **Precision**: The ratio of correctly predicted positive
        entities to the total predicted positives.
    -   **Recall**: The ratio of correctly predicted positive entities
        to the total actual positives.
    -   **F1 Score**: The harmonic mean of precision and recall.
    -   **Exact Matching Ratio**: This might refer to the ratio of data
        points where the predicted entities exactly match the true
        entities.
-   The models and their respective performance metrics are tabulated
    for comparison.

|      Model       | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Exact Matching Ratio (%) |
|:----------:|:----------:|:----------:|:----------:|:----------:|:------------:|
|   One vs Rest    |    26.07     |     87.19     |   27.40    | 28.04  |          82.20           |
| Binary Relevance |    26.65     |     86.08     |   28.29    | 28.40  |          82.47           |
|   Class Chain    |    27.93     |     83.47     |   29.78    | 29.32  |          82.87           |
|  Label Powerset  |    28.86     |     85.53     |   30.51    | 30.05  |          83.14           |
|      BiLSTM      |    37.11     |     53.81     |   58.51    | 42.12  |          76.23           |
|       HAN        |    52.98     |     65.64     |   74.82    | 55.78  |          83.07           |
|       BERT       |    62.52     |     74.73     |   80.01    | 64.59  |          87.23           |

This comprehensive pipeline takes the data from raw, annotated form and
processes it into a format that various machine learning and deep
learning models can be trained on. The models' performances are then
evaluated and compared using the mentioned metrics.

## Other LLM comparisons on this task

## Related Literature in NER

# Classification with ConfliBERT

Text classification is the process of assigning tags or categories to
text according to its content. ConfliBERT can be trained to classify
text into categories such as 'Verbal Cooperation', 'Material
Cooperation', 'Verbal Conflict', and 'Material Conflict', based on the
context and content of the input text.

### Expanding Classification Categories

@osorio2019translating

Similar to entity recognition, classification categories can be
tailored. Researchers can introduce additional categories like "Economic
Sanctions", "Legal Actions", "Diplomatic Events", etc., by fine-tuning
ConfliBERT with appropriately tagged training data. This custom
classification can help in creating more nuanced datasets that capture
the complexity of international relations and conflict scenarios.

The fine-tuning process for expanding classification capabilities
involves:

1.  **Category Definition**: Define clear and distinct categories that
    you want ConfliBERT to classify the text into.
2.  **Data Annotation**: Label a sufficiently large corpus of text with
    these categories.
3.  **Model Training**: Use this labeled data to fine-tune ConfliBERT so
    that it can predict the probability of any given text belonging to
    these categories.
4.  **Model Evaluation**: Test the model's classification accuracy on
    unseen data and refine your categories and data as needed.

| Sentence                                                                      | Make a Statement | Verbal Cooperation | Material Cooperation | Verbal Conflict | Material Conflict |  Predicted Category  |
|-------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| President Biden said that the U.S. alliance with Japan is stronger than ever. |       0.9        |        0.05        |         0.02         |      0.02       |       0.01        |   Make a Statement   |
| Saudi Arabia expressed intent to restore diplomatic relations with Iran.      |       0.1        |        0.8         |         0.05         |      0.03       |       0.02        |  Verbal Cooperation  |
| China provided humanitarian aid to Turkey.                                    |       0.05       |        0.1         |         0.8          |      0.03       |       0.02        | Material Cooperation |
| President Zelenskyy accused Russia of war crimes.                             |       0.02       |        0.03        |         0.05         |      0.85       |       0.05        |   Verbal Conflict    |
| Israeli forces attacked Hamas in Gaza City.                                   |       0.01       |        0.02        |         0.03         |       0.1       |       0.84        |  Material Conflict   |

Please note, the logits (probabilities) here are illustrative and assume
that the model is making predictions with high confidence and that each
sentence strongly represents its respective category. The "Predicted
Category" is determined by the highest logit score for each sentence. In
real-world scenarios, the logits would be the actual output from a model
like ConfliBERT after processing the sentences.

Here's the revised section, incorporating HTML inline styling for color
changes as per your examples:

------------------------------------------------------------------------

## Understanding Classification Metrics in ConfliBERT

This document provides an overview of classification metrics in the
context of political event categorization using ConfliBERT, drawing upon
foundational concepts in machine learning classification such as True
Positives (TP), False Positives (FP), False Negatives (FN), and True
Negatives (TN) as explained in the Google Machine Learning Crash Course
(see @googleMLCrashCourse). We will use a hypothetical example to
illustrate these concepts clearly:

**Metrics Explained:**

<p>[**True Positives (TP):** Sentences where the Predicted Category
matches the actual category (highest logit
score).]{style="color: green;"}</p>

<p>[**False Positives (FP):** Cases where the Predicted Category was
chosen, but it is not the actual category.]{style="color: red;"}</p>

<p>[**False Negatives (FN):** Cases where the actual category was
correct, but the Predicted Category was
different.]{style="color: red;"}</p>

<p>[**True Negatives (TN):** All other cases where neither the predicted
nor the actual category was chosen.]{style="color: green;"}</p>

**Hypothetical Example for Classification:**

Consider an event: *Israeli forces attacked Hamas in Gaza City.* Let's
place this event in a 2x2 confusion matrix to understand potential
outcomes from ConfliBERT's classification:

| True Positive (TP)                                                                                                                | False Negative (FN)                                                                                                                                |
|----------------------------------|--------------------------------------|
| ConfliBERT accurately classifies the event as "Material Conflict" due to the active engagement and physical altercation reported. | ConfliBERT incorrectly classifies the event as a non-conflict category, such as "Verbal Cooperation," missing the material nature of the conflict. |

| False Positive (FP)                                                                                                                      | True Negative (TN)                                                                                                |
|---------------------------------------|---------------------------------|
| ConfliBERT incorrectly classifies a peaceful negotiation as "Material Conflict," suggesting a physical altercation where there was none. | ConfliBERT correctly identifies that no material conflict occurred when Israeli forces did not engage with Hamas. |

**Detailed Breakdown:**

<p>[**True Positive (TP):** - *Reality:* Israeli forces engage in a
physical altercation with Hamas in Gaza City. - *ConfliBERT Says:*
"Material Conflict." - *Outcome:* ConfliBERT's classification aligns
with the real-world event, providing accurate data for
analysis.]{style="color: green;"}</p>

<p>[**False Positive (FP):** - *Reality:* A diplomatic meeting between
Israeli and Hamas representatives occurs without physical
confrontation. - *ConfliBERT Says:* "Material Conflict." - *Outcome:*
ConfliBERT erroneously signals a conflict, potentially leading to
misinformed analysis.]{style="color: red;"}</p>

<p>[**False Negative (FN):** - *Reality:* Israeli forces attack Hamas in
Gaza City. - *ConfliBERT Says:* "Verbal Cooperation" or another
non-conflict category. - *Outcome:* Misclassification downplays the
severity, potentially affecting response
strategies.]{style="color: red;"}</p>

<p>[**True Negative (TN):** - *Reality:* Israeli and Hamas
representatives engage in peaceful talks. - *ConfliBERT Says:* "Verbal
Cooperation" or an appropriate non-conflict category. - *Outcome:*
ConfliBERT accurately reflects the peaceful nature of the event, aiding
correct analysis and policy decisions.]{style="color: green;"}</p>

**Applying Metrics for Evaluation:**

The subsequent sections will explore how to use these
metrics---sensitivity, specificity, precision, and accuracy---to
evaluate ConfliBERT's performance in classifying complex political
events. These metrics are essential for assessing ConfliBERT's accuracy
and fine-tuning its predictions to align with real-world observations.

Here is a deep dive into how ConfliBERT enhances classification in the
context of conflict research:

1.  **Granular Event Categorization**: While it is essential to identify
    an incident as a protest or a riot, the true value lies in
    discerning the nature of these events. ConfliBERT's classification
    capability can distinguish between a peaceful protest, a violent
    uprising, or even a state-sanctioned crackdown. This granularity
    ensures a comprehensive understanding of the event's nature.

2.  **Adaptive Learning with Domain-Specific Data**: Given ConfliBERT's
    training on conflict-related texts, it possesses a heightened
    sensitivity to the subtle nuances within conflict narratives. This
    domain-specific expertise translates to a more accurate and
    contextual classification, be it categorizing a skirmish as
    sectarian, ethnic, or political.

3.  **Temporal Classification**: Conflicts evolve, and so do their
    narratives. ConfliBERT can classify events based on their temporal
    context, distinguishing between a long-standing civil war's initial
    skirmishes and its climax battles or identifying the phases of
    diplomatic negotiations.

4.  **Automating Large-Scale Data Processing**: Conflict research often
    grapples with voluminous data from diverse sources like news
    articles, social media, and firsthand accounts. Manual
    classification of such vast data is not only labor-intensive but
    also prone to inconsistencies. ConfliBERT automates this, ensuring
    uniformity and efficiency.

5.  **Real-time Classification for Dynamic Responses**: In the rapidly
    changing landscape of political conflicts, timely responses are
    crucial. With its swift classification capabilities, ConfliBERT can
    process real-time data, ensuring that researchers and policymakers
    are equipped with classified insights as events transpire.

6.  **Supporting Cross-Referencing and Validation**: By classifying
    data, ConfliBERT also aids in cross-referencing. For instance, if
    two sources provide conflicting narratives of an event, having them
    classified can help researchers quickly juxtapose and validate the
    information.

7.  **Assisting Predictive Analysis**: Once events are classified
    systematically, it becomes feasible to perform predictive analysis.
    Recognizing patterns from past events can provide insights into
    potential future scenarios, aiding preemptive measures and
    strategies.

In summary, ConfliBERT's classification capabilities not only structure
the unorganized maze of conflict data but also elevate the depth and
breadth of analysis. By providing categorized, context-aware insights,
it becomes an invaluable asset for researchers, analysts, and
policymakers navigating the intricate domain of conflict research.

For a different example, let's create a hypothetical news report about a
political demonstration that turned violent. This example will have more
variance in the types of police actions and outcomes, and it will
include different categorizations.

**Original Text (Pre-Processing)**: "During a large political
demonstration in the capital, clashes erupted between protesters and
police. Police reportedly used water cannons and rubber bullets to
control the situation. Several protesters were detained, and there were
reports of injuries among both police and demonstrators. A local shop
was vandalized during the chaos. The protest was eventually dispersed,
and order was restored by the authorities."

**Processed Text (Post-Processing)**: The text is segmented into
individual sentences and processed for analysis, with each sentence
receiving specific categorizations based on the content.

| Processed Sentence                                                                                   | USE_OF_FORCE | DETENTION | INJURIES | VANDALISM | ORDER_RESTORED |
|--------------------------------|--------|--------|--------|--------|--------|
| During a large political demonstration in the capital, clashes erupted between protesters and police | 0            | 0         | 0        | 0         | 0              |
| Police reportedly used water cannons and rubber bullets to control the situation                     | 1            | 0         | 0        | 0         | 0              |
| Several protesters were detained                                                                     | 0            | 1         | 0        | 0         | 0              |
| There were reports of injuries among both police and demonstrators                                   | 0            | 0         | 1        | 0         | 0              |
| A local shop was vandalized during the chaos                                                         | 0            | 0         | 0        | 1         | 0              |
| The protest was eventually dispersed, and order was restored by the authorities                      | 0            | 0         | 0        | 0         | 1              |

In this table: - "USE_OF_FORCE" indicates if the sentence suggests the
use of force by police. - "DETENTION" denotes if the sentence implies
that protesters were detained. - "INJURIES" signifies if there were
injuries reported among the police or demonstrators. - "VANDALISM" marks
if there was property damage or vandalism mentioned. - "ORDER_RESTORED"
is affirmative if the sentence indicates that order was restored by the
authorities.

This structured approach to analyzing news text provides a clear
understanding of the various aspects of the event, which is essential
for conflict analysis and research.

## Inputs for Classification

To harness the capabilities of ConfliBERT for classification, one needs
to be well-acquainted with the expected input format, necessary
pre-processing steps, and customization options:

**Text Pre-processing:**

1.  **Tokenization:**
    -   Tokenizing is the process of breaking down the text into smaller
        chunks, often words or sub-words.
    -   Example: The sentence "UNRWA said it had suspended aid" will be
        tokenized as ["UNRWA", "said", "it", "had", "suspended", "aid"].
2.  **Formatting:**
    -   Based on the dataset structure provided, the text should be
        divided into specific segments namely 'sentence', 'source', and
        'target'.
    -   Example: For the text "UNRWA said it had suspended aid
        deliveries to Gaza", "UNRWA" will be the source, "Gaza" will be
        the target, and the whole string is the sentence.
3.  **Special Character Removal:**
    -   It is crucial to ensure that any non-textual or special
        characters that do not contribute to the semantic meaning of the
        sentence are removed.
    -   Example: If the sentence is "UNRWA said it had suspended
        aid!!!", the exclamatory marks can be removed for a cleaner
        input as "UNRWA said it had suspended aid".
4.  **Lowercasing/Uppercasing:**
    -   Depending on the model variant being used (cased or uncased),
        ensure that the text is transformed accordingly.
    -   Example: For the uncased variant, "UNRWA said it had suspended
        aid" becomes "unrwa said it had suspended aid".
5.  **Sentence Segmentation:**
    -   If the input is a long paragraph or document, break it down into
        individual sentences.
    -   Example: For a paragraph "UNRWA said it had suspended aid. It
        was a major decision.", the segmented sentences will be ["UNRWA
        said it had suspended aid.", "It was a major decision."].

**Customizing Classification with ConfliBERT:**

1.  **Fine-tuning:**
    -   Although ConfliBERT is pre-trained, its versatility shines when
        fine-tuned on a specific dataset. This way, the model becomes
        more attuned to the nuances of the data.
    -   Example: If the focus is on classifying statements made in a
        legal context, fine-tune ConfliBERT on legal documents or court
        verdicts.
2.  **Using Different Model Variants:**
    -   Choose a ConfliBERT version that aligns with the nature of the
        text data. If the distinction between uppercase and lowercase
        letters is significant, opt for a cased model.
3.  **Adjusting Model Parameters:**
    -   During fine-tuning or prediction, tweak model parameters like
        learning rate, batch size, or epoch count to optimize
        performance.
4.  **Feedback Loop:**
    -   Consider setting up a feedback loop where misclassifications can
        be corrected and fed back into the model for continuous
        learning.

### Define and discuss update methods

There are four versions of ConfliBERT, each serving a specific purpose:

| Version                  | Definition                                                                                                                                                                        |
|--------------|---------------------------------------------------------|
| ConfliBERT-scr-uncased:  | This version is pre-trained from scratch using a custom uncased vocabulary. This is the preferred version as it has been built from the ground up with a specific vocabulary set. |
| ConfliBERT-scr-cased:    | This version is also pre-trained from scratch but uses a custom cased vocabulary. This means it differentiates between uppercase and lowercase letters.                           |
| ConfliBERT-cont-uncased: | This version is built by continually pre-training on top of the original BERT's uncased vocabulary.                                                                               |
| ConfliBERT-cont-cased:   | This version is similar to the above but uses the original BERT's cased vocabulary.                                                                                               |

**Scratch versus Continuous Defined:** 1. **Scratch:** This refers to
building the model from scratch, meaning the model is trained without
any prior knowledge. The vocabulary is also built from the base without
any prior context. 2. **Continuous:** Continuous pre-training means the
model is built on top of an existing model (in this case, BERT). This
leverages the knowledge already present in BERT and fine-tunes it for a
specific task.

## Outputs from Classification

When ConfliBERT provides a classification output, it organizes the
results hierarchically and presents detailed information for clarity and
to aid interpretation. Here is a deep dive into what each segment of the
output signifies:

1.  **Index and Basic Information:**
    -   **Index:** Each entry or instance in the dataset being
        classified is assigned a unique identifier or index.
    -   **Gold Penta:** This is a reference label, indicating the
        correct classification as provided in the dataset.
    -   **Sentence:** The original textual input that was classified.
    -   **Source:** The entity or group mentioned as the originator or
        subject of the action in the sentence.
    -   **Target:** The entity or group mentioned as the recipient or
        object of the action in the sentence.
2.  **Level 1 Classification:**
    -   **Tense:** The temporal context in which the event occurred,
        like 'past' or 'future'.
    -   **Prompt Text:** A standardized, simplified representation of
        the sentence. For instance, if the sentence is "UNRWA said it
        had suspended aid deliveries to Gaza", the prompt text might be
        "'Source' reduced aid to 'Target'", where 'Source' represents
        the source (UNRWA) and 'Target' represents the target (Gaza).
    -   **Root Code:** This signifies the primary, overarching
        classification category. For example, "SANCTION".
    -   **Score:** This provides a confidence measure. The higher the
        score, the more confident the model is about its classification.
3.  **Level 1 Evaluation:**
    -   **Gold Root vs. Prediction:** Here, the model's prediction for
        the root code is compared to the correct answer or "gold"
        standard. For instance, if the gold standard is "SANCTION" and
        the model also predicts "SANCTION", then this comparison would
        indicate a match.
    -   **Gold Penta vs. Prediction:** Similarly, the model's prediction
        for the gold penta is compared to the correct answer. If both
        match, the model's prediction is considered correct for that
        instance.
4.  **Level 2 Classification:**
    -   **Root Code:** The primary classification category is reiterated
        here for clarity.
    -   **Prompt Text:** Similar to level 1, this provides a
        standardized representation of the sentence, but might be
        slightly more detailed or provide alternate interpretations.
    -   **Future and Past Scores:** For each classification prompt,
        there might be separate scores indicating the model's confidence
        for both future and past contexts. This helps understand how the
        model perceives the temporal aspect of the event.
    -   **L2 Root:** This provides a more detailed or nuanced
        classification based on the primary root code from level 1.
5.  **Level 2 Evaluation:**
    -   This section is similar to the level 1 evaluation but pertains
        to the more detailed classifications from level 2.
    -   **Gold Root vs. Prediction:** The model's level 2 root code
        prediction is compared to the correct or "gold" root code. If
        both are the same, the prediction is accurate.
    -   **Gold Penta vs. Prediction:** The predicted gold penta value at
        level 2 is compared against the correct value. If they match,
        the model's prediction is deemed correct for that instance.

In essence, the output from ConfliBERT provides both a high-level and a
detailed classification of the input sentence, complete with confidence
scores, standardized representations, and a direct comparison to known
correct answers for evaluation purposes. This hierarchical structure
ensures a comprehensive understanding of the sentence's context,
entities involved, and the nature of their interaction.

## Extant Evaluation datasets:

[The
datasets](https://github.com/eventdata/ConfliBERT#evaluation-datasets)

## Huggingface usage here

(To be filled with information on how to use ConfliBERT with the
Huggingface library, if applicable.)

## Metrics for Classification

(To be filled with relevant metrics used to evaluate the performance of
ConfliBERT for classification, e.g., accuracy, F1 score, etc.)

## Evaluation of Classification

(To be filled with details on how ConfliBERT's classification
performance was evaluated, including methodologies, datasets used, and
results.)

## Other

# Masking and Coding Tasks with ConfliBERT

**[MASK]** said that the U.S. alliance with Japan is stronger than ever.

Saudi Arabia expressed intent to restore diplomatic relations with
**[MASK]**.

**[MASK]** provided humanitarian aid to Turkey.

President Zelenskyy accused **[MASK]** of war crimes.

**[MASK]** forces attacked Hamas in Gaza City.

Language models like ConfliBERT employ a technique known as "masked
language modeling" during the training process. Here is an in-depth
breakdown of how masking functions within ConfliBERT:

**1. The Principle of Masking:**\
Masking is the process where a certain percentage of input tokens are
replaced with a special `[MASK]` token. This technique challenges the
model to predict the masked word from its context. This approach ensures
that the model pays attention to both the left and the right context of
a word, fostering bidirectional understanding.

**2. Benefits for Conflict Research:**\
In conflict research, data can often be inconsistent, fragmented, or
incomplete due to the sensitive nature of the information or the
challenges in data collection in conflict zones. Masking in ConfliBERT
can be pivotal as: - It trains the model to reconstruct missing or
obscured information. - It allows the model to predict contextually
relevant terms, crucial for accurate event classification and
relationship extraction.

**3. Applications in Downstream Tasks:**\
The masked language modeling task equips ConfliBERT to handle various
downstream tasks such as named entity recognition, relationship
extraction, and event classification. The model's ability to predict
masked entities can aid in: - Identifying relevant actors or groups in
conflict narratives. - Extracting the type of conflict event (e.g.,
bombing, armed assault). - Predicting relationships between entities in
a given text.

**4. Enhancing Domain-Specific Understanding:**\
The specific vocabulary and context associated with the political
conflict domain make it challenging. When ConfliBERT undergoes training
with masking on conflict-specific datasets, it becomes better at
recognizing domain-specific terms, like "separatists", and understanding
their relevance. This is crucial, as standard models might misinterpret
or fragment these terms.

**5. Integration with Other Pre-training Techniques:**\
While masking is essential, it is one of many pre-training techniques.
In conjunction with techniques like next-sentence-prediction, ConfliBERT
can develop a deeper understanding of text sequences and relationships,
enhancing its performance on complex conflict data.

In conclusion, masking is not just a technique but a cornerstone in the
training regimen of ConfliBERT. It ensures that the model is
well-equipped to tackle the intricate nuances and challenges associated
with political conflict research. Through effective masking strategies,
ConfliBERT becomes adept at providing comprehensive insights into
conflict narratives, making it a valuable tool for conflict researchers
and analysts.

## Inputs for Masking and Coding

To utilize ConfliBERT for masking and coding tasks, the input data must
be structured and pre-processed according to specific guidelines.

1.  **Tokenization**: The text should be tokenized into sub-words or
    words. The tokenization method will depend on the pre-trained model
    and tokenizer available for ConfliBERT.

2.  **Formatting**: For the `[MASK]` task, specific tokens or words in
    the sentence should be replaced with the `[MASK]` token. The model
    will then predict the missing token based on the context provided by
    the rest of the sentence.

3.  **Segmentation**: If the input consists of multiple sentences, they
    should be divided into distinct segments. Each segment will be input
    separately into the model.

4.  **Padding**: All inputs should be padded to have the same length for
    batching during model training or inference.

## Outputs from Masking and Coding

1.  **Predicted Tokens**: The primary output from the masking task is
    the token predicted by ConfliBERT to replace the [MASK] token. The
    model will provide a probability distribution over the vocabulary,
    and the token with the highest probability is usually taken as the
    prediction.

2.  **Embeddings**: ConfliBERT will also produce embeddings for each
    token in the input sequence. These embeddings can be used for
    downstream tasks or further analysis.

3.  **Coding Outputs**: Depending on the coding task, ConfliBERT might
    classify a given input into predefined categories or produce other
    forms of structured output.

## Metrics for Masking and Coding

1.  **Accuracy**: Measures the percentage of [MASK] tokens that the
    model predicted correctly.

2.  **Perplexity**: Provides insight into how well the probability
    distribution predicted by the model aligns with the actual
    distribution of the [MASK] token.

3.  **F1-Score**: Used mainly for coding tasks, it considers both
    precision and recall to provide a more holistic view of model
    performance.

4.  **Loss**: The model's objective function value, which it tries to
    minimize during training.

## Evaluation of Masking and Coding

1.  **In-domain vs. Out-of-domain Evaluation**: It is essential to
    evaluate the model's performance both on data similar to its
    training data and on entirely different datasets.

2.  **Human Evaluation**: Sometimes, multiple tokens can correctly fill
    a [MASK] token, so human evaluators can provide more nuanced
    feedback on the model's predictions.

3.  **Ablation Studies**: Evaluating the model's performance by removing
    or modifying certain parts can help understand the importance of
    various components.

## Other LLM comparisons on this task

1.  **BERT**: BERT is the pioneering model that introduced the [MASK]
    task. It is insightful to compare newer models like ConfliBERT with
    BERT to see the advancements.

2.  **RoBERTa**: A variant of BERT, RoBERTa changes some key
    hyperparameters and training strategies, which can have different
    performances on the masking task.

# Computational Considerations and Benchmarks for ConfliBERT

> Sultan & Dr. Osorio - Benchmarks. Relative time - 4, 8 vs 16 cores for
> ConfliBERT vs denominator being BERT/...BERT. Do the same for GPUs.
> Graphs would be nice for this task - scale factors for future.

When implementing BERT-based models such as ConfliBERT, one needs to
consider the computational implications. While BERT benchmarks provide
useful insights, ConfliBERT, as a derivative of BERT, might exhibit
different behaviors. Here is a general breakdown based on the
information provided:

## CPU Usage (1 Core)

BERT, and by extension models like ConfliBERT, are heavily parallelized
models, which means they benefit significantly from multiple cores or
GPUs. Using just a single core would be significantly slower and is not
recommended for any sizable inference or training task. However, for
very light tasks or experimentation, a single core could suffice, albeit
with extended processing times.

## CPU (Multicore)

Based on BERT benchmarks
[<https://vincentteyssier.medium.com/bert-inference-cost-performance-analysis-cpu-vs-gpu-b58a2420b2c8>]:

### 8 vCPUs (e2-highcpu-8):

-   **Processing Time per File**: 69.4s - 70.5s for 1500 samples.
-   **Overall vCPU Usage**: Approximately 75%.
-   **Cost**: \$0.197872/hour or \$144.45 monthly.

### 16 vCPUs (e2-highcpu-16):

-   **Processing Time per File**: 40.80s - 43.34s for 1500 samples.
-   **Overall vCPU Usage**: Approximately 60%.
-   **Cost**: \$0.395744/hour or \$288.89 monthly.

### 32 vCPUs (e2-highcpu-32):

-   **Processing Time per File**: 35.73s - 40.21s for 1500 samples.
-   **Overall vCPU Usage**: Approximately 40%.
-   **Cost**: \$0.791488/hour.

The overarching pattern reveals that as the number of CPU cores
increases, the processing time decreases. However, there is a point of
diminishing returns: scaling from 16 to 32 vCPUs does not halve the
processing time. Additionally, the vCPU usage decreases, indicating
underutilization of resources with higher cores.

## With GPUs and other Architectures (non-Intel/AMD)

GPUs, especially those designed for deep learning tasks, like the NVIDIA
Tesla series, provide significant speed-ups for BERT-like models due to
their parallel processing capabilities.

### 1 GPU (n1-standard-8 + NVIDIA Tesla V100):

-   **Processing Time per File**: 2.16s - 3.28s for 1500 samples. This
    speed is around 25 times faster than the 8 vCPU instance and
    approximately 15 times faster than the 32 vCPU instance.
-   **Cost**: \$2.004/hour or \$1,462.99 monthly.

For specialized architectures beyond Intel/AMD, the benchmarks would
vary based on the specific architecture. For example, FPGAs or TPUs
might offer different performance and cost metrics. However, TPUs, in
particular, have been known to provide excellent performance for
TensorFlow-based models, including BERT derivatives.

In conclusion, while BERT benchmarks give an idea of the computational
requirements, users should perform their own benchmarks to ascertain the
exact requirements for ConfliBERT. Always consider the scale of the
task, available budget, and desired processing time when choosing the
computational infrastructure.

### Apple Silicon & ARM Computing Architectures

## Experience with ATLAS / DELTA

# ConfliBERT Variants

## ConfliBERT Arabic (AR)

## ConfliBERT Spanish (ES)

## Extending ConfliBERT to other languages

# References / Citations
