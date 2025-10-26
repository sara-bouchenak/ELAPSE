# ELAPSE Datasets

**ELAPSE** framework uses 9 real-world datasets known for their potential bias issues, covering diverse application domains such as activity recognition and socio-economic prediction.

**Download Link:** [ELAPSE Datasets Download](https://partage.liris.cnrs.fr/index.php/s/zNEtqKgwXMLSiZo)

---

| **Dataset** | **#Tuples** | **Sensitive Attributes** | **Topic**              |
|--------------|-------------|--------------------------|------------------------|
| **Image**    |             |                          |                        |
| CelebA       | 202,599     | Gender, Age              | Emotion analysis       |
| FairFace     | 108,501     | Race, Age                | Demographic analysis   |
| **Audio**    |             |                          |                        |
| AudioMNIST   | 5,000       | Gender, Age              | Education              |
| VoxCeleb     | 5,000       | Race                     | Demographic analysis   |
| **Tabular**  |             |                          |                        |
| MobiAct      | 1,344,397   | Gender, Age              | Healthcare             |
| KDD          | 272,507     | Gender, Race, Age        | Finance                |
| ARS          | 75,128      | Gender                   | Healthcare             |
| DC           | 60,420      | Gender, Age              | Finance                |
| Adult        | 48,842      | Gender, Race, Age        | Finance                |


---


**CelebA.** The CelebA dataset comprises 202,599 celebrity face images annotated with 40 facial attributes. In our experiments, we focus on gender and age as sensitive attributes. The main task involves emotion recognition, where the objective is to predict whether a person is smiling or not.

**FairFace.** This dataset contains 108,501 human face images. We consider race and age as sensitive attributes. Race is binarized into two groups, while age is binarized into young (under 40) and old (40 and above). The prediction task consists of demographic classification, typically focusing on identifying a person’s gender.

**AudioMNIST.** The original AudioMNIST dataset contains 30,000 audio recordings of spoken digits (0–9) by 60 speakers. In our experiments, we use a subset of 5,000 samples. We consider gender and age as sensitive attributes. The task is to recognize whether the spoken digits are even or odd numbers.

**VoxCeleb.** VoxCeleb contains audio recordings from thousands of speakers extracted from YouTube interviews. We use a subset of 5,000 samples and consider race as the sensitive attribute. The dataset is used for a demographic prediction task where models are evaluated on their ability to classify speaker gender.

**MobiAct.** A large-scale dataset with 1,344,397 records from accelerometer and gyroscope sensors, used for human activity recognition. Age and gender are sensitive attributes, with age binarized into young people (i.e., aged under 50 years) and old people. The primary objective is to identify individuals who are standing and those who are not.

**ARS.** Used for the activity recognition of healthy older individuals, ARS consists of 75,128 records collected from two different clinical settings equipped with RFID antennas. 14 volunteers aged between 66 and 86 years were trialed. Each wore a wearable sensor and undertook a series of scripted activities. The classification task is to predict whether a person is lying or not for ambulatory monitoring, and the only sensitive attribute present in this dataset is gender.

**KDD.** Derived from the 1994 and 1995 US population surveys, this dataset contains 272,507 records with gender, race, and age as sensitive attributes. The prediction task involves assessing financial outcomes. We have binarized all non-binary sensitive attributes. For race, the first group comprises Whites, Asians, and Pacific Islanders, while all others constitute the second group. Meanwhile, for age, individuals aged between 30 and 60 make up the active individual group, and all others (i.e., inactive individuals) represent the other group.

**DC.** The Dutch Census dataset focuses on socio-economic factors to predict occupational prestige. It contains 60,420 records with 12 attributes including gender and age sensitive attributes. As in Adult and KDD, to binarize age we consider active people aged between 15 and 74 years, and the others as inactive.

**Adult.** Extracted from the 1994 US Census database, the Adult dataset contains 48,842 records with gender, race, and age as sensitive attributes. These attributes are binarized using the same methodology applied to KDD’s non-binary sensitive attributes. The primary prediction task for this dataset is to determine whether an individual’s salary exceeds 50𝑘.
