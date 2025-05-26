# ELAPSE Datasets

The **ELAPSE** framework uses five real-world datasets known for their potential bias issues, covering diverse application domains like activity recognition and socio-economic prediction.

**Download Link:** [ELAPSE Datasets Download](https://partage.liris.cnrs.fr/index.php/s/zNEtqKgwXMLSiZo)

---

| Dataset   | Records   | Sensitive Attributes             | Task                                     |
|-----------|-----------|---------------------------------|-----------------------------------------|
| MobiAct   | 1,344,397 | Age (young/old), Gender         | Human activity recognition (standing)   |
| ARS       | 75,128    | Gender                          | Activity recognition (lying)            |
| KDD       | 272,507   | Gender, Race, Age               | Financial outcome prediction            |
| DC        | 60,420    | Gender, Age                     | Occupational prestige prediction        |
| Adult     | 48,842    | Gender, Race, Age               | Salary prediction (>50k)                |

---

**MobiAct.** A large-scale dataset with 1,344,397 records from accelerometer and gyroscope sensors, used for human activity recognition. Age and gender are sensitive attributes, with age binarized into young people (i.e., aged under 50 years) and old people. The primary objective is to identify individuals who are standing and those who are not.

**ARS.** Used for the activity recognition of healthy older individuals, ARS consists of 75,128 records collected from two different clinical settings equipped with RFID antennas. 14 volunteers aged between 66 and 86 years were trialed. Each wore a wearable sensor and undertook a series of scripted activities. The classification task is to predict whether a person is lying or not for ambulatory monitoring, and the only sensitive attribute present in this dataset is gender.

**KDD.** Derived from the 1994 and 1995 US population surveys, this dataset contains 272,507 records with gender, race, and age as sensitive attributes. The prediction task involves assessing financial outcomes. We have binarized all non-binary sensitive attributes. For race, the first group comprises Whites, Asians, and Pacific Islanders, while all others constitute the second group. Meanwhile, for age, individuals aged between 30 and 60 make up the active individual group, and all others (i.e., inactive individuals) represent the other group.

**DC.** The Dutch Census dataset focuses on socio-economic factors to predict occupational prestige. It contains 60,420 records with 12 attributes including gender and age sensitive attributes. As in Adult and KDD, to binarize age we consider active people aged between 15 and 74 years, and the others as inactive.

**Adult.** Extracted from the 1994 US Census database, the Adult dataset contains 48,842 records with gender, race, and age as sensitive attributes. These attributes are binarized using the same methodology applied to KDD’s non-binary sensitive attributes. The primary prediction task for this dataset is to determine whether an individual’s salary exceeds 50𝑘.
