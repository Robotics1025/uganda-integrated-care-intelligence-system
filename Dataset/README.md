# Low levels of hypertension screening in HIV care clinics in rural Uganda: a mixed methods study

[https://doi.org/10.5061/dryad.9p8cz8wqg](https://doi.org/10.5061/dryad.9p8cz8wqg)

This is the data from a baseline status assessment of Hypertension (HTN) screening, treatment, and control in 52 Ugandan public health facilities, participating in an ongoing cluster randomized trial of an integrated HIV/HTN care model.  The study was funded by the *European and Developing Countries Clinical Trials Partnership* (EDCTP).

From November 2020 to March 2021, we reviewed patient records and randomly sampled 50 persons living with HIV (PLHIV) without a documented HTN diagnosis per health facility and all PLHIV with a documented HTN diagnosis per health facility. We surveyed the sampled participants, took their blood pressure measurements, and described the HTN care cascade.

## Description of the data and file structure

The data provided are in a comma-separated-value (CSV) file that includes variable names as column headers.

The data file name is Uganda_Int_HIV_HTN_baseline_EUopensci.deident.v3.csv

The corresponding data dictionary is an excel document filename is Uganda_Int_HIV_HTN_baseline_EUopensci.datadictionary.v3.xlsx

| Variable Name   | Question                                                                                                                                                       | Variable Type | Variable Codes                                                                                              |
| :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ | :---------------------------------------------------------------------------------------------------------- |
| hc\_code        | Health Center (coded)                                                                                                                                          | Numeric       | De-identifed code                                                                                           |
| clinicid        | Participant/Clinic ID: collected from the ART card/register - deidentified                                                                                     | Numeric       | De-identifed code                                                                                           |
| age\_category   | Age <=40                                                                                                                                                       | Numeric       | 0 = age > 40 1 = age <= 40                                                                                  |
| female          | Reported as female                                                                                                                                             | Numeric       | 0 = Male 1=Female                                                                                           |
| artyr           | Years on ART - calculated from ART start date (grouped)                                                                                                        | Numeric       |  1 = up\_to\_1\_year  2 = from\_2\_to\_5   3 = from\_6\_to\_9   4 = from\_10\_to\_35                        |
| hc4             | Is Clinic a Health Center 4 level                                                                                                                              | Numeric       | 0 = No 1 = Yes                                                                                              |
| smoke           | Current smoking                                                                                                                                                | Numeric       | 1 = Yes 2 = No                                                                                              |
| alcohol         | Current alcohol use                                                                                                                                            | Numeric       | 1 = Yes 2 = No                                                                                              |
| overweight      | Obesity or overweight                                                                                                                                          | Numeric       | 0 = No 1 = Yes -9 = missing                                                                                 |
| marital\_status | Current marital status                                                                                                                                         | Numeric       | 1 = Single never married 2 = Married/Cohabitating 3 = Divorced/ separated/ widowed -9 = missing             |
| exercise        | Lack of physical work or exercise                                                                                                                              | Numeric       | 0 = No 1 = Yes                                                                                              |
| bpmdate6mon     | Blood pressure measured in last 6 months at clinic                                                                                                             | Numeric       | 0 = No 1 = Yes                                                                                              |
| htn\_now        | Prior HTN diagnosis or prior HTN diagnosis not documented in health record or new HTN diagnosis                                                                | Numeric       | 0 = No 1 = Yes                                                                                              |
| category        | Patient category at baseline visit                                                                                                                             | Numeric       | 1 = HTN 2 = No known HTN                                                                                    |
| prior\_unknown  | Prior HTN diagnosis not documented in health record                                                                                                            | Numeric       | 0 = No 1 = Yes                                                                                              |
| new\_dx         | New HTN diagnosis at visit                                                                                                                                     | Numeric       | 0 = No 1 = Yes                                                                                              |
| bpfinal         | Final BP recording  (Sys/Dia):                                                                                                                                 | String        | Sys/Dia -9/-9 = missing                                                                                     |
| htn\_stage      | Severity of HTN during the survey categorized using the 2020 guidelines of the  American Society of Hypertension and the International Society of Hypertension | Numeric       | 0 = Normal (<140/90 mmHg) 1 = Grade 1 (140-159/90-99) 2 = Grade 2 (160-179/100-109) 3 = Grade 3 (>=180/110) |
| treat           | On HTN treatment                                                                                                                                               | Numeric       | 0 = No 1 = Yes                                                                                              |

## Sharing/Access information

This  data is shared as required by the publishing journal and under the terms of the Creative Commons Zero "No rights reserved" data waiver (CC0 1.0 Public domain dedication)([http://creativecommons.org/publicdomain/zero/1.0/](http://creativecommons.org/publicdomain/zero/1.0/))
