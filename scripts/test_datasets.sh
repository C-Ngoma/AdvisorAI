#!/bin/bash

# Dataset 1: Resume Matching
echo "job_id,job_title,required_skills,years_experience,salary_usd" > datacollection/resume_matching_dataset.csv
echo "AI00001,AI Research Scientist,"tableau, pytorch, kubernetes, linux, nlp",9,90376" >> datacollection/resume_matching_dataset.csv
echo "AI00002,AI Software Engineer,"deep learning, aws, mathematics, python, docker",1,61895" >> datacollection/resume_matching_dataset.csv
echo "AI00003,AI Specialist,"kubernetes, deep learning, java, hadoop, nlp",2,152626" >> datacollection/resume_matching_dataset.csv
	
# Dataset 2: AI Job Dataset Merged
echo "job_id,job_title,salary_usd,salary_currency,experience_level,employment_type,company_location,company_size,employee_residence,remote_ratio,required_skills,education_required,years_experience,industry,posting_date,application_deadline,job_description_length,benefits_score,company_name,salary_local" > datacollection/ai_job_dataset_merged.csv
echo "AI00001,AI Research Scientist,90376,USD,SE,CT,China,M,China,50,"Tableau, PyTorch, Kubernetes, Linux, NLP",Bachelor,9,Automotive,2024-10-18,2024-11-07,1076,5.9,Smart Analytics,219728" >> datacollection/ai_job_dataset_merged.csv
echo "AI00002,AI Software Engineer,61895,USD,EN,CT,Canada,M,Ireland,100,"Deep Learning, AWS, Mathematics, Python, Docker",Master,1,Media,2024-11-20,2025-01-11,1268,5.2,TechCorp Inc,25326070" >> datacollection/ai_job_dataset_merged.csv
echo "AI00003,AI Specialist,152626,USD,MI,FL,Switzerland,L,South Korea,0,"Kubernetes, Deep Learning, Java, Hadoop, NLP",Associate,2,Education,2025-03-18,2025-04-07,1974,9.4,Autonomous Tech,109557" >> datacollection/ai_job_dataset_merged.csv

# Dataset 3: AI Job Dataset1
echo "job_id,job_title,salary_usd,salary_currency,experience_level,employment_type,company_location,company_size,employee_residence,remote_ratio,required_skills,education_required,years_experience,industry,posting_date,application_deadline,job_description_length,benefits_score,company_name" > datacollection/ai_job_dataset1_cleaned.csv
echo "AI00001,AI Research Scientist,90376,USD,SE,CT,China,M,China,50,"Tableau, PyTorch, Kubernetes, Linux, NLP",Bachelor,9,Automotive,2024-10-18,2024-11-07,1076,5.9,Smart Analytics" >> datacollection/ai_job_dataset1_cleaned.csv
echo "AI00002,AI Software Engineer,61895,USD,EN,CT,Canada,M,Ireland,100,"Deep Learning, AWS, Mathematics, Python, Docker",Master,1,Media,2024-11-20,2025-01-11,1268,5.2,TechCorp Inc" >> datacollection/ai_job_dataset1_cleaned.csv
echo "AI00003,AI Specialist,152626,USD,MI,FL,Switzerland,L,South Korea,0,"Kubernetes, Deep Learning, Java, Hadoop, NLP",Associate,2,Education,2025-03-18,2025-04-07,1974,9.4,Autonomous Tech" >> datacollection/ai_job_dataset1_cleaned.csv

# Dataset 4: AI Job Dataset2
echo "job_id,job_title,salary_usd,salary_currency,salary_local,experience_level,employment_type,company_location,company_size,employee_residence,remote_ratio,required_skills,education_required,years_experience,industry,posting_date,application_deadline,job_description_length,benefits_score,company_name" > datacollection/ai_job_dataset2_cleaned.csv
echo "AI00001,Data Scientist,219728,USD,219728,EX,PT,Sweden,M,Sweden,0,"Python, Computer Vision, R, Docker",Associate,13,Transportation,2024-09-23,2024-10-31,1132,6.6,TechCorp Inc" >> datacollection/ai_job_dataset2_cleaned.csv
echo "AI00002,Head of AI,230237,JPY,25326070,EX,PT,Japan,L,Japan,50,"Kubernetes, MLOps, Tableau, Python",Bachelor,10,Transportation,2024-07-26,2024-09-12,2299,8.5,Cloud AI Solutions" >> datacollection/ai_job_dataset2_cleaned.csv
echo "AI00003,Data Engineer,128890,EUR,109557,EX,CT,Germany,S,Germany,100,"Spark, Scala, Hadoop, PyTorch, GCP",Bachelor,12,Automotive,2025-01-19,2025-03-28,1329,5.5,Quantum Computing Inc" >> datacollection/ai_job_dataset2_cleaned.csv
