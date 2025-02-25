"""
Objectives: 
• Build a basic search system.

Tasks:
Modify the code from HW01 in order to perform the processing steps as shown in the figure below. 

```mermaid
graph TD
   %% Document Processing Path
   ReadDocs[📄 Read Documents] --> Tokenize[✂️ Tokenize<br/>hello world → hello,world]
   Tokenize --> Lowercase[⬇️ Convert to Lowercase<br/>Hello → hello]
   Lowercase --> RemovePunc[❌ Remove Punctuation<br/>hello! → hello]
   RemovePunc --> RemoveStop[🚫 Remove Stopwords<br/>the,and,is → removed]
   RemoveStop --> Stemming[🌱 Stemming<br/>running → run]
   Stemming --> CreateIndex[🏗️ Create Index]
   CreateIndex --> Index[🗂️ Index<br/>term1 → 1→4→7<br/>term2 → 2→5→8]

   %% Query Processing Path
   Query[❓ Query] --> QueryProc[🔍 Query Processing<br/>and Expansion]
   
   %% Convergence
   QueryProc --> Retrieval[📊 Retrieval]
   Index --> Retrieval
   
   %% Results
   Retrieval --> Results[📝 Results<br/>doc29<br/>doc32<br/>doc57]
```

Write new code to load the inverted index from question 1 (homework 1) and answer user queries as follows

• Load the user queries from a file. 
• Each query will have a maximum four words
• For each query provide the answer using all possible combination. if the query is: q= A B C, the answer should be as follows: 
```
================== User Query 1: .............. ================= 
============= Results for: A and B and C =================
file 12
file 59 
... 
==================================================
================ Results for: A and B or C ===================
file 12
file 59
... 
==================================================
=============== Results for: A or B and C ==================== 
file 12 
file 59 
... 
==================================================
=============== Results for: A or B or C ===================== 
file 12 
file 59
... 
================================================
================== User Query 2: .............. ===============
```

"""