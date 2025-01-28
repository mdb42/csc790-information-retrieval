# CSC790 Information Retrieval

## Project Journal

### January 24. 2025, 7:11 AM

- Initial commit as CSC790 Indexer project

- Toyed around with a handful of project ideas, but it was hinted in class that this would be first, and I'll try to keep it simple for the sake of the assignment.

- Created auxiliary projects for web crawling (InfoQuest) and retrieval augmented generation (Lucid), which will both need a document indexer. The other projects are using SQLite for tracking documents and metadata, and for Lucid, I'm thinking actually like a Knowledge docked widget on a document editor and chat gui, where you can drag and drop files into the Knowledge dock and it will map to whatever filepath, use the Microsoft "Anything-to-Markdown" library and index whatever is currently in knowledge, use it for document retrieval in LLM generations, save knowledge bases on the fly, incorporate the crawler we'll make later in class, and also probably an in-built web browser using QWebEngineView for navigating and adding direct web sources.

### January 25, 2025, 12:53 AM

- First prototypes for document collection, index building, and search

- Looks like my naive approach is going to require an nltk tokenizer called 'punkt_tab' to be present. I'm just checking for it and downloading as necessary. This is apparently fine for class assignments, but I worry that it might be a problem for deployment in real-world projects - potentially setting off antivirus warnings if packaged into a single executable. 

- For data, I pulled a list of public domain books from Project Gutenberg, using a test_data.py file that I borrowed from a past project - the the simple-gpt demo in the ml-demos repo on my github. I got this overall implementation running right before class and was running two versions when testing on my laptop during class with two different book lists (one just a few novels, the other being nearly a hundred), and while there wasn't much difference in the smaller dataset, it's kind of wild how much faster my desktop performed comparably for the larger dataset. The large collection indexing took only moments on the desktop, but ended up taking over two hours on my laptop - which is crazy! I don't think there should be that much disparity between them. I'm not sure if it's the processor or the RAM or if I'm a moron and hadn't notice some other variable or function I'd changed. I'll profile this later to make sure the final version doesn't have a catastrophic goof in there.

- The index_builder.py is pretty much a direct replication of the workflow as presented in the lecture presentation, using the nltk examples from the end almost verbatim. I just tossed a handful of stopwords at it that occurred to me at the time, but I'll look at other options later. Porter stemmer is all we talked about in class, but I also saw a Lancaster stemmer and a Snowball stemmer (Improved Porter? Improved how?), so I'll look into those as well.

- Adding documents, I just started around this basic outlined approach:
  1) Tokenize
  2) Normalize & filter:
     a) Case-fold
     b) Remove stop words
     c) Stem
  3) Insert into index

I seem to recall in the lecture that we had spoken of removing the stopwords before doing anything else, as it would be more efficient to avoid them in the other normalization steps. I'm a terrible listener though, and so I may have misheard. I'm thinking though that to remove stopwords before tokenization, you'd need to do some form of string matching/replacement on the raw text, essentially scanning through the text multiple times looking for each stopword, or using a regex... but rather NLTK can do the tokenization on a single pass, then if using a Python set, it's just an O(1) operation to perform a lookup for stopwords, and we can just hit those up in a single loop while performing the other normalization steps.

- Query parsing took more effort than anything, because I'm a noob. I'm handling grouping by preserving parantheses and converting the query into a a set expression, using eval then for querying set containment. It's kind of fragile still though, and it'll break right now if you try two terms without an operator, so I should probably set it up to infer an "and" in the absence of an operator.

- Python string manipulation is honestly super hard for me. I tend to only do it once in a blue moon, wrap it up in a function, and then never touch it again. Six months later, I need to write another for some reason, and it feels like learning all the available string methods all over again. I observed a lot of examples here before I got this working as intended, and I'm sure there is more I can do still.

### January 27, 2025, 10:33 AM

- We received the actual assignment specification just now in class about half an hour ago. To accommodate it, I'm renaming the repo as csc790-information-retrieval and will use it for all the future homework projects. The document indexer is now set up in an HW01 directory. I added the assignment pdf to the directory, stopwords.txt, and imported the documents. It's not actually using the stopwords.txt file yet though. The main.py file is a bit of a kludge right now (importing the InvertedIndex class and build_invered_index method to orchestrate them there), but this was the most expedient way to get it working again while in class. I'll clean it up.

- EXTRA CREDIT: 50 points for parallelization/multiprocressing. I removed the assignment pdf file from the project since the professor said he would replace it with the official inclusion of the extra credit.

I'm still waiting on the precise specification for the extra credit, but my initial thoughts are that I can attack this on a few different fronts:

- Loading documents in parallel (IO-bound)
- Tokenizing and normalizing in parallel (CPU-bound)
- Building the index in parallel (CPU-bound)

After a bit of searching, I'll check out first the ProcessPoolExecutor class from the concurrent.futures module.
Alternatively, I could use the multiprocessing module. I don't know enough yet to decide.

### January 27, 2025, 11:39 AM

- The project is now fully adapted to meet the minimal requirements of the assignment, consolidated into a single file.

- Banner now displays the course and my name. I don't know if I appreciate my full name being displayed on a public repo in all of my homework assignments, but I guess it's fine for now. I will probably feed it from an ignored config file later, and just turn the config file in with the project. I will check with the professor if that's acceptable.

- Now loading the stopwords from the stopwords.txt file, tracking frequencies, reporting sizes in megabytes, and displaying the top 10 most common words in the index. I think that covers all the basics for the assignment.

- I'm using the Counter class from collections, which might seem like overkill, but it gives us the most_common(n) method out of the box and has an update() method which will be helpfully simple in a parallel implementation later.

### Current State of the Project

- So, initially, I tried using ProcessPoolExecuter, and I didn't understand why it was failing. I was trying to pass a nested function to it, and I didn't realize that it was causing a pickling error. I gutted that out then and tried using the multiprocessing module instead, getting it to work, but only after also needing to refactor to similarly avoid the nested function pickling. That being the case, I don't actually know which is better between the two approaches, but this multiprocessing implementation appears to be working at least.

- Current implementation only parallelizes the document processing and indexing phase, while document loading and final merging remain sequential. There's room for improvement here.

- Interestingly, since this multithreaded version will assign each worker their own instance of an InvertedIndex object, they were all separately checking for that nltk punkt_tab file, spamming the terminal as all the workers from the different cores checked to see if the tokenizer was there. So, I thought, "Hey, let's put it at the module level with the imports." But that too was causing the workers to all check for the tokenizer. Evidently, when creating a new process using multiprocessing, each worker executes the full module from the top. Putting it just in the main method before the rest of the orchestration avoided the multiple calls.

- Still need to investigate the performance disparity between desktop and laptop from the initial implementation. I'm going to go bottleneck hunting.


