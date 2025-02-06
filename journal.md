# CSC790 Information Retrieval

## Project Journal

### January 24. 2025, 7:11 AM

- Initial commit as CSC790 Indexer project

- Toyed around with a handful of project ideas, but it was hinted in class that this would be first, and I'll try to keep it simple for the sake of the assignment.

- Created auxiliary projects for web crawling (InfoQuest) and retrieval augmented generation (Lucid), which will both need a document indexer. The other projects are using SQLite for tracking documents and metadata, and for Lucid, I'm thinking actually like a Knowledge docked widget on a document editor and chat gui, where you can drag and drop files into the Knowledge dock and it will map to whatever filepath, use the Microsoft "Anything-to-Markdown" library and index whatever is currently in knowledge, use it for document retrieval in LLM generations, save knowledge bases on the fly, incorporate the crawler we'll make later in class, and also probably an in-built web browser using QWebEngineView for navigating and adding direct web sources.

### January 25, 2025, 12:53 AM

- First prototypes for document collection, index building, and search

- Looks like my naive approach is going to require an nltk tokenizer called 'punkt_tab' to be present. I'm just checking for it and downloading as necessary. This is apparently fine for class assignments, but I worry that it might be a problem for deployment in real-world projects - potentially setting off antivirus warnings if packaged into a single executable. 

- For data, I pulled a list of public domain books from Project Gutenberg, using a test_data.py file that I borrowed from a past project - the the simple-gpt demo in the ml-demos repo on my github. I got this overall implementation running right before class and was running two versions when testing on my laptop during class with two different book lists (one just a few novels, the other being nearly a hundred), and while there wasn't much difference in the smaller dataset, it's kind of wild how much faster my desktop performed comparably for the larger dataset. The large collection indexing took only moments on the desktop, but ended up taking over two hours on my laptop - which is crazy! I don't think there should be that much disparity between them. I'm not sure if it's the processor or the RAM or if I'm a moron and hadn't noticed some other variable or function I'd changed. I'll profile this later to make sure the final version doesn't have a catastrophic goof in there.

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

### January 28, 2025, 8:20 AM

- So, initially, I tried using ProcessPoolExecuter, and I didn't understand why it was failing. I was trying to pass a nested function to it, and I didn't realize that it was causing a pickling error. I gutted that out then and tried using the multiprocessing module instead, getting it to work, but only after also needing to refactor to similarly avoid the nested function pickling. That being the case, I don't actually know which is better between the two approaches, but this multiprocessing implementation appears to be working at least.

- Current implementation only parallelizes the document processing and indexing phase, while document loading and final merging remain sequential. There's room for improvement here.

- Interestingly, since this multithreaded version will assign each worker their own instance of an InvertedIndex object, they were all separately checking for that nltk punkt_tab file, spamming the terminal as all the workers from the different cores checked to see if the tokenizer was there. So, I thought, "Hey, let's put it at the module level with the imports." But that too was causing the workers to all check for the tokenizer. Evidently, when creating a new process using multiprocessing, each worker executes the full module from the top. Putting it just in the main method before the rest of the orchestration avoided the multiple calls.

- Still need to investigate the performance disparity between desktop and laptop from the initial implementation.

### January 28, 2025, 11:49 AM

- Implemented temporary profiling. This was way more complex than anticipated due to the distributed processing - each worker process needed to track its own timing metrics, which then had to be aggregated and compared against wall clock time. Not perfect! But it gave me some idea of what is going on.

It would seem at least:
- Tokenization is the primary bottleneck
- Manual testing with different core counts showed total tokenization time appears to scale with the square root of the core count
- Document loading and index merging are relatively efficient in comparison

- Otherwise, I created a main.py entry point wrapper again to avoid dancing between directories and to better manage the parameters.

### January 28, 2025, 11:43 PM

- I tried batching the tokenization and had like zero improvement. I guess that makes sense, since it's already broken up and distributed among the workers by document, and just shuffling the data around isn't going to help. If we had more variety in document sizes, I imagined it might have allowed for some load balancing among the workers, but ultimately the process overhead just isn't that big to begin with.

- So, time to think about the query process, which is basically still in its original form from the initial prototype implementation before the assignment even started. If this was the real world, I would actually just use a parsing library and not devote cognitive effort towards this. We have a new lecture now though with pseudocode for finding the intersection of two postings...

It shows:
- Walk through two posting lists simultaneously
- Compare document IDs, and when they match, add to the answer
- When they don't match, advance the pointer of the list with the smaller document ID

I don't like it though. There's nothing wrong with it on its face, and I imagine something similar is already happening under the hood when using Python sets. The key issue is that Python sets are implemented in C, and trying to reinvent the wheel here is only going to slow things down comparably. I'm going to keep playing around and see if I can implement logical NOTs and more robust grouping while continuing to use Python sets.

- So, set theory... To handle NOTs, we need a universal set of document IDs. When we see a NOT operator, we can interpret it as the universal set minus some set of terms, essentially just checking for intersection with the complement. This seems to work as expected, but I've surely just not yet discovered all the ways to break it.

- I've adapted the query logic into a while loop and just extended the chain of else-ifs in the query parsing to handle the new operators and special cases. I actually still plan to make this recursive, and though not fully implemented yet, I did extract out the logic for gathering within parantheses and for subexpression-building into separate helper functions to make that easier when I get to it. I'm passing around some parameters too that I don't use yet, but I will probably need them later.

- Lastly, I put in a quick check beforehand for if an operator is missing between two terms, and I am inserting an AND to handle it like Google does.

### January 29, 2025, 2:52 AM

I created a smaller set of documents so I could more easily test the query logic, and I set up a query list so I could just punch through all the query variations each run to see if they were working as expected. This revealed a few issues:

- Mid-expression NOTs still needed to have an implied preceding AND operator. This was easily fixed by just inserting it in the NOT handling logic.
- NOTs that were preceding grouped ORs were coming up malformed. Really, I had two processes for expression interpretations occurring - a more rigorous one at the top scope, but a hacky one for the subexpressions. I needed to extract that logic out and just use it for both cases, so I created the parse_tokens_to_expression() function and call it wherever needed.
- I was also getting an off-by-one error in the NOT handling logic in some cases for the paren_count, easily fixed though by just adjusting the conditional for when returning how many tokens were consumed.

After fixing these issues, I stress-tested the query parser with increasingly complex cases: deeply nested parentheses with mixed operators, case sensitivity and irregular spacing, malformed queries with extra punctuation and operators, complex combinations of NOTs and parenthetical grouping. It didn't explode.

### February 1, 2025, 6:16 AM

Okay, I shouldn't be trusted to write a query parser. I mean, using eval on user-generated input is practically inviting arbitrary code execution, and while I felt like I was santizing everything, it's just a bad idea to leave it as even a hypothetical vulnerability. It's not even part of the actual assignment and just a relic from my own experiments before the assignment was even given. That being the case, I've extracted it out to a separate file to unclutter the graded portion of the assignment and rebuilt it to use a simpler implementation for demonstration purposes, starting from an example for how a query parser might be implemented by a sane programmer. It's not as versatile as the eval version, but it's safer. I can always improve this approach later, but at least it's getting off on the right foot. I ended up removing the query_list option too as it felt like clutter that distracted from the graded parts of the assignment. I may reimplement it in the query_parser.py file later.

I decided to go back and look at my approach to parallelization too, since I never did a get a good chance to compare the mp.Pool approach with a working implementation of ProcessPoolExecuter. I noted how the mp.Pool approach was forcing serialization of the entire object per worker, and using a large enough dataset, it was actually going to cause worse performance than the baseline sequential implementation. I switched it out for ProcessPoolExecuter, and it's functioning as expected. I'll do some measurements next. I will try another implementaiton later with dynamic chunking too, but I doubt it will affect things greatly with the given dataset since the documents are all roughly the same size.

This morning's quiz taught me something. I am a moron when it comes to to calculating the memory footprint of whatever I'm working on. I come home then, looked at my code, and I recognized I was totally miscalculating the memory usage of the index here as well. I was only getting the size of the defaultdict object, not the actual size of the strings and integers stored in it. The contained sets and other structures were not being accounted for at all. I was reporting sizes as low as 0.3 MB, which I now realize after traversing the index and summing the sizes of the contained objects, should be reported closer to 6.14 MB. I'd thought also to just dump the pickle file and check the size of that, but I realized that the pickle file is compressed and not a good representation of the actual memory usage.

Otherwise, I've made some structural changes to suit the assignment submission, bringing the main method back into the indexer's file and removing the main.py wrapper. Since the assignment doesn't actually mention anything about querying, I don't really know how the professor intends to inspect the results. The index is kept as a pickle file, which he can't directly inspect, so I added a method to export the index and its metadata to json. I also added some argument parsing to the main method to allow for customization at the command line. I adjusted a few container definitions also that simplified some of the checks for object existence and allowed more efficient loading as sets, and I also added a few more status printouts and improved error handling.

### February 5, 2024, 4:04 AM

- Major refactoring of the project structure today, though most of it was just shuffling existing parts around for better encapsulation and streamlining the initialization process. Instead of orchestrating the index creation from outside the class, the constructor now manages the whole initialization workflow.Moved stopwords loading into a static method on the InvertedIndex class and fixed the chunk processing to now operate within the class as well. That was the only part that really required me to stop and think. I assume this is more efficient, but I am too tired to test it right now.

- Wrote a new memory reporting method, providing a breakdown of memory consumption across all the components of the index. I mostly only did this because without seeing the full componentwise breakdown, one's naive expectation might just be that the dumped pickle size alone would suffice. It's like, yeah, I know my method is convoluted, but look at the numbers and you'll see why.

- Improved error handling in the parallel processing implementation. Rather than letting errors in individual documents potentially disrupt the entire indexing process, I'm now collecting errors per chunk and reporting them after successful index construction.

- The project is due tomorrow night, and so I don't have a lot of time left to make improvements. Some last minute changes I'll attempt tomorrow will be to try an alternative implementation of the index that uses a dictionary of dictionaries, accounting for term_frequency per doc_id, since that's really what we'd need for TF-IDF or BM25 scoring. Besides, since removing the eval-based query parser, the original rationale for using sets no longer applies. Otherwise, I'll probably adapt all the print statements to use a logging interface, add some unit tests, and draft new commenting and readme instructions.

### Current State of the Project

- Assignment submitted.
- Note to self: When your bright idea to perform a last-minute fancy-pants refactoring comes together in one draft without a sign of trouble, don't trust it. Test it thoroughly before you spend the entire day extensively commenting a totally borked implementation. What beautiful comments they were too! I elaborated error handling throughout too! All lost to the ether now. It's like a construction crew spending all night and day demolishing and rebuilding a house, only to realize they were at the wrong address.
- Realizing my mistake - which is too dumb to recount - literally in the eleventh hour, I rolled everything back to this morning's state, rushed through minimal commenting, zipped it up and hit send - barely devoting ten minutes to the final submission's documentation. It hurts! Oh well! I linked the journal in the readme, so if you're reading this now, professor, then... ooga booga! I'll do better next time.
- On the bright side, not bound now by the prescriptions of any assignment, I can return to working on Lucid. I will probably tear this indexer apart, extract the document processing into a separate class, otherwise compartmentalizing everything along the boundaries defined in the IIR textbook's workflow diagram.
