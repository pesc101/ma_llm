## First Try to generate questions
Based on the assistant's response, please predict the instruction from the user for the provided answer from the ASSISTANT and write it as a question. This is a chat between a curious user and a helpful programming assistant with artificial intelligence.

ASSISTANT: Certainly! You can create a function in Python that solves this:


## Second Try
Based on the assistant's response, predict the user's instruction for the given response from the ASSISTANT and write it as a complex question showing that parts of the code have been understood. This is a chat between a curious user and a helpful programming assistant with artificial intelligence.

ASSISTANT: Certainly! You can create a function in Python that solves this:


## Procedure
1. Create Unlabbeled Data -> No Model needed! (TODO: Refactor Code to class Based approach)
2. Create Augmentation with unlabbeled Data (TeacherDataSet) -> Model needed! (Done!)
3. Create Augmentation with Teacher Dataset to create Questions to the generated Outputs from 2 (QuestionDataSet) -> Model needed! (Done!)
4. Create DataSet with Question - Answer Pairs from Step 2 and 3 (CurationCreateDataSet) -> No Model neeeded! (Done!)
5. Create a Score for each Question - Answer Pair from Step 4 (CurationEvaluationDataSet) -> Model needed! (Done!)
6. Filter all Q&A Pairs with a low Score (CurationFilter) -> No model needed! (Done!)


## How should the data look like?
1. Meta Data + Code
2. Meta Data (Question included) + Code + Full Prompt Teacher + Response Teacher
3. Meta Data (Question included) + Code + Full Prompt Teacher + Response Teacher + Full Prompt Question + Response Teacher
4. "" + Curation Full Prompt
5. "" + Curation Evaluation Response
6. Nothing Changes
(Done!)

