# lyric-agent
Probabilistic model trained to generate song lyrics

Dataset:
https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics

Abstract:
We propose a probabilistic model that is trained on song lyrics and generates new lyrics given user provided input keywords. This agent is utility-based rather than goal-based because rather than try to achieve a specific outcome, the generated song should maximize fluency/flow. In addition to using likelihood as a metric, we can employ a public tool like LanguageTool or NLTK to check how grammatically consistent our generated lyrics are. These criteria will be our performance measure; a song maximizes utility if it outputs coherent English and matches user provided keywords.

The environment of this agent is simply just the dataset of songs we provide it and the user provided prompts: this environment is fully observable. The actuator of this agent is its output lyrics. This agent’s sensor is the user’s keyboard, which will feed it keywords and topics to generate the song from (like a genre, starting words, etc.). This agent’s actions are sequential because each word generated will depend on the previously generated words.
Training a probabilistic model would look something like this: given the previous words in the song, and the given keywords (like a given genre or topic), what is the next most likely word? As our model increases in complexity, we can incorporate song structure and create more and more coherent lyrics. We will be using different methods introduced in class to implement our agent, such as Naive Bayes, HMM, and more advanced methods like reinforcement learning.

PEAS:
Performance Measure
Likelihood: How probable the generated lyrics are based on training data.
Fluency & Grammar: Measured using tools like NLTK or LanguageTool to ensure coherent sentence structure.
Keyword Relevance: The song should incorporate user-provided keywords meaningfully.
Song Structure

Environment
Training Dataset: A corpus of song lyrics from Hugging Face.
User Input: Keywords, topics, and optional starting words.
Fully Observable: The model has access to all necessary information to make decisions.

Actuators	
Word-by-word lyric generation.

Sensors
The AI agent takes in user inputs via a keyboard interface
Keywords & Themes: Provided by the user.
Sequential Dependence: Each word prediction depends on previously generated words.


To generate words, we use an n-gram model. This model is a probabalistic Markov model because it generates words based on how likely they are given only the current previous words. It isn't a Hidden Markov Model because there are no hidden states - it makes decisions only on the current observations (the words provided or already generated). This model uses CPTs generated from tokenized lyrics, trying to maximize P(next token | last n tokens). 

Dataset and preprocessing:
In the dataset we chose, each row contains information on a single song - it's lyrics, artist, genre, language, features, etc. For the purposes of this first model, most of these columns are irrelevant. We are mainly concerned with the lyrics and the "tag" column, which contains the genre (rap, pop) of the song. Before we calculated CPTs, we filtered songs by "language," and kept only songs in English. After performing this filter, we had 3,374,198 rows in our dataset.

Lyrics were tokenized by splitting the lyrics, including punctuation. For our initial model, we trained a trigram (n=3) model on only pop songs.

Overfitting / Underfitting
