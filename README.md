# lyric-agent
# Probabilistic model trained to generate song lyrics

## Dataset:
https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics

## Abstract:
We propose a probabilistic model that is trained on song lyrics and generates new lyrics given user provided input keywords. This agent is utility-based rather than goal-based because rather than try to achieve a specific outcome, the generated song should maximize fluency/flow. In addition to using log likelihood as a metric, we can employ a public tool like LanguageTool or NLTK to check how grammatically consistent our generated lyrics are. These criteria will be our performance measure; a song maximizes utility if it outputs coherent English and matches user provided keywords.

The environment of this agent is simply just the dataset of songs we provide it and the user provided prompts: this environment is fully observable. The actuator of this agent is its output lyrics. This agent’s sensor is the user’s keyboard, which will feed it keywords and topics to generate the song from (like a genre, starting words, etc.). This agent’s actions are sequential because each word generated will depend on the previously generated words.
Training a probabilistic model would look something like this: given the previous words in the song, and the given keywords (like a given genre or topic), what is the next most likely word? As our model increases in complexity, we can incorporate song structure and create more and more coherent lyrics. We will be using different methods introduced in class to implement our agent, such as Naive Bayes, HMM, and more advanced methods like reinforcement learning.

## PEAS:
*Performance Measure* \
Log Likelihood: How probable the generated lyrics are based on training data.
Fluency & Grammar: Measured using tools like NLTK or LanguageTool to ensure coherent sentence structure.
Keyword Relevance: The song should incorporate user-provided keywords meaningfully.
Song Structure

*Environment* \
Training Dataset: A corpus of song lyrics from Hugging Face.
User Input: Keywords, topics, and optional starting words.
Fully Observable: The model has access to all necessary information to make decisions.

*Actuators*	\
Word-by-word lyric generation.

*Sensors* \
The AI agent takes in user inputs via a keyboard interface
Keywords & Themes: Provided by the user.
Sequential Dependence: Each word prediction depends on previously generated words.

To generate words, we use an n-gram model. This model is a probabalistic Markov model because it generates words based on how likely they are given only the current previous words. It isn't a Hidden Markov Model because there are no hidden states - it makes decisions only on the current observations (the words provided or already generated). This model uses CPTs generated from tokenized lyrics, trying to maximize P(next token | last n tokens). 

[The notebook where all of our data was cleaned, models were trained, and over/underfitting was calculated is here](milestone2.ipynb)

## Dataset and preprocessing:
In the dataset we chose, each row contains information on a single song - it's lyrics, artist, genre, language, features, etc. For the purposes of this first model, most of these columns are irrelevant. We are mainly concerned with the lyrics and the "tag" column, which contains the genre (rap, pop) of the song. Before we calculated CPTs, we filtered songs by "language," and kept only songs in English. After performing this filter, we had *3,374,198* rows in our dataset.

The cleaned datasets are way too big to upload into the repository, but we include a sample .csv of 100 songs after preprocessing and before tokenization [here](genius_lyrics_small.csv). This .csv wasn't used in training, and is just an example of the cleaning we did on the much larger dataset.

Lyrics were tokenized by splitting the lyrics, including punctuation, and then trained. We applied laplace smooth to avoid 0 probability n-grams, which was necessary to get meaningful log-likliehoods in the next step.

## Overfitting / Underfitting

We think lower values of n will underfit and the higher values of n will overfit. Lower values of n won't be enough to capture all the patterns in our dataset of songs. Higher values will create good training data, but will perform bad on test data because unseen song lyrics will be viewed as unlikely even if they capture general patterns

Ideally, to test for overfitting and underfitting, we'd train the model using different values of n (bigram, trigram, etc). The problem with this is that because the dataset is so huge, and since computation because more and more costly for higher values of n, it becomes unfeasible for us to train so many models. 

We randomly sampled 100,000 songs and tested models of n = [2, 3, 4, 5].

Here are some sample lyrics generated:

`n = 2, input = ['hi']`
hi, i know that i know that i know that i know that i know that

Here, we loop between "i know that," which is to be expected with a low n.

`n = 5, input = ['hi', ',', 'you', 'are']`
hi, you are of the lord 2x my god has made his favour, to be manifested upon

These lyrics are actually pretty coherent, and look like they come from a Christian song.

`n = 6, input = ['hi', ',', 'you', 'are', 'my']`
hi, you are my, beautiful mess baby you are my beautiful, beautiful mess baby you are my

These lyrics also start to repeat, but the lyrics at least form a coherent message.

Then, we used sample lyrics from hit pop songs and calculated the negative log likelihood of each set of lyrics.

Below is the results of our different n-gram models on the lyrics `['now', 'and', 'then', ',', 'i', 'think', 'of', 'all', 'the', 'times', 'you', 'screwed', 'me', 'over']` from Gotye's "Somebody That I Used To Know"
![](gotye_results.png)

Here are the results of our different n-gram models on a simpler test sequence `['i', 'wanna', 'dance', 'with', 'somebody']` from Whitney Houston's "I Wanna Dance With Somebody"
![](whitney_houston_results.png)

In general, we found that values of smaller n had the best log-likelihoods (closest to 0). In particular, it seems like our bigram model performed the best on those specific lyrics. As n increased, our log likelihood became further away from 0, indicating signs of overfitting since we were performing worse on the "test" data. Given our model, this makes sense, however this doesn't necessarily mean that lower n-grams are the best for our purpose of generating lyrics. Log-likelihood might not be the best metric for evaluating overfitting and underfitting, and we might need to experiment and research more complex methods, like perplexity. However, given the nature of n-gram, it is likely that we underfit on the lower values of n given that not much context is retained with lower values of n, however, as n gets larger, our model overfits since it is trained on relatively long specific sequences of words.

## Conclusion section
The first model has a lot of pitfalls. The model frequently runs into cycles, probably because of the way songs are structured. Choruses and repetitive parts of song lyrics will inflate the probabilities in the CPT table, making it harder to generate interesting and unique lyrics.

In future models, we hope to find ways to let users input keywords and genres rather than starting prompts, and to give songs more structure
