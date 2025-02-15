# lyric-agent
Probabilistic model trained to generate song lyrics

Dataset:
https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics

Abstract:
We propose a probabilistic model that is trained on song lyrics and generates new lyrics given user provided input keywords. This agent is utility-based rather than goal-based because rather than try to achieve a specific outcome, the generated song should maximize fluency/flow. In addition to using likelihood as a metric, we can employ a public tool like LanguageTool or NLTK to check how grammatically consistent our generated lyrics are. These criteria will be our performance measure; a song maximizes utility if it outputs coherent English and matches user provided keywords.

The environment of this agent is simply just the dataset of songs we provide it and the user provided prompts: this environment is fully observable. The actuator of this agent is its output lyrics. This agent’s sensor is the user’s keyboard, which will feed it keywords and topics to generate the song from (like a genre, starting words, etc.). This agent’s actions are sequential because each word generated will depend on the previously generated words.
Training a probabilistic model would look something like this: given the previous words in the song, and the given keywords (like a given genre or topic), what is the next most likely word? As our model increases in complexity, we can incorporate song structure and create more and more coherent lyrics. We will be using different methods introduced in class to implement our agent, such as Naive Bayes, HMM, and more advanced methods like reinforcement learning.
