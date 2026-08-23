# Capstone Conclusion

I learned a lot in this project about RAG, LLM applications, agents, and deploying apps into production. A lot of what I originally thought was "magical" about RAG and AI agents just turns out to be simple python in harnesses with effective strategy. This project gave me a lot of direction about where I want to go as a data scientist and lifelong CS/SWE learner, and I want to get more experimental with agents as my foundation gets stronger. 

## What Worked Well

Getting a functional RAG pipeline, building vector databases, and designing a Streamlit frontend were pretty easy. LoRA training and learning concepts was also not as hard as I thought it would be.

## What Was Hard

Getting the pipeline to work was easy. The hard part was:

- Determinsitic semantic retrieval
- Categorizing prompts correctly
- Making it not break in deployment
- Testing the full pipeline
- Making the responses quicker
- Disorganized repo structure
- Removing previously implemented changes

## Obstacles

Here are three things I would do differently if I could start over:

### 1. CI

This repo needed an effective test harness. A quick way to build the instance and run it through test questions- only needing a few small changes as we add more features. GitHub Actions would work well for this, but I would put python files in the repo that allow me to get quick answer responses and compare them to older responses- something that I can keep using as I add more features.

### 2. App first, design last

In other circumstances, the absolute last thing I would build would be the streamlit frontend and the docker/kubernetes integration- focusing on finding and removing all possible backend bugs should have been a first priority and I would not do the project in the same order. A working backend Kubernetes instances should come before a working frontend Kubernetes instance.

### 3. Real data

Using synthetic, AI-generated data meant there was no real "ground truth." I had to just accept whatever the AI put in those documents as the standard for what is true and false about plan information- which made it hard to verify correct answers, especially with AI models' non-determinisitic nature. Next time I would use either: more specific generation that adheres to listed policies.

### 4. Don't Delete/Overwrite Files

I wouldn't ever delete file versions when experimenting with one version of a file. After integrating fine tuning, MCP servers, and different prompt types, I wouldn't delete older versions in case it turns out to be better.