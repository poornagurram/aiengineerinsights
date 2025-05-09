interface Project {
  title: string
  description: string
  href?: string
  imgSrc?: string
}

const projectsData: Project[] = [
  // CORE LIBRARIES & FRAMEWORKS
  {
    title: 'TensorFlow',
    description: 'Google’s scalable open-source library for deep learning and machine learning. Used in research and production at scale.',
    imgSrc: 'https://www.tensorflow.org/images/tf_logo_social.png',
    href: 'https://github.com/tensorflow/tensorflow',
  },
  {
    title: 'PyTorch',
    description: 'Meta’s flexible deep learning framework, now the default for research and production in vision, NLP, and generative AI.',
    imgSrc: 'https://pytorch.org/assets/images/pytorch-logo.png',
    href: 'https://github.com/pytorch/pytorch',
  },
  {
    title: 'Keras',
    description: 'High-level neural networks API, user-friendly and modular. Runs on top of TensorFlow.',
    imgSrc: 'https://keras.io/img/logo.png',
    href: 'https://github.com/keras-team/keras',
  },
  {
    title: 'HuggingFace Transformers',
    description: 'The essential library for state-of-the-art transformer models (BERT, GPT, T5, etc.) for NLP, vision, and more.',
    imgSrc: 'https://huggingface.co/front/assets/huggingface_logo-noborder.svg',
    href: 'https://github.com/huggingface/transformers',
  },
  {
    title: 'scikit-learn',
    description: 'The gold standard for classical machine learning in Python: regression, classification, clustering, and more.',
    imgSrc: 'https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png',
    href: 'https://github.com/scikit-learn/scikit-learn',
  },
  {
    title: 'XGBoost',
    description: 'The most popular gradient boosting library for tabular data, competitions, and production ML.',
    imgSrc: 'https://avatars.githubusercontent.com/u/21003710?s=200&v=4',
    href: 'https://github.com/dmlc/xgboost',
  },
  {
    title: 'LightGBM',
    description: 'A fast, distributed, high-performance gradient boosting framework by Microsoft.',
    imgSrc: 'https://lightgbm.readthedocs.io/en/latest/_static/logo.svg',
    href: 'https://github.com/microsoft/LightGBM',
  },
  {
    title: 'CatBoost',
    description: 'Yandex’s gradient boosting library with great support for categorical features.',
    imgSrc: 'https://avatars.githubusercontent.com/u/21003710?s=200&v=4', // No official logo, using XGBoost as placeholder
    href: 'https://github.com/catboost/catboost',
  },
  {
    title: 'spaCy',
    description: 'Industrial-strength NLP library for fast, production-ready pipelines.',
    imgSrc: 'https://spacy.io/static/img/spacy-logo.svg',
    href: 'https://github.com/explosion/spaCy',
  },
  {
    title: 'OpenCV',
    description: 'The leading library for computer vision, image processing, and video analysis.',
    imgSrc: 'https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_no_text.png',
    href: 'https://github.com/opencv/opencv',
  },
  {
    title: 'Ultralytics YOLOv8',
    description: 'State-of-the-art real-time object detection and segmentation models.',
    imgSrc: 'https://docs.ultralytics.com/logo.png',
    href: 'https://github.com/ultralytics/ultralytics',
  },
  {
    title: 'Diffusers',
    description: 'HuggingFace’s library for diffusion models (Stable Diffusion, DALL-E, etc.) for image and audio generation.',
    imgSrc: 'https://huggingface.co/front/assets/huggingface_logo-noborder.svg',
    href: 'https://github.com/huggingface/diffusers',
  },

  // AGENT FRAMEWORKS & LLM ECOSYSTEM
  {
    title: 'LangChain',
    description: 'The most popular framework for building LLM-powered applications and AI agents.',
    imgSrc: 'https://www.langchain.com/favicon.ico',
    href: 'https://github.com/langchain-ai/langchain',
  },
  {
    title: 'LlamaIndex',
    description: 'Framework for building LLM applications with custom data, retrieval-augmented generation, and more.',
    imgSrc: 'https://avatars.githubusercontent.com/u/113019107?s=200&v=4',
    href: 'https://github.com/jerryjliu/llama_index',
  },
  {
    title: 'Haystack',
    description: 'An open-source framework for building production-ready search systems and RAG pipelines with LLMs.',
    imgSrc: 'https://haystack.deepset.ai/_next/static/media/logo.2c3b2b5b.svg',
    href: 'https://github.com/deepset-ai/haystack',
  },
  {
    title: 'Auto-GPT',
    description: 'The original open-source autonomous GPT agent. Inspiring a wave of agentic AI development.',
    imgSrc: 'https://avatars.githubusercontent.com/u/130668900?s=200&v=4',
    href: 'https://github.com/Significant-Gravitas/Auto-GPT',
  },
  {
    title: 'CrewAI',
    description: 'A modern framework for orchestrating multiple collaborative AI agents.',
    imgSrc: 'https://www.crewai.com/favicon.ico',
    href: 'https://github.com/joaomdmoura/crewAI',
  },
  {
    title: 'Open Interpreter',
    description: 'Run code with natural language instructions, locally or in the cloud. Great for building coding agents.',
    imgSrc: 'https://avatars.githubusercontent.com/u/137642843?s=200&v=4',
    href: 'https://github.com/OpenInterpreter/open-interpreter',
  },
  {
    title: 'GPT Engineer',
    description: 'Turn natural language prompts into codebases. A trending project for AI-powered software engineering.',
    imgSrc: 'https://ai.google.dev/static/images/gemini-favicon.png',
    href: 'https://github.com/AntonOsika/gpt-engineer',
  },
  {
    title: 'MetaGPT',
    description: 'Multi-agent framework for collaborative AI agents that can plan, code, and execute complex tasks.',
    imgSrc: 'https://avatars.githubusercontent.com/u/136297844?s=200&v=4',
    href: 'https://github.com/geekan/MetaGPT',
  },
  {
    title: 'BabyAGI',
    description: 'A minimal, open-source autonomous AI agent framework for experimentation.',
    imgSrc: 'https://babyagi.org/favicon.ico',
    href: 'https://github.com/yoheinakajima/babyagi',
  },
  {
    title: 'SuperAGI',
    description: 'A robust, production-ready open-source AGI research platform for building and deploying autonomous agents.',
    imgSrc: 'https://superagi.com/favicon.ico',
    href: 'https://github.com/TransformerOptimus/SuperAGI',
  },
  {
    title: 'Camel-AI',
    description: 'A framework for role-playing AI agents that can interact and solve tasks together.',
    imgSrc: 'https://avatars.githubusercontent.com/u/130668900?s=200&v=4',
    href: 'https://github.com/camel-ai/camel',
  },
  {
    title: 'OpenAI Cookbook',
    description: 'Official OpenAI repository with practical guides, code, and examples for using GPT models and APIs.',
    imgSrc: 'https://openai.com/favicon.ico',
    href: 'https://github.com/openai/openai-cookbook',
  },

  // DATA-CENTRIC AI & EVALUATION
  {
    title: 'Cleanlab',
    description: 'Finds and fixes label errors in datasets, improving ML model performance.',
    imgSrc: 'https://cleanlab.ai/favicon.ico',
    href: 'https://github.com/cleanlab/cleanlab',
  },
  {
    title: 'Evidently',
    description: 'Open-source tool for monitoring data and ML model quality in production.',
    imgSrc: 'https://evidentlyai.com/favicon.ico',
    href: 'https://github.com/evidentlyai/evidently',
  },
  {
    title: 'Giskard',
    description: 'Automated testing and evaluation platform for ML models, focusing on robustness and fairness.',
    imgSrc: 'https://giskard.ai/favicon.ico',
    href: 'https://github.com/Giskard-AI/giskard',
  },
  {
    title: 'Great Expectations',
    description: 'Data validation and pipeline testing framework for ensuring data quality.',
    imgSrc: 'https://greatexpectations.io/favicon.ico',
    href: 'https://github.com/great-expectations/great_expectations',
  },
  {
    title: 'Truera',
    description: 'AI explainability and model monitoring tool used in enterprise ML deployments.',
    imgSrc: 'https://truera.com/favicon.ico',
    href: 'https://github.com/truera/trulens',
  },

  // PROMPT ENGINEERING & LLM TRAINING
  {
    title: 'Prompt Engineering Guide',
    description: 'The definitive community-driven guide to prompt engineering for LLMs.',
    imgSrc: 'https://dair.ai/favicon.ico',
    href: 'https://github.com/dair-ai/Prompt-Engineering-Guide',
  },
  {
    title: 'Unsloth',
    description: 'Toolkit for fast, memory-efficient LLM fine-tuning and quantization.',
    imgSrc: 'https://avatars.githubusercontent.com/u/150654176?s=200&v=4',
    href: 'https://github.com/unslothai/unsloth',
  },
  {
    title: 'Lit-GPT',
    description: 'Lightning-fast, hackable open-source LLM implementation for research and fine-tuning.',
    imgSrc: 'https://lightning.ai/favicon.ico',
    href: 'https://github.com/Lightning-AI/lit-gpt',
  },
  {
    title: 'Axolotl',
    description: 'Open-source framework for fine-tuning and evaluating LLMs at scale.',
    imgSrc: 'https://avatars.githubusercontent.com/u/137642843?s=200&v=4',
    href: 'https://github.com/OpenAccess-AI-Collective/axolotl',
  },
  {
    title: 'FastChat',
    description: 'Open platform for training, serving, and evaluating large chat models like Vicuna.',
    imgSrc: 'https://avatars.githubusercontent.com/u/130668900?s=200&v=4',
    href: 'https://github.com/lm-sys/FastChat',
  },
  {
    title: 'Text Generation WebUI',
    description: 'A powerful web UI for running and experimenting with LLMs locally.',
    imgSrc: 'https://avatars.githubusercontent.com/u/130668900?s=200&v=4',
    href: 'https://github.com/oobabooga/text-generation-webui',
  },

  // DEPLOYMENT, MONITORING, & MLOPS
  {
    title: 'MLflow',
    description: 'Open-source platform for managing the ML lifecycle: tracking, packaging, and deploying models.',
    imgSrc: 'https://mlflow.org/favicon.ico',
    href: 'https://github.com/mlflow/mlflow',
  },
  {
    title: 'BentoML',
    description: 'Modern framework for building, deploying, and scaling machine learning applications.',
    imgSrc: 'https://bentoml.com/favicon.ico',
    href: 'https://github.com/bentoml/BentoML',
  },
  {
    title: 'Seldon Core',
    description: 'Open-source platform for deploying and monitoring ML models on Kubernetes.',
    imgSrc: 'https://seldon.io/wp-content/uploads/2022/09/cropped-favicon-192x192.png',
    href: 'https://github.com/SeldonIO/seldon-core',
  },
  {
    title: 'DVC',
    description: 'Data Version Control - versioning for datasets and ML pipelines, essential for reproducible research.',
    imgSrc: 'https://dvc.org/favicon.ico',
    href: 'https://github.com/iterative/dvc',
  },
  {
    title: 'Weights & Biases',
    description: 'Popular tool for experiment tracking, model management, and collaboration.',
    imgSrc: 'https://wandb.ai/favicon.ico',
    href: 'https://github.com/wandb/client',
  },
  {
    title: 'ZenML',
    description: 'MLOps framework for creating reproducible, production-ready ML pipelines.',
    imgSrc: 'https://zenml.io/favicon.ico',
    href: 'https://github.com/zenml-io/zenml',
  },

  // EXPLAINABILITY, INTERPRETABILITY, & FAIRNESS
  {
    title: 'LIME',
    description: 'Explains predictions of any classifier in a human-understandable way.',
    imgSrc: 'https://avatars.githubusercontent.com/u/12854235?s=200&v=4',
    href: 'https://github.com/marcotcr/lime',
  },
  {
    title: 'SHAP',
    description: 'Game-theoretic approach to explain the output of any ML model.',
    imgSrc: 'https://avatars.githubusercontent.com/u/12854235?s=200&v=4',
    href: 'https://github.com/slundberg/shap',
  },
  {
    title: 'Fairlearn',
    description: 'Tools for assessing and improving fairness in ML models.',
    imgSrc: 'https://fairlearn.org/favicon.ico',
    href: 'https://github.com/fairlearn/fairlearn',
  },

  // COMMUNITY CURATED LISTS & PROJECT COLLECTIONS
  {
    title: 'Awesome Artificial Intelligence',
    description: 'A curated list of the best AI resources: courses, books, research papers, and code.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/owainlewis/awesome-artificial-intelligence',
  },
  {
    title: 'Awesome AI & Data GitHub Repos',
    description: 'A comprehensive list of the most essential AI and ML repositories on GitHub.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/youssefHosni/Awesome-AI-Data-GitHub-Repos',
  },
  {
    title: 'Awesome LLM',
    description: 'A curated list of large language model resources, libraries, and tools.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/Hannibal046/Awesome-LLM',
  },
  {
    title: 'Awesome Data Science',
    description: 'A highly starred, community-driven resource for data science, machine learning, and AI.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/academic/awesome-datascience',
  },
  {
    title: '500 AI, ML, DL, CV, NLP Projects with Code',
    description: 'A massive collection of 500+ real-world projects with code, spanning ML, DL, CV, and NLP.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code',
  },
  {
    title: 'Made With ML',
    description: 'A practical guide for designing, developing, and deploying production-grade ML applications.',
    imgSrc: 'https://madewithml.com/favicon.ico',
    href: 'https://github.com/GokuMohandas/Made-With-ML',
  },
  {
    title: 'fastai/fastbook',
    description: 'The official fastai book and notebooks for practical deep learning.',
    imgSrc: 'https://fastai.fast.ai/images/favicon.ico',
    href: 'https://github.com/fastai/fastbook',
  },
  {
    title: '100 Days of ML Code',
    description: 'A popular repository for hands-on learning, guiding you through 100 days of machine learning code.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/Avik-Jain/100-Days-Of-ML-Code',
  },
  {
    title: 'Data Science For Beginners',
    description: 'A Microsoft-curated, beginner-friendly curriculum for learning data science and AI concepts from scratch.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/microsoft/Data-Science-For-Beginners',
  },

  // TRENDING & REDDIT-RECOMMENDED PROJECTS
  {
    title: 'OpenAI Whisper',
    description: 'Automatic Speech Recognition (ASR) system trained on 680k hours of multilingual data.',
    imgSrc: 'https://openai.com/favicon.ico',
    href: 'https://github.com/openai/whisper',
  },
  {
    title: 'Stable Diffusion WebUI',
    description: 'A popular web UI for running Stable Diffusion and other diffusion models locally.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/AUTOMATIC1111/stable-diffusion-webui',
  },
  {
    title: 'ComfyUI',
    description: 'A powerful and modular stable diffusion GUI and pipeline builder.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/comfyanonymous/ComfyUI',
  },
  {
    title: 'KoboldAI',
    description: 'A locally running, web-based front end for GPT-like models, popular in the AI writing community.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/KoboldAI/KoboldAI-Client',
  },
  {
    title: 'DeepSpeed',
    description: 'A deep learning optimization library for distributed training of large models.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/microsoft/DeepSpeed',
  },
  {
    title: 'Colossal-AI',
    description: 'A unified system for large-scale model training, inference, and serving.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/hpcaitech/ColossalAI',
  },
  {
    title: 'Open-Assistant',
    description: 'Open-source chat-based assistant that aims to be a free alternative to ChatGPT.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/LAION-AI/Open-Assistant',
  },
  {
    title: 'OpenMMLab',
    description: 'A collection of open-source computer vision toolboxes from the MMLab group.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/open-mmlab',
  },
  {
    title: 'DeepFace',
    description: 'A lightweight face recognition and facial attribute analysis framework for Python.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/serengil/deepface',
  },
  {
    title: 'FaceSwap',
    description: 'A popular open-source tool for deepfakes and face swapping.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/deepfakes/faceswap',
  },
  {
    title: 'Awesome AI Data-Guided Projects',
    description: 'A curated collection of hands-on, guided AI and data science projects.',
    imgSrc: 'https://github.githubassets.com/images/icons/emoji/unicode/1f4a1.png',
    href: 'https://github.com/youssefHosni/Awesome-AI-Data-Guided-Projects',
  },
];


export default projectsData
