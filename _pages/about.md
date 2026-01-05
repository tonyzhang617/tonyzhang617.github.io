---
layout: about
title: about
permalink: /
subtitle: '<span style="color: gray;">Bridging Algorithms and Hardware for Efficient AI</span>'

profile:
  align: right
  image: prof_pic.jpg
  image_circular: true # crops the image to make it circular
  more_info: 

selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

I am an AI researcher specializing in the design of efficient training and inference methods for foundational models. My research focuses on bridging the gap between algorithmic innovation and hardware acceleration, with emphasis on model compression, quantization, quantization-aware training, and custom CUDA kernels.

### Education

I received my Ph.D. in Computer Science from **Rice University** in August 2025, advised by [Prof. Anshumali Shrivastava](https://www.cs.rice.edu/~as143/). Prior to that, I completed my undergraduate studies in Computer Science at the **University of Waterloo** in Canada, graduating with distinction in 2021.

### Research Highlights

My recent work, **DFloat11** [NeurIPS '25], is a pioneering approach for losslessly compressing foundation models, including Large Language Models (LLMs) and Diffusion Transformers, to enable efficient GPU inference. Unlike quantization or pruning methods that degrade model quality, DFloat11 produces outputs that are bit-for-bit identical to the original BFloat16 models while reducing model size by 32%, enabling efficient deployment on lower-tier GPUs. The project has gained significant community attention and adoption:

* Trended at **#2 on Hacker News** ([Discussion](https://news.ycombinator.com/item?id=43796935))
* 30K+ downloads of the open-source Python package on [PyPI](https://pepy.tech/projects/dfloat11)
* 200K+ downloads of the open-source models on [Hugging Face](https://huggingface.co/DFloat11)
* Media coverage by [MIT Technology Review (China)](https://www.mittrchina.com/news/detail/14704), [AI Era (新智元)](https://mp.weixin.qq.com/s/uX6UEl1lUCNI_OFcseZz3Q), and [Synced (机器之心)](https://mp.weixin.qq.com/s/1krObzWLaX8CrzUW2OR_kQ)

### Entrepreneurship

I co-founded the AI startup **xMAD.ai** in 2024, where I led research and product development. Our mission of building customized, cost-effective AI agents was supported by funding from Non-Sibi Ventures, AISprouts, and the Hopper-Dean Foundation.

In August 2025, xMAD.ai was acquired by **Workato** to launch their first AI research lab. Read [the official press release here](https://www.businesswire.com/news/home/20250814122383/en/Workato-Launches-AI-Research-Lab-and-Accelerates-as-an-AI-First-Enterprise).

### Industry Experience

I have held research internships at **Visa Research**, **Amazon**, and **Intel**. At Visa Research, I was a key contributor to **TransactionGPT**, developing a novel compression algorithm for transaction foundation models that reduced communication bottlenecks and significantly accelerated training on H100 clusters ([paper](https://arxiv.org/abs/2511.08939)).

### Academic Service

* **Top Reviewer** Award, NeurIPS 2025
* Reviewer @ NeurIPS, ICML, ICLR, ACL, EMNLP, KDD, WWW