# Download PDF file
import os
import requests
import pandas as pd
import random
import fitz
import re
import time
import torch
import textwrap
import random
import numpy as np
import matplotlib.pyplot as plt

# for progress bars, requires !pip install tqdm
from tqdm.auto import tqdm
# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from time import perf_counter as timer
from spacy.lang.en import English
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings



