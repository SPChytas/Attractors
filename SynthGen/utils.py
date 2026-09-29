from datasets import load_dataset
from datasets import Dataset
from tqdm.auto import tqdm
import numpy as np 
import re


def get_original_samples(dataset, split):

    if (dataset == 'BoolQ'):
        assert split in ['train', 'validation'], 'Unknown split %s (should be \'train\' or \'validation\')' %(split)

        ds = load_dataset("google/boolq", cache_dir='/data/takis/datasets/hf_hub')[split]

        prompts = []
        for i, row in tqdm(enumerate(ds), total=len(ds)):
            prompts.append('passage: %s, question: %s, answer: %s' %(row['passage'], row['question'], row['answer']))
        prompts = np.array(prompts)


    elif (dataset == 'GSM8K'):  
        assert split in ['train', 'test'], 'Unknown split %s (should be \'train\' or \'test\')' %(split)

        ds = load_dataset("openai/gsm8k", 'main', cache_dir='/data/takis/datasets/hf_hub')[split]

        prompts = []
        for i, row in tqdm(enumerate(ds), total=len(ds)):
            prompts.append('question: %s, answer: %s' %(row['question'], row['answer']))
        prompts = np.array(prompts)

    elif (dataset == 'AG'):

        assert split in ['train', 'test'], 'Unknown split %s (should be \'train\' or \'test\')' %(split)

        ds = load_dataset("fancyzhx/ag_news", cache_dir='/data/takis/datasets/hf_hub')[split]

        prompts = []
        for i, row in tqdm(enumerate(ds), total=len(ds)):
            
            if (row['label'] == 0 or row['label'] == 'World'):
                response = 'World'
            elif (row['label'] == 1 or row['label'] == 'Sports'):
                response = 'Sports'
            elif (row['label'] == 2 or row['label'] == 'Business'):
                response = 'Business'
            elif (row['label'] == 3 or row['label'] == 'Technology'):
                response = 'Technology'

            prompts.append('TEXT: %s, CLASS: %s' %(row['text'], response))
        prompts = np.array(prompts)


    else:
        raise ValueError('Unknown dataset %s' %(dataset))




    return prompts


def get_in_context_prompt(dataset):

    if (dataset == 'BoolQ'):
        prompt = '''Now generate 3 different passages, questions, and answers similar to the example above. Please make sure each question you generate has a boolean answer that can be answered by the passage. Make sure each passage and question is different and sufficiently rephrased. Please make sure you generate passages, questions and both true and false answers.'''
    elif (dataset == 'GSM8K'):
        prompt = '''Now generate 3 different questions and answers that require solving a grade-school math problem similar to the example above. Make sure each question is different and sufficiently rephrased and the answer is elaborate. Please make sure each question you generate has a well-defined answer.'''
    elif (dataset == 'AG'):
        prompt = '''Now generate 3 different texts and their corresponding class similar to the example above. Make sure each text is not too long and it is different and sufficiently rephrased. Please make sure each class you generate belongs to one of the four classes (Technology, World, Business, Sports).'''
    else:
        raise ValueError('Unknown dataset %s' %(dataset))

    return prompt




def _split_boolq_sample(sample, max_additions=-1):

    sample = re.sub(r'(?i)\bp\s*\d*\s*:', 'Passage:', sample)
    sample = re.sub(r'(?i)\bq\s*\d*\s*:', 'Question:', sample)
    sample = re.sub(r'(?i)\ba\s*\d*\s*:', 'Answer:', sample)

    # print (sample)

    splits = re.split(r'(?i)\b\W*passage\s*\d*\s*\W*', sample)

    for i in range(len(splits)):
        splits[i] = re.split(r'(?i)\b\W*question\s*\d*\s*\W*', splits[i])[:2]
    for i in range(len(splits)):
        if (len(splits[i]) > 1):
            splits[i][1] = re.split(r'(?i)\b\W*answer\s*\d*\s*\W*', splits[i][1])[:2]


    passages = []
    questions = []
    answers = []
    additions = 0

    for s in splits:
        if (len(s) != 2 or len(s[1]) != 2 or len(s[0].strip()) < 20  or len(s[1][0].strip()) < 10):
            continue 

        passages.append(s[0].strip())
        questions.append(s[1][0].strip())

        answer = s[1][1].strip()
        if ('True' in answer or 'true' in answer or 'Yes' in answer or 'yes' in answer):
            answers.append(True)
        else:
            answers.append(False)
    
        additions += 1
        if (max_additions > 0 and additions >= max_additions):
            break

        # if (len(passages) >= 3):
        #     break

    return passages, questions, answers


def _split_gsm8k_sample(sample, max_additions=-1):

    sample = re.sub(r'(?i)\bq\s*\d*\s*:', 'Question:', sample)
    sample = re.sub(r'(?i)\ba\s*\d*\s*:', 'Answer:', sample)

    # print (sample)

    splits = re.split(r'(?i)\b\W*question\s*\d*\s*\W*', sample)
    for i in range(len(splits)):
        splits[i] = re.split(r'(?i)\b\W*answer\s*\d*\s*\W*', splits[i])[:2]


    questions = []
    answers = []
    additions = 0

    for s in splits:
        if (len(s) != 2 or len(s[0].strip()) < 20  or len(s[1].strip()) < 10):
            continue 

        questions.append(s[0].strip())
        answers.append(s[1].strip())

        additions += 1
        if (max_additions > 0 and additions >= max_additions):
            break

    return questions, answers


def _split_imdb_sample(sample, max_additions=-1):

    # sample = re.sub(r'(?i)\bq\s*\d*\s*:', 'Question:', sample)
    # sample = re.sub(r'(?i)\ba\s*\d*\s*:', 'Answer:', sample)

    # print (sample)

    splits = re.split(r'(?i)\b\W*review\s*\d*\s*\W*', sample)
    for i in range(len(splits)):
        splits[i] = re.split(r'(?i)\b\W*sentiment\s*\d*\s*\W*', splits[i])[:2]


    reviews = []
    sentiments = []
    additions = 0

    for s in splits:
        if (len(s) != 2 or len(s[0].strip()) < 10):
            continue 

        reviews.append(s[0].strip())
        sentiments.append(s[1].strip())

        additions += 1
        if (max_additions > 0 and additions >= max_additions):
            break

    return reviews, sentiments


def _split_ag_sample(sample, max_additions=-1):

    # sample = re.sub(r'(?i)\bq\s*\d*\s*:', 'Question:', sample)
    # sample = re.sub(r'(?i)\ba\s*\d*\s*:', 'Answer:', sample)

    # print (sample)

    splits = re.split(r'(?i)\b\W*text\s*\d*\s*\W*', sample)
    for i in range(len(splits)):
        splits[i] = re.split(r'(?i)\b\W*class\s*\d*\s*\W*', splits[i])[:2]

    texts = []
    classes = []


    additions = 0
    for s in splits:
        if (len(s) != 2 or len(s[0].strip()) < 10):
            continue 
        
        cur_class = s[1].lower().strip()

        if ('technology' in cur_class):
            class_id = 3
        elif ('world' in cur_class):
            class_id = 0
        elif ('business' in cur_class):
            class_id = 2
        elif ('sports' in cur_class):
            class_id = 1
        else:
            continue

        texts.append(s[0].strip())
        classes.append(class_id)
        
        additions += 1
        if (max_additions > 0 and additions >= max_additions):
            break

    return texts, classes



def parse_new_samples(samples, dataset):

    # Remove any white spaces
    for i in range(len(samples)):
        samples[i] = re.sub(r'\s+', ' ', samples[i].strip())


    if (dataset == 'BoolQ'):
        
        passages = []
        questions = []
        answers = []

        for i in range(len(samples)):
            p, q, a = _split_boolq_sample(samples[i], -1)
            passages.extend(p)
            questions.extend(q)
            answers.extend(a)

        ds = Dataset.from_dict({'passage': passages, 'question': questions, 'answer': answers})
        return ds

    elif (dataset == 'GSM8K'):

        questions = []
        answers = []

        for i in range(len(samples)):
            q, a = _split_gsm8k_sample(samples[i], -1)
            questions.extend(q)
            answers.extend(a)

        ds = Dataset.from_dict({'question': questions, 'answer': answers})
        return ds

    elif (dataset == 'AG'):
        
        texts = []
        classes = []

        for i in range(len(samples)):
            t, c = _split_ag_sample(samples[i], -1)
            texts.extend(t)
            classes.extend(c)

        ds = Dataset.from_dict({'text': texts, 'label': classes})
        return ds


def process_boolq_prediction(pred):
    pred = pred.lower()
    if ('true' in pred or 'yes' in pred or 'yeah' in pred):
        return True
    else:
        return False

def process_gsm8k_prediction(pred):
    match = re.search(r'####\s*[\d|.]+', pred)
    if (match is None):
        return '' 
    else:
        return match.group(0)[4:].strip()

def process_imdb_prediction(pred):
    pred = pred.lower()
    if ('pos' in pred or 'positive' in pred):
        return 'positive'
    elif ('neg' in pred or 'negative' in pred):
        return 'negative'
    else:
        return ''

def process_ag_prediction(pred):
    pred = pred.lower().strip()
    return pred