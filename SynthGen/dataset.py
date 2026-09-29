from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
import torch




class BoolQ_dataset(torch.utils.data.Dataset):

    def __init__(self, keep=100, additional_data_path=None, split='train'):

        self.ds = load_dataset("google/boolq", cache_dir='/data/takis/datasets/hf_hub')[split]
        
        if (split == 'train'):
            self.ds = Dataset.from_dict(self.ds[:keep])

        if (additional_data_path is not None):
            self.ds = concatenate_datasets([self.ds, load_from_disk(additional_data_path)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        prompt = 'Please answer the following question based on the passage. Your answer should be either True or False. Do not provide any other justification. passage: %s, question: %s, answer: ' %(self.ds[idx]['passage'], self.ds[idx]['question'])
        response = '%s' %(self.ds[idx]['answer'])
        return prompt, response




class GSM8K_dataset(torch.utils.data.Dataset):

    def __init__(self, keep=100, additional_data_path=None, split='train'):

        self.ds = ds = load_dataset("openai/gsm8k", 'main', cache_dir='/data/takis/datasets/hf_hub')[split]
        
        if (split == 'train'):
            self.ds = Dataset.from_dict(self.ds[:keep])
            self.in_context_prompt = None
        else:
            train_ds = load_dataset("openai/gsm8k", 'main', cache_dir='/data/takis/datasets/hf_hub')['train']
            self.in_context_prompt = None #'\n'.join(['QUESTION: ' + train_ds[i]['question'] + ' ANSWER: ' + train_ds[i]['answer'] for i in range(2)])

        if (additional_data_path is not None):
            self.ds = concatenate_datasets([self.ds, load_from_disk(additional_data_path)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if (self.in_context_prompt is None):
            prompt = 'QUESTION: ' + self.ds[idx]['question'] + ' ANSWER: '
        else:
            prompt = self.in_context_prompt + '\n' + 'QUESTION: ' + self.ds[idx]['question']  + ' ANSWER: '
            
        response = self.ds[idx]['answer'] + '\n'
        return prompt, response



class IMDB_dataset(torch.utils.data.Dataset):

    def __init__(self, keep=100, additional_data_path=None, split='train'):

        self.ds = ds = load_dataset("stanfordnlp/imdb", cache_dir='/data/takis/datasets/hf_hub')[split]
        
        if (split == 'train'):
            self.ds = Dataset.from_dict(self.ds[:keep])

        if (additional_data_path is not None):
            self.ds = concatenate_datasets([self.ds, load_from_disk(additional_data_path)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        prompt = 'REVIEW: ' + self.ds[idx]['text'] + ' SENTIMENT (positive/negative): '
        response = 'negative' if (self.ds[idx]['label']==0 or self.ds[idx]['label']=='0') else 'negative'
        return prompt, response



class AG_dataset(torch.utils.data.Dataset):

    def __init__(self, keep=100, additional_data_path=None, split='train'):

        self.ds = ds = load_dataset("fancyzhx/ag_news", cache_dir='/data/takis/datasets/hf_hub')[split]
        
        if (split == 'train'):
            self.ds = Dataset.from_dict(self.ds[:keep])

        if (additional_data_path is not None):
            self.ds = concatenate_datasets([self.ds, load_from_disk(additional_data_path)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        prompt = 'TEXT: ' + self.ds[idx]['text'] + ' CLASS (Technology/Business/World/Sports): '
        if (self.ds[idx]['label'] == 0):
            response = 'World'
        elif (self.ds[idx]['label'] == 1):
            response = 'Sports'
        elif (self.ds[idx]['label'] == 2):
            response = 'Business'
        elif (self.ds[idx]['label'] == 3):
            response = 'Technology'
        else:
            raise ValueError()
        
        return prompt, response
