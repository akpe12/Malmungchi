import json
import re
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default='beomi/llama-2-ko-7b')
parser.add_argument("--save_path", type=str, default="./upload.json")
parser.add_argument("--model_path", type=str, default="./checkpoint")
parser.add_argument("--adapter_path", type=str, default="./checkpoint")
parser.add_argument("--generate_batch_size", type=int, default=4)
parser.add_argument("--test_file_path", type=str,
                    default='nikluge-sc-2023-test.jsonl')
parser.add_argument("--inference_mode", type=str,
                    choices=['full', 'qlora', 'lora'])
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
special_tokens = {
    'pad_token': DEFAULT_PAD_TOKEN,
    'eos_token': DEFAULT_EOS_TOKEN,
    'bos_token': DEFAULT_BOS_TOKEN,
    'unk_token': DEFAULT_UNK_TOKEN,
}
tokenizer.add_special_tokens(special_tokens)

# load model
if args.train_mode == 'qlora':
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().device}
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter_path).to(device)

if args.train_mode == 'lora':
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().device}
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter_path).to(device)

if args.train_mode == 'full':
    model_base = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.model_path)
    model = model_base.cpu()
    model.load_state_dict(state_dict)
    model.to(device)

# preprocessing
dataset = load_dataset("json", data_files={"test": args.test_file_path})
generation_config = dict(num_beams=5)


def generate_prompt(data_point):
    return f'''
        문장1 : 우리는 의자를 큰 책상 주위에 빙 둘러놓았다.
        문장3 : 학생들이 모두 착석한 뒤 회의가 시작되었다.
        문장2 : 그러자 의자에 학생들이 하나 둘 앉기 시작했다.
        
        32토큰 이하로 두 문장 사이에 올 자연스러운 문장을 작성하세요 -\n\n
        순서: 문장1, 문장2, 문장3 \n\n
        문장1 : {data_point["input"]["sentence1"]} \n\n
        문장3 : {data_point["input"]["sentence3"]} \n\n
        문장2 :
        '''.strip()


def gen_token_prompt(data_point):
    prompt = generate_prompt(data_point)
    return {'prompt': prompt}


dataset = dataset['test'].map(gen_token_prompt)


class Custom(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokens = tokenizer(item, padding='max_length', truncation=True,
                           max_length=165, return_tensors='pt')  # 165, 110
        return {'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask']
                }


batch_size = args.generate_batch_size
dataset2 = Custom(dataset['prompt'])
dataloader = DataLoader(dataset2, batch_size=batch_size)

# generate
model.eval()
output = []
output2 = []
output3 = []

for j in tqdm(dataloader):
    gened = model.generate(
        input_ids=j['input_ids'].squeeze(1).to(device),
        attention_mask=j['attention_mask'].squeeze(1).to(device),
        max_length=215,  # 215, 150
        **generation_config,
    )
    decoding = tokenizer.batch_decode(gened, skip_special_tokens=True)
    print(decoding)
    output.append(decoding)

output2 = sum(output, [])
output3 = []

# postprocessing


def postprocessing(decode):
    pattern = r"문장1 : .*? \n\n\n        문장3 : .*? \n\n\n        문장2 : (.*?) \n\n"

    match = re.search(pattern, decode, re.DOTALL)
    if match:
        k = match.group(1).strip()
    return k


def postprocessing_2(decode):
    parts = decode.split("\n\n\n        ")
    for part in parts:
        if "문장2" in part:
            return part.replace("문장2 : ", "").strip()


def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]

    return j_list


def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


for i in range(len(output2)):
    y = postprocessing_2(output2[i][205:])  # 205, 77
    output3.append(y)

pattern = '\n\n'
pattern2 = '\u200b'
for n, i in enumerate(output3):
    if pattern in i:
        output3[n] = i.split(pattern)[0]
    if pattern2 in i:
        output3[n] = i.split(pattern2)[0]

j_list = jsonlload(args.test_file_path)

# save file
for idx, oup in enumerate(output3):
    j_list[idx]["output"] = oup
jsonldump(j_list, args.save_path)
