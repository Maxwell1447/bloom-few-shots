from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import tqdm
import re
import sys


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model-path", default="/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-560m", help="path to model")
    args.add_argument("--file", required=True, help="path to context input file")
    args.add_argument("--max-new-tokens", default=512, type=int, help="maximum number of generated tokens")
    args.add_argument("--out", required=True, help="path to generated output file")
    args.add_argument("--tqdm", action='store_true', help="progress bar display")
    args.add_argument("--debug", action='store_true', help="when true, execute debug function instead")
    return args.parse_args()


def load(model_path):
    print("loading model...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer.add_special_tokens({"sep_token": "\n###\n"})
    return model, tokenizer


def generate_from_ctx(input_ctx, model, tokenizer, eos_token_id, max_new_tokens, return_tensors=False):
    # Encode the initial input text
    input_ids = tokenizer(input_ctx, return_tensors="pt").input_ids

    cut_start = input_ids.shape[-1]

    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id
    )
    # Convert the generated token IDs to text
    generated_text = tokenizer.decode(out[0][cut_start:], skip_special_tokens=True)[1:]
    if return_tensors:
        return input_ids, out[0][cut_start:], generated_text
    return generated_text


def main(args):
    out_file = open(args.out, 'w')

    with open(args.file) as f:

        model, tokenizer = load(args.model_path)

        sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        equal_token_id = 564
        break_tok_id = 189
        point_break_tok_id = 336
        colon_tok_id = 915
        point_tok_id = 17

        eos_token_id = [
            tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
            sep_token_id,
            equal_token_id,
            break_tok_id,
            point_break_tok_id,
            point_tok_id
        ]

        if args.tqdm:
            f = tqdm.tqdm(f)

        for line in f:
            input_ctx = re.sub('\t', '\n###\n', line.rstrip('\n'))
            generated_text = generate_from_ctx(input_ctx, model, tokenizer, eos_token_id, args.max_new_tokens)
            if generated_text[-1] == '=':
                generated_text = generated_text[:-1]

            out_file.write(generated_text.rstrip('\n').split('\n')[0] + '\n')
            out_file.flush()
            
    out_file.close()


def debug(args):
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # break_ids = list()
    # for i in range(tokenizer.vocab_size):
    #     if '\n' in tokenizer.decode(i):
    #         break_ids.append(i)
    
    # print(break_ids)
    # print([tokenizer.decode(i) for i in break_ids])

    model, tokenizer = load(args.model_path)
    sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    equal_token_id = 564
    break_tok_id = 189
    eos_token_id = [
        sep_token_id,
        equal_token_id,
        tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
        break_tok_id,
        336
    ]
    print("stop tokens:\n", eos_token_id)

    input_ctx = """A 1
B 2 
C 3
D 4
"""

    input_ids, output_ids, generated_text = generate_from_ctx(input_ctx, model, tokenizer, eos_token_id, 5, return_tensors=True)

    print(generated_text)
    print("input")
    print(input_ids)
    print([tokenizer.decode(i) for i in input_ids[0].tolist()])
    print("output")
    print(output_ids)
    print([tokenizer.decode(i) for i in output_ids.tolist()])



if __name__ == "__main__":

    args = parse_args()

    if args.debug:
        debug(args)
    else:
        main(args)

    

