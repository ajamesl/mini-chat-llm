import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Update the MODEL_PATH to reflect the relative folder from the project root
MODEL_PATH = "checkpoints/merged"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
model.eval()


def generate_stream(prompt, max_new_tokens=256, temperature=0.7, top_p=0.95):
    input_text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    import threading
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    output_so_far = ""
    prefix = input_text
    prefix_removed = False

    for new_text in streamer:
        output_so_far += new_text
        if not prefix_removed:
            # Remove any occurrence of the prompt (even with stray whitespace)
            idx = output_so_far.find("Assistant:")
            if idx != -1:
                # Remove everything up to and including 'Assistant:'
                output_so_far = output_so_far[idx + len("Assistant:"):].lstrip("\n ")
                prefix_removed = True
            else:
                # If 'Assistant:' isn't found yet, keep buffering
                output_so_far = ""
                continue  # Wait for the next chunk
        yield output_so_far
        output_so_far = ""  # Only yield new content each time