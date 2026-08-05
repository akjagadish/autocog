"""
Compatibility shim for older Pillow versions:
Some transformer/unsloth code expects PIL.Image.Resampling, which is available
from Pillow >= 9.1. On older stacks (e.g., Anaconda 2021.11 with Pillow 8.x),
importing transformers may fail with:
    AttributeError: module 'PIL.Image' has no attribute 'Resampling'
We defensively add a Resampling enum backed by existing constants if missing,
BEFORE importing unsloth/transformers.
"""
try:
    from PIL import Image as _PIL_Image  # type: ignore
    if not hasattr(_PIL_Image, "Resampling"):
        class _Resampling:  # minimal shim mapping
            NEAREST = _PIL_Image.NEAREST
            BOX = getattr(_PIL_Image, "BOX", _PIL_Image.BOX) if hasattr(_PIL_Image, "BOX") else _PIL_Image.BILINEAR
            BILINEAR = _PIL_Image.BILINEAR
            HAMMING = getattr(_PIL_Image, "HAMMING", _PIL_Image.BILINEAR)
            BICUBIC = _PIL_Image.BICUBIC
            LANCZOS = getattr(_PIL_Image, "LANCZOS", _PIL_Image.BICUBIC)

        # Attach shim to PIL.Image namespace so downstream imports succeed
        setattr(_PIL_Image, "Resampling", _Resampling)
except Exception:
    # If PIL isn't available yet or anything goes wrong, proceed as-is; the
    # real error will surface later, but we avoid masking unrelated issues.
    pass

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import transformers 
import torch
from peft import PeftModel

class UnslothAgent:
    """
    class for loading a model with unsloth
    there are two distinct ways of providing a generator for prediction:
            - using standard transformers pipeline
            - using model.generate() to allow retrieval of logprobs
    """
    def __init__(self, model_name, max_seq_length=None, dtype=None, load_in_4bit=True, logprob=False, top_k=10, local_files_only=False):
        """
        args:
            model_name  [str] name of the model to load
            ...
            logprob     [bool] when True the generator is model.generate(), othewise pipeline
            top_k       [int] number of top tokens for which logprobs are provided (only if logprob=True)
        """

        if local_files_only:
            BASE = "/scratch/gpfs/GRIFFITHS/aj9225/.cache/huggingface/Meta-Llama-3.1-70B-bnb-4bit"
            ADAPT = "/scratch/gpfs/GRIFFITHS/aj9225/.cache/huggingface/Llama-3.1-Centaur-70B-adapter"

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=BASE,
                max_seq_length = max_seq_length,
                dtype = dtype,
                device_map = "auto",
                load_in_4bit = load_in_4bit,
                local_files_only = local_files_only,
            )
            model = PeftModel.from_pretrained(model, ADAPT, is_trainable = False,   # or True if you want to keep fine-tuning
                                            device_map = "auto")
        else:
            raise NotImplementedError("Loading models without local_files_only needs to tested before use.")
        FastLanguageModel.for_inference(model)

        self.model          = model
        self.tokenizer      = tokenizer
        self.eos_token_id   = tokenizer( ">>" ).input_ids[ -1 ]
        
        if logprob:         # if internal token scores are required, do not use transformers pipeline
            self.pipeline   = None
            self.top_k      = top_k
        else:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=True,
                trust_remote_code=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=1, 
            )

    def next_token_logprobs( self, prompt ):
        """
        executes model.generate requiring scores
        return:
            [tuple] text of the next token [str], lgopropbs of the top_k tokens [dict]
        """
        enc     = self.tokenizer(prompt, return_tensors="pt")
        enc     = {k: v.to(self.model.device) for k, v in enc.items()}

        gen_out = self.model.generate(
            **enc,
            max_new_tokens=1,                       # 1 for just the next token
            do_sample=False,                        # False = greedy (like temperature=0)
            eos_token_id=self.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,                     # make logprobs available
        )

        # For max_new_tokens=1, there is exactly one score tensor
        logits      = gen_out.scores[0][0]              # shape: (vocab_size,)
        logprobs    = torch.log_softmax(logits, dim=-1)

        next_id     = gen_out.sequences[0, -1].item()
        next_tok    = self.tokenizer.decode([next_id]).strip()
        t_lp, t_id  = torch.topk( logprobs, k=self.top_k )
        top_dict    = dict()
        for lp, i in zip( t_lp, t_id ):
            k       = self.tokenizer.decode( [i.item()] ).strip()
            top_dict[ k ] = lp.item()

        return next_tok, top_dict


    def __call__(self, prompt, **gen_kwargs):
        if self.pipeline is not None:
            return self.pipeline(prompt, **gen_kwargs)[0]['generated_text'][len(prompt):]

        return self.next_token_logprobs( prompt )
