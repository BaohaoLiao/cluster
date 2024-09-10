from tqdm import tqdm
import torch
import torch.nn as nn
from calibrate.data_utils import get_loaders

@torch.no_grad()
def evaluate(model, tokenizer, args, logging):
    logging.info("=== start evaluation ===")
    results = {}
    model = model.to(args.device, dtype=torch.bfloat16)
    
    if args.eval_ppl:
        for dataset in ["wikitext2", "c4", "ptb"]:
            _, testloader = get_loaders(
                dataset,
                tokenizer,
                args.cache_dir,
                seed=args.seed,
                seqlen=2048,
            )

            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // args.seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * args.seqlen) : ((i + 1) * args.seqlen)].to(args.device)
                outputs = model.model(batch)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * args.seqlen) : ((i + 1) * args.seqlen)][
                    :, 1:
                ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * args.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen))
            logging.info(f'{dataset} : {ppl.item()}')
            model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    return results