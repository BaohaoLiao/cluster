import os
import gc
import pdb
import math
import copy
import torch
import torch.nn as nn

from calibrate.utils import NativeScalerWithGradNormCount, get_parameters, vector_bank_state_dict


def cal(ori_model, clus_model, args, dataloader, logging=None):
    logging.info("Starting ...")
    use_cache = ori_model.config.use_cache
    ori_model.config.use_cache = False
    clus_model.config.use_cache = False

    is_llama = True
    ori_layers = ori_model.model.layers
    clus_layers = clus_model.model.layers
    ori_model.model.embed_tokens = ori_model.model.embed_tokens.to(args.device)
    ori_model.model.norm = ori_model.model.norm.to(args.device)
    ori_model.model.rotary_emb = ori_model.model.rotary_emb.to(args.device)
    clus_model.model.rotary_emb = clus_model.model.rotary_emb.to(args.device)
    
    ori_layers[0] = ori_layers[0].to(args.device)
    dtype = torch.bfloat16
    traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, args.seqlen, ori_model.config.hidden_size), dtype=dtype, device=args.device
    )
    cache = {"i": 0}
    
    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    ori_layers[0] = Catcher(ori_layers[0])
    ori_layers[0].is_llama = is_llama
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                ori_model(batch[0].to(args.device))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    ori_layers[0] = ori_layers[0].module
    ori_layers[0] = ori_layers[0].cpu()
    ori_model.model.embed_tokens = ori_model.model.embed_tokens.cpu()
    ori_model.model.norm = ori_model.model.norm.cpu()
    torch.cuda.empty_cache()

    # same input for the first layer of ori model and clus model
    clus_inps = inps
    ori_inps = copy.deepcopy(inps)   # take output of fp model as input
    ori_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # qlayer and layer use the same quant_inps
    
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logging.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if args.resume:
        vector_banks = torch.load(os.path.join(args.resume, "vector_banks.pth"))
    else:
        vector_banks = {}

    for i in range(len(ori_layers)):
        logging.info(f"=== Start calibrate layer {i} ===")
        ori_layer = ori_layers[i].to(args.device)
        clus_layer = clus_layers[i].to(args.device)

        # obtain output of full-precision model
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        ori_inps[j] = ori_layer(
                            ori_inps[j].unsqueeze(0), 
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        )[0]
                        if args.aug_loss:
                            ori_inps_2[j] = ori_layer(
                                clus_inps[j].unsqueeze(0), 
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )[0]

        if args.resume:
            clus_layer.load_state_dict(vector_banks[i], strict=False)

        if args.epochs > 0:
            with torch.no_grad():
                clus_layer.float()  # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW([
                {"params": clus_layer.parameters(), "lr": args.lr, "weight_decay": args.wd},
            ])
            loss_scaler = NativeScalerWithGradNormCount()

            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples // args.batch_size):    
                    index = j * args.batch_size
                    with traincast():
                        clus_out = clus_layer(
                            clus_inps[index:index+args.batch_size,], 
                            attention_mask=attention_mask_batch,
                            position_ids=position_ids
                        )[0]
                        loss = loss_func(ori_inps[index:index+args.batch_size,], clus_out)
                        if args.aug_loss:
                            loss += loss_func(ori_inps_2[index:index+args.batch_size,], clus_out)

                    if not math.isfinite(loss.item()):
                        logging.info("Loss is NAN, stopping training")
                        pdb.set_trace()

                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=get_parameters(clus_layer)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logging.info(f"layer {i} epoch {epoch} \t|| loss: {loss_mean}\t"
                             f"norm: {norm_mean}\tmax memory_allocated: {torch.cuda.max_memory_allocated(args.device) / 1024**2}")
            del optimizer

        clus_layer.half()
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        clus_inps[j] = clus_layer(
                            clus_inps[j].unsqueeze(0), 
                            attention_mask=attention_mask, 
                            position_ids=position_ids
                        )[0]
            clus_layers[i] = clus_layer.to("cpu")
            vector_banks[i] = vector_bank_state_dict(clus_layer)
            torch.save(vector_banks, os.path.join(args.save_dir, f"vector_banks.pth"))
        else:
            clus_layers[i] = clus_layer.to("cpu")

        del ori_layer, clus_layer
        torch.cuda.empty_cache()

    del inps
    del clus_inps
    del ori_inps
    del ori_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    clus_model.config.use_cache = use_cache

    return clus_model