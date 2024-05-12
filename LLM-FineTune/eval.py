import transformers
import peft
import hydra
import utils
import torchinfo
import evaluate
import os
import torch
import accelerate
import json
# from tqdm.auto import tqdm
from accelerate.utils import tqdm
import pandas as pd

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):        
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    model, tokenizer = utils.prepare_model_and_tokenizer(cfg.model)
    
    if cfg.eval.combine_peft:
        model = peft.PeftModel.from_pretrained(model, cfg.eval.peft_path)

    if accelerator.is_local_main_process:
        torchinfo.summary(model)

    dataset = utils.prepare_data(cfg.data)
    eval_dataset = dataset[cfg.eval.eval_split]
    data_dict = eval_dataset.to_dict()

    ground_truth = eval_dataset[cfg.data.dataset.label_field]

    dataloader = torch.utils.data.DataLoader(
        eval_dataset.with_format('torch'),
        batch_size=2
    )

    model, eval_dataset = accelerator.prepare(model, dataloader)
    model.to(device)
    # print(device)


    responses = []
    with torch.no_grad():
        for batch in tqdm(True, dataloader):
            tokens = tokenizer(
                batch[cfg.data.dataset.input_field],
                return_tensors='pt',
                padding=True,
            )
            response_tokens = model.generate(
                input_ids=tokens['input_ids'].to(device),
                attention_mask=tokens['attention_mask'].to(device),
                max_new_tokens=100
            )

            # print(responses)
            responses.extend(tokenizer.batch_decode(response_tokens.cpu().numpy(), skip_special_tokens=True))
    # print(responses)
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        if not os.path.exists(cfg.eval.results_path):
            os.mkdir(cfg.eval.results_path)
        accelerator.clear()
        data_dict['generation'] = responses
        pd.DataFrame(data_dict).to_json(f"{os.path.join(cfg.eval.results_path, cfg.model.name)}.json", indent=4)
        # ground_truth, responses = accelerator.gather_for_metrics(ground_truth, responses)
        for metric_name, metric_values in cfg.eval.metrics.items():
            print(metric_name)
            metric = hydra.utils.instantiate(metric_values.OBJ)
            func = getattr(metric, metric_values.compute_fn)
            res = func(references=ground_truth, predictions=responses, **metric_values.get('compute_args', {}))
            print(f"{metric_name}: {res[metric_values.target_key]}")




if __name__ == "__main__":
    main()