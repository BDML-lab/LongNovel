
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from diskcache import Cache
import asyncio
from tqdm import tqdm
import json
import re
import argparse
import numpy as np
import os
from transformers import AutoTokenizer

class VLLMInference:
    def __init__(self, model_path, tensor_parallel_size=8,long_context=None):
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        }
        rope_scaling_llama =  {
            "factor": 8.0,  
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        if long_context == "16k":
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.7, 
                trust_remote_code=True
            )
        elif  model_path=="llama3_1-8b-instruct":
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.7,
                max_model_len=131072,
                rope_scaling=rope_scaling_llama,
                trust_remote_code=True
            )            
        else:
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.7,  
                max_model_len=131072,
                rope_scaling=rope_scaling,
                trust_remote_code=True
            )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    async def generate(self, model_path, prompt, instruction=None, **sampling_kwargs):
        # Llama 3 Instruct：与 HuggingFace 官方用法一致，仅用模型自带 chat_template
        # https://huggingface.co/docs/transformers/main/en/chat_templating
        mp = model_path.lower()
        is_llama = "llama" in mp
        is_qwen = "qwen" in mp
        inst = instruction if instruction is not None else ""

        if is_llama:
            messages = [
                {"role": "system", "content": inst},
                {"role": "user", "content": prompt},
            ]
            texts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif is_qwen:
            messages = [{"role": "user", "content": inst + "\n" + prompt}]
            texts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            messages = [{"role": "user", "content": inst + "\n" + prompt}]
            texts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=sampling_kwargs.get('temperature', 0.0),
            top_p=sampling_kwargs.get('top_p', 1.0),
            max_tokens=sampling_kwargs.get('max_tokens', 2048),
            stop=sampling_kwargs.get('stop', None),
            skip_special_tokens=False
            
        )
        
        # 异步生成
        request_id = f"req_{id(prompt)}"

        async for request_output in self.engine.generate(
            texts, sampling_params, request_id
        ):
            pass
        return request_output.outputs[0].text

async def summary_detect(data,instruction,cache,model_path,long_context,inferencer=None):
    if inferencer is None:
        inferencer = VLLMInference(model_path,tensor_parallel_size=8,long_context=long_context)
    result = []
    batch_size = 32
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        
        # 批量生成
        tasks = []
        task_indices = []  # 记录需要推理的item在batch中的索引
        for idx, item in enumerate(batch):
            article = item["input"]
            instr = item.get("instruction", instruction)
            cache_key = (instr, article)
            # 检查缓存，如果存在则跳过

            if  cache_key in cache :
                _,ec=extract_json_from_text(cache[cache_key])
                if _!="":
                    continue  # cache中有的就跳过
                else:
                    #print(cache[cache_key])
                    # _=="" 时，需要重新推理，删除旧的无效缓存
                    del cache[cache_key]

            tasks.append(inferencer.generate(model_path,
                prompt=article,
                instruction=instr,
                temperature=0.0,
                max_tokens=8192
            ))
            task_indices.append(idx)  # 记录这个任务对应的batch索引
        
        # 并发执行
        if tasks:
            results = await asyncio.gather(*tasks)
            for j, text in enumerate(results):
                batch_idx = task_indices[j]  # 获取正确的batch索引
                it = batch[batch_idx]
                _instr = it.get("instruction", instruction)
                text = await ensure_model_text_json(inferencer, model_path, text)
                cache[(_instr, it["input"])] = text
        
        # 保存结果
        for item in batch:
            ans = {}
            article = item["input"]
            instr = item.get("instruction", instruction)
            text = cache.get((instr, article), "")
            ans["input"] = item["input"]
            ans["language"] = item["language"]
            _,ec=extract_json_from_text(text)
            ans["thinking"] = _
            if ec:
                ans["eval"] = ec
            else:
                ans["eval"] = text
            ans["output"] = item["output"]
            result.append(ans)
    return result
    

async def summary_detect_CoT(data,instruction,cache,model_path,long_context,inferencer=None):
    if inferencer is None:
        inferencer = VLLMInference(model_path,tensor_parallel_size=8,long_context=long_context)
    result = []
    batch_size = 32
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        # 批量生成
        tasks = []
        task_indices = []
        for idx, item in enumerate(batch):
            article = item["input"]
            instr = item.get("instruction", instruction)
            cache_key = (instr, article)
            if  cache_key in cache :
                _,ec=extract_json_from_text(cache[cache_key])
                if _!="":
                    continue  
                else:
                    #print(cache[cache_key])
                    # _=="" 时，需要重新推理，删除旧的无效缓存
                    del cache[cache_key]

            tasks.append(inferencer.generate(model_path,
                prompt=article,
                instruction=instr,
                temperature=0.0,
                max_tokens=8192
            ))
            task_indices.append(idx)  # 记录这个任务对应的batch索引
        
        # 并发执行
        if tasks:
            results = await asyncio.gather(*tasks)
            for j, text in enumerate(results):
                batch_idx = task_indices[j]  # 获取正确的batch索引
                it = batch[batch_idx]
                _instr = it.get("instruction", instruction)
                # CoT 模式保留模型原始推理内容，不做 strict JSON 修复
                cache[(_instr, it["input"])] = text
        
        # 保存结果
        for item in batch:
            ans = {}
            article = item["input"]
            instr = item.get("instruction", instruction)
            text = cache.get((instr, article), "")
            ans["input"] = item["input"]
            ans["language"] = item["language"]
            _,ec=extract_json_from_text(text)
            ans["thinking"] = _
            if ec:
                ans["eval"] = ec
            else:
                ans["eval"] = text
            ans["output"] = item["output"]
            result.append(ans)
    return result


async def summary_detect_PromptB(data, instruction, cache, model_path, long_context, inferencer=None):
    if inferencer is None:
        inferencer = VLLMInference(model_path, tensor_parallel_size=8, long_context=long_context)
    result = []
    batch_size = 32
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        tasks = []
        task_indices = []
        for idx, item in enumerate(batch):
            input_text = item["input"]
            instr = item.get("instruction", instruction)

            # 检测摘要分隔符，将 summary 前置
            sep = None
            summary_tag = ""
            summary = ""
            article = input_text
            if "\n摘要：" in input_text:
                sep = "\n摘要："
                summary_tag = "摘要："
            elif "\n概述：" in input_text:
                sep = "\n概述："
                summary_tag = "概述："
            elif "\nSummary:" in input_text:
                sep = "\nSummary:"
                summary_tag = "Summary:"
            elif "\nSummary：" in input_text:
                sep = "\nSummary："
                summary_tag = "Summary："
            elif "\nOverview:" in input_text:
                sep = "\nOverview:"
                summary_tag = "Overview:"
            elif "\nOverview：" in input_text:
                sep = "\nOverview："
                summary_tag = "Overview："

            if sep is not None and sep in input_text:
                article, rest = input_text.split(sep, 1)
                summary = rest
                inp = instr + f"\n{summary_tag}" + summary + "\n" + article
            else:
                inp = instr + "\n" + input_text

            cache_key = (instr, inp)
            if cache_key in cache:
                _, ec = extract_json_from_text(cache[cache_key])
                if _ != "":
                    continue
                else:
                    del cache[cache_key]

            tasks.append(inferencer.generate(
                model_path,
                prompt=inp,
                instruction=instr,
                temperature=0.0,
                max_tokens=8192
            ))
            task_indices.append(idx)

        if tasks:
            results = await asyncio.gather(*tasks)
            for j, text in enumerate(results):
                batch_idx = task_indices[j]
                it = batch[batch_idx]
                _instr = it.get("instruction", instruction)
                input_text = it["input"]

                sep = None
                summary_tag = ""
                summary = ""
                article = input_text
                if "\n摘要：" in input_text:
                    sep = "\n摘要："
                    summary_tag = "摘要："
                elif "\n概述：" in input_text:
                    sep = "\n概述："
                    summary_tag = "概述："
                elif "\nSummary:" in input_text:
                    sep = "\nSummary:"
                    summary_tag = "Summary:"
                elif "\nSummary：" in input_text:
                    sep = "\nSummary："
                    summary_tag = "Summary："
                elif "\nOverview:" in input_text:
                    sep = "\nOverview:"
                    summary_tag = "Overview:"
                elif "\nOverview：" in input_text:
                    sep = "\nOverview："
                    summary_tag = "Overview："

                if sep is not None and sep in input_text:
                    article, rest = input_text.split(sep, 1)
                    summary = rest
                    inp = _instr + f"\n{summary_tag}" + summary + "\n" + article
                else:
                    inp = _instr + "\n" + input_text

                cache[(_instr, inp)] = text

        for item in batch:
            ans = {}
            input_text = item["input"]
            instr = item.get("instruction", instruction)

            sep = None
            summary_tag = ""
            summary = ""
            article = input_text
            if "\n摘要：" in input_text:
                sep = "\n摘要："
                summary_tag = "摘要："
            elif "\n概述：" in input_text:
                sep = "\n概述："
                summary_tag = "概述："
            elif "\nSummary:" in input_text:
                sep = "\nSummary:"
                summary_tag = "Summary:"
            elif "\nSummary：" in input_text:
                sep = "\nSummary："
                summary_tag = "Summary："
            elif "\nOverview:" in input_text:
                sep = "\nOverview:"
                summary_tag = "Overview:"
            elif "\nOverview：" in input_text:
                sep = "\nOverview："
                summary_tag = "Overview："

            if sep is not None and sep in input_text:
                article, rest = input_text.split(sep, 1)
                summary = rest
                inp = instr + f"\n{summary_tag}" + summary + "\n" + article
            else:
                inp = instr + "\n" + input_text

            text = cache.get((instr, inp), "")
            ans["input"] = inp
            ans["language"] = item["language"]
            _, ec = extract_json_from_text(text)
            ans["thinking"] = _
            ans["eval"] = ec if ec else text
            ans["output"] = item["output"]
            result.append(ans)
    return result



# main函数
async def main(args):
    ROOT="/data/LongNovel/infer"
    if args.model.startswith("/"):
        model_path = args.model
    else:
        model_path = f"/data/model/{args.model}"
    long_context = args.long_context
    mode = args.mode
    DATA_ROOT = "/data/360-LLaMA-Factory/data"
    split = (getattr(args, "data_split", None) or "test").strip().lower()
    is_validation = split == "validation"
    cs = "_validation" if is_validation else ""
    out_tag = "_validation" if is_validation else ""

    is_AB = split == "ab"
    cs = "_AB" if is_AB else ""
    out_tag = "_AB" if is_AB else ""

    model=model_path.split("/")[-1]
    print(model, long_context, mode, "data_split=", split)
    if mode == "":
        cache = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}")
    else:
        cache = Cache(f"{ROOT}/cache_{model}_{long_context}_{mode}{cs}")

    if is_validation:
        if long_context == "16k":
            data_path = f"{DATA_ROOT}/validation_16k.json"
        elif long_context == "32k":
            data_path = f"{DATA_ROOT}/validation_32k.json"
        elif long_context in ("64k", "100k"):
            raise ValueError(
                "data_split=validation 时仅支持 validation_16k.json / validation_32k.json，请将 --long_context 设为 16k 或 32k"
            )
        else:
            raise ValueError(f"Invalid long_context: {long_context}")


    if is_AB:
        if long_context == "16k":
            data_path = f"{DATA_ROOT}/test_16k_B.json"
        elif long_context == "32k":
            data_path = f"{DATA_ROOT}/test_32k_B.json"
        elif long_context == "64k":
            data_path = f"{DATA_ROOT}/test_64k_B.json"
        elif long_context == "100k":
            data_path = f"{DATA_ROOT}/test_100k_B.json"
        else:
            raise ValueError(f"Invalid long_context: {long_context}")


    elif long_context == "100k":
        data_path = f"{DATA_ROOT}/test_100k.json"
    elif long_context == "64k":
        data_path = f"{DATA_ROOT}/test_64k.json"
    elif long_context == "32k":
        data_path = f"{DATA_ROOT}/test_32k.json"
    elif long_context == "16k":
        data_path = f"{DATA_ROOT}/test_16k.json"
    else:
        raise ValueError(f"Invalid long_context: {long_context}")

    #读取数据
    with open(data_path, "r") as f:
        data=json.load(f)
    inferencer = VLLMInference(model_path, tensor_parallel_size=8, long_context=long_context)
    
    if args.mode == "":

        #实际的prompt是summary_detect函数中得到的数据自带的instruction字段
        prompt = open(f"{ROOT}/prompt_v2_en.txt", "r").read() +"\n\n"

        cache_2 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}")
        result_2 = await summary_detect(data,prompt,cache_2,model_path,long_context,inferencer)
        output_dir_2 = f"{ROOT}/step_1"
        os.makedirs(output_dir_2, exist_ok=True)
        with open(f"{output_dir_2}/{model}_{long_context}_HD_result{out_tag}.json", "w") as f:
            json.dump(result_2, f, ensure_ascii=False, indent=4)

  
        cache_2 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_2")
        result_2 = await summary_detect(data,prompt,cache_2,model_path,long_context,inferencer)

        output_dir_2 = f"{ROOT}/step_2"
        os.makedirs(output_dir_2, exist_ok=True)
        with open(f"{output_dir_2}/{model}_{long_context}_HD_result{out_tag}.json", "w") as f:
            json.dump(result_2, f, ensure_ascii=False, indent=4)
        
        cache_3 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_3")
        result_3 = await summary_detect(data,prompt,cache_3,model_path,long_context,inferencer)
        output_dir_3 = f"{ROOT}/step_3"
        os.makedirs(output_dir_3, exist_ok=True)
        with open(f"{output_dir_3}/{model}_{long_context}_HD_result{out_tag}.json", "w") as f:
            json.dump(result_3, f, ensure_ascii=False, indent=4)

    elif args.mode == "CoT":
        prompt_zh = open(f"{ROOT}/prompt_v2_COT_zh.txt", "r").read() + "\n\n"
        prompt_en = open(f"{ROOT}/prompt_v2_en_COT.txt", "r").read() + "\n\n"

        cot_data = []
        for item in data:
            item_id = str(item.get("language", ""))
            if "zh" in item_id:
                instr = prompt_zh
            elif "en" in item_id:
                instr = prompt_en
            else:
                # 未命中语言标识时默认英文，避免中断整批推理
                instr = prompt_en
            new_item = dict(item)
            new_item["instruction"] = instr
            cot_data.append(new_item)
        # 第一次推理：保存到 step_1
        cache_1 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_CoT_1")
        result_1 = await summary_detect_CoT(cot_data, prompt_en, cache_1, model_path, long_context, inferencer)
        output_dir_1 = f"{ROOT}/step_1"
        os.makedirs(output_dir_1, exist_ok=True)
        with open(f"{output_dir_1}/{model}_{long_context}_HD_result_CoT{out_tag}.json", "w") as f:
            json.dump(result_1, f, ensure_ascii=False, indent=4)

        # 第二次推理：保存到 step_2
        cache_2 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_CoT_2")
        result_2 = await summary_detect_CoT(cot_data, prompt_en, cache_2, model_path, long_context, inferencer)
        output_dir_2 = f"{ROOT}/step_2"
        os.makedirs(output_dir_2, exist_ok=True)
        with open(f"{output_dir_2}/{model}_{long_context}_HD_result_CoT{out_tag}.json", "w") as f:
            json.dump(result_2, f, ensure_ascii=False, indent=4)

        # 第三次推理：保存到 step_3
        cache_3 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_CoT_3")
        result_3 = await summary_detect_CoT(cot_data, prompt_en, cache_3, model_path, long_context, inferencer)
        output_dir_3 = f"{ROOT}/step_3"
        os.makedirs(output_dir_3, exist_ok=True)
        with open(f"{output_dir_3}/{model}_{long_context}_HD_result_CoT{out_tag}.json", "w") as f:
            json.dump(result_3, f, ensure_ascii=False, indent=4)

    elif args.mode == "PromptB":
        prompt_zh = open(f"{ROOT}/prompt_v2_zh.txt", "r").read() + "\n\n"
        prompt_en = open(f"{ROOT}/prompt_v2_en.txt", "r").read() + "\n\n"
        # 按样本 id 动态分配 CoT prompt
        cot_data = []
        for item in data:
            item_id = str(item.get("language", ""))
            if "zh" in item_id:
                instr = prompt_zh
            elif "en" in item_id:
                instr = prompt_en
            else:
                # 未命中语言标识时默认英文，避免中断整批推理
                instr = prompt_en
            new_item = dict(item)
            new_item["instruction"] = instr
            cot_data.append(new_item)
        # 第一次推理：保存到 step_1
        cache_1 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_CoT_1")
        result_1 = await summary_detect_PromptB(cot_data, prompt_en, cache_1, model_path, long_context, inferencer)
        output_dir_1 = f"{ROOT}/step_1"
        os.makedirs(output_dir_1, exist_ok=True)
        with open(f"{output_dir_1}/{model}_{long_context}_HD_result_PromptB{out_tag}.json", "w") as f:
            json.dump(result_1, f, ensure_ascii=False, indent=4)

        # 第二次推理：保存到 step_2
        cache_2 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_CoT_2")
        result_2 = await summary_detect_PromptB(cot_data, prompt_en, cache_2, model_path, long_context, inferencer)
        output_dir_2 = f"{ROOT}/step_2"
        os.makedirs(output_dir_2, exist_ok=True)
        with open(f"{output_dir_2}/{model}_{long_context}_HD_result_PromptB{out_tag}.json", "w") as f:
            json.dump(result_2, f, ensure_ascii=False, indent=4)

        # 第三次推理：保存到 step_3
        cache_3 = Cache(f"{ROOT}/cache_{model}_{long_context}{cs}_CoT_3")
        result_3 = await summary_detect_PromptB(cot_data, prompt_en, cache_3, model_path, long_context, inferencer)
        output_dir_3 = f"{ROOT}/step_3"
        os.makedirs(output_dir_3, exist_ok=True)
        with open(f"{output_dir_3}/{model}_{long_context}_HD_result_PromptB{out_tag}.json", "w") as f:
            json.dump(result_3, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-32B")
    parser.add_argument("--long_context", type=str, default="64k")
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        help="test：使用 test_*.json（默认）；validation：16k/32k 使用 validation_16k.json / validation_32k.json（大小写不敏感，可写 validation）",
    )
    args = parser.parse_args()
    asyncio.run(main(args))