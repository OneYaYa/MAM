import sys
import os


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import *


import llava.model.language_model.llava_llama
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch

from PIL import Image

class HuatuoChatbot():
    def __init__(self, model_dir="Qwen/Qwen2-VL-2B-Instruct", device=None):  # 替换为Qwen2-VL-2B-Instruct作为默认模型
        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_dir = model_dir

        self.gen_kwargs = {
            'do_sample': True,
            'max_new_tokens': 512,
            'min_new_tokens': 1,
            'temperature': .2,
            'repetition_penalty': 1.2
        }
        self.device = device
        self.init_components()
        self.history = []
        self.images = []
        self.debug = True
        self.max_image_num = 6
        

    def init_components(self):
        
        d = self.model_dir
        if 'qwen2-vl' in d.lower():
            print(f'loading Qwen2-VL from {self.model_dir}')
            
            from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            import torch
            
            # 加载模型、分词器和处理器（避免使用 device_map 来解决兼容性问题）
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir='E:/hugging_face/huggingface'  # 明确指定缓存目录
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                cache_dir='E:/hugging_face/huggingface'  # 明确指定缓存目录
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                cache_dir='E:/hugging_face/huggingface'  # 明确指定缓存目录
            )
            
            # 手动移动模型到设备
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            # 设置生成参数
            self.gen_kwargs = {
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.2,
                'repetition_penalty': 1.1
            }
            
            # 设置模型为评估模式
            self.model.eval()
            
        elif 'huatuogpt-vision-7b' in d.lower():
            print(f'loading from {self.model_dir}')


            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            # 添加量化配置以减少内存使用（仅在支持时使用）
            use_quantization = True
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except:
                use_quantization = False

            # 使用原有的 llava 架构但添加量化和内存优化
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            
            # 准备加载参数
            load_kwargs = {
                'init_vision_encoder_from_ckpt': True,
                'output_loading_info': True,
                'torch_dtype': torch.float16,  # 使用float16以节省内存
                'low_cpu_mem_usage': True
            }
            
            if use_quantization:
                load_kwargs['quantization_config'] = quantization_config
                load_kwargs['device_map'] = "auto"
            
            model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
                self.model_dir,
                **load_kwargs
            )

            missing_keys = loading_info['missing_keys']
            unexpected_keys = loading_info['unexpected_keys']
            assert all(['vision_tower' in k for k in unexpected_keys])

            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
                vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(self.model_dir)
            
            # 根据是否使用量化设置数据类型
            if use_quantization:
                # 量化模型的vision tower不需要手动转换dtype
                pass
            else:
                vision_tower.to(dtype=torch.float16, device=model.device)
            
            image_processor = vision_tower.image_processor

            self.model = model
            self.tokenizer = tokenizer
            self.processor = image_processor
            self.model.config.tokenizer_padding_side = 'left'

            '''
            from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
            from transformers import AutoModel


            model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
                self.model_dir,
                init_vision_encoder_from_ckpt=True,
                output_loading_info=True,
                torch_dtype=torch.bfloat16
            
            '''
        elif 'huatuogpt' in d.lower():
            print(f'loading from {self.model_dir}')
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            model, loading_info = LlavaLlamaForCausalLM.from_pretrained(self.model_dir, init_vision_encoder_from_ckpt=True, output_loading_info=True, torch_dtype=torch.float16)
            missing_keys = loading_info['missing_keys']
            unexpected_keys = loading_info['unexpected_keys']
            assert all(['vision_tower' in k for k in unexpected_keys])

            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
                vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(self.model_dir)
            vision_tower.to(dtype=torch.float16, device=model.device)
            image_processor = vision_tower.image_processor

            self.model = model
            self.tokenizer = tokenizer
            self.processor = image_processor
            self.model.config.tokenizer_padding_side = 'left'

        else:
            raise NotImplementedError

        # 对于非Qwen2-VL模型，需要检查量化设置
        if 'qwen2-vl' not in d.lower():
            model.eval()
            if 'use_quantization' in locals() and not use_quantization:  
                self.model = self.model.to(self.device)


    def clear_history(self,):
        self.images = []
        self.history = []

    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None): # copied from llava
        prompt_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def preprocess(self, data: list, return_tensors='pt'):
        '''
        [
            {
                'from': 'human',
                'value': xxx,
            },
            {
                'from': 'gpt',
                'value': xxx
            }
        ]
        '''
        if not isinstance(data, list):
            raise ValueError('must be a list')        
        return self.preprocess_huatuo(data, return_tensors=return_tensors)
    
    def preprocess_huatuo(self, convs: list, return_tensors) -> list: # tokenize and concat the coversations
        input_ids = None
        convs = [ conv for conv in convs if conv['value'] is not None]
        round_num = len(convs)//2

        for ind in range(round_num):
            h = convs[ind*2]['value'].strip()
            h = f"<|user|>\n{h}\n" 

            g = convs[ind*2+1]['value']
            g = f"<|assistant|>\n{g} \n"

            cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)

            if input_ids is None:
                input_ids = cur_input_ids
            else:
                input_ids = torch.cat([input_ids, cur_input_ids])
            
            cur_input_ids = self.tokenizer(g, add_special_tokens= False, truncation=True, return_tensors='pt').input_ids[0]
            input_ids = torch.cat([input_ids, cur_input_ids])
        
        h = convs[-1]['value'].strip()
        h = f"<|user|>\n{h}\n<|assistant|>\n"
        cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)

        if input_ids is None:
            input_ids = cur_input_ids
        else:
            input_ids = torch.cat([input_ids, cur_input_ids])
        
        if self.debug:
            self.debug = False

        return input_ids


    def input_moderation(self, t: str):
        blacklist = ['<image>', '<s>', '</s>']
        for b in blacklist:
            t = t.replace(b, '')
        return t
    
    def insert_image_placeholder(self, t, num_images, placeholder='<image>', sep='\n'):
        for _ in range(num_images):
            t = f"{placeholder}{sep}" + t

        return t
    
    def get_conv(self, text):
        ret = []
        if self.history is None:
            self.history = []
        
        for conv in self.history:
            ret.append({'from': 'human', 'value': conv[0]})
            ret.append({'from': 'gpt', 'value': conv[1]})

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})

        return ret

    def get_conv_without_history(self, text):
        ret = []

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})

        return ret
    
    def get_image_tensors(self, images):
        list_image_tensors = []
        crop_size = self.processor.crop_size
        processor = self.processor
        for fp in images:
            if fp is None: # None is used as a placeholder
                continue
            elif isinstance(fp, str):
                image = Image.open(fp).convert('RGB')
            elif isinstance(fp, Image.Image):
                image = fp # already an image
            else:
                raise TypeError(f'Unsupported type {type(fp)}')

            if True or self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0] # a tensor
            list_image_tensors.append(image.to(self.device))
        if len(list_image_tensors) == 0:
            list_image_tensors.append(torch.zeros(3, crop_size['height'], crop_size['width']).to(self.device))
        return list_image_tensors

    def inference(self, text, images=None):
        '''
        text: str
        images: list[str]
        '''
        
        # 检查是否为 Qwen2-VL 模型
        if 'qwen2-vl' in self.model_dir.lower():
            return self.inference_qwen2vl(text, images)
        
        # 原有的 inference 逻辑（用于其他模型）
        # image
        if images is None:
            images = []

        if isinstance(images,str):
            images = [images]

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                valid_images.append(img)
            except:
                print(f'{img} This image is wrong.')
                continue
        images = valid_images
        if len(valid_images) > self.max_image_num:
            images = images[:self.max_image_num]

        # text
        text = self.input_moderation(text)
        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)

        conv = self.get_conv_without_history(text)
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        if len(images) > 0:
            list_image_tensors = self.get_image_tensors(images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)
        else:
            image_tensors = None

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                use_cache=True,
                **self.gen_kwargs)
        answers = []
        for output_id in output_ids:
            answers.append(self.tokenizer.decode(output_id, skip_special_tokens=True).strip())
        return answers

    def inference_qwen2vl(self, text, images=None):
        '''
        专门为 Qwen2-VL 设计的推理方法
        '''
        if images is None:
            images = []
            
        if isinstance(images, str):
            images = [images]
        
        # 验证图片
        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB')
                    valid_images.append({"type": "image", "image": img})
                elif isinstance(img, Image.Image):
                    # 保存临时图片
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img.save(tmp.name)
                        valid_images.append({"type": "image", "image": tmp.name})
            except:
                print(f'{img} This image is wrong.')
                continue
        
        if len(valid_images) > self.max_image_num:
            valid_images = valid_images[:self.max_image_num]
        
        # 构建消息格式
        content = []
        content.extend(valid_images)
        content.append({"type": "text", "text": text})
        
        messages = [{"role": "user", "content": content}]
        
        # 应用聊天模板
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理输入 - 修复图像路径问题（inference_qwen2vl）
        try:
            # 直接使用 PIL.Image 对象而非路径
            from PIL import Image as PILImage
            pil_images = []
            if valid_images:
                for img_info in valid_images:
                    img_path = img_info["image"]
                    # 转换为绝对路径
                    if not os.path.isabs(img_path):
                        img_path = os.path.abspath(img_path)
                    
                    if os.path.exists(img_path):
                        # 直接加载为 PIL.Image 对象
                        pil_img = PILImage.open(img_path).convert('RGB')
                        pil_images.append(pil_img)
                    else:
                        print(f"[WARN] 图像文件不存在: {img_path}")
            
            inputs = self.processor(
                text=[text_prompt],
                images=pil_images if pil_images else None,
                padding=True,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"[ERROR] 图像处理失败: {e}")
            print("[INFO] 回退到纯文本处理...")
            # 回退到纯文本处理
            inputs = self.processor(
                text=[text_prompt],
                images=None,
                padding=True,
                return_tensors="pt"
            )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.gen_kwargs
            )
        
        # 解码
        generated_ids = [
            output_ids[i][len(inputs["input_ids"][i]):] 
            for i in range(len(output_ids))
        ]
        
        response = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return [response]

    def chat(self, text: str, images: list[str]=None, ):
        '''
        images: list[str], images for this round
        text: str
        '''
        text = self.input_moderation(text)
        if text == '':
            return 'Please type in something'

        if isinstance(images, str) or isinstance(images, Image.Image):
            images = [images]
        
        # 检查是否为 Qwen2-VL 模型
        if 'qwen2-vl' in self.model_dir.lower():
            return self.chat_qwen2vl(text, images)
        
        # 原有的 chat 逻辑（用于其他模型）
        valid_images = []
        if images is None:
            images = []
        
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                valid_images.append(img)
            except:
                continue

        images = valid_images

        self.images.extend(images)


        assert len(images) < self.max_image_num, f'at most {self.max_image_num} images'

        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)
        # make conv
        conv = self.get_conv(text)
        # make input ids
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        if len(self.images) > 0:
            list_image_tensors = self.get_image_tensors(self.images)
            image_tensors = torch.stack(list_image_tensors)
        else:
            image_tensors = None

        streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs=input_ids,images=image_tensors.to(dtype=torch.bfloat16) if image_tensors is not None else image_tensors, streamer=streamer,use_cache=True,**self.gen_kwargs)


        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ''
            sep = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
            for new_text in streamer:
                if sep in new_text:
                    new_text = self.remove_overlap(generated_text,new_text[:-len(sep)])
                    for char in new_text:
                        generated_text += char
                        #print(char,end='',flush = True)
                    break
                for char in new_text:
                    generated_text += char
                    #print(char,end='',flush = True)
        answer = generated_text

        self.history.append([text, answer])

        return answer

    def chat_qwen2vl(self, text: str, images: list[str]=None):
        '''
        专门为 Qwen2-VL 设计的聊天方法
        '''
        if images is None:
            images = []
            
        # 验证图片
        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB')
                    valid_images.append({"type": "image", "image": img})
                elif isinstance(img, Image.Image):
                    # 保存临时图片
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img.save(tmp.name)
                        valid_images.append({"type": "image", "image": tmp.name})
            except:
                continue
        
        if len(valid_images) > self.max_image_num:
            valid_images = valid_images[:self.max_image_num]
        
        # 构建对话历史
        messages = []
        
        # 添加历史对话
        for hist in self.history:
            messages.append({"role": "user", "content": [{"type": "text", "text": hist[0]}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": hist[1]}]})
        
        # 添加当前消息
        content = []
        content.extend(valid_images)
        content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})
        
        # 应用聊天模板
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理输入 - 修复图像路径问题（chat_qwen2vl）
        try:
            # 直接使用 PIL.Image 对象而非路径
            from PIL import Image as PILImage
            pil_images = []
            if valid_images:
                for img_info in valid_images:
                    img_path = img_info["image"]
                    # 转换为绝对路径
                    if not os.path.isabs(img_path):
                        img_path = os.path.abspath(img_path)
                    
                    if os.path.exists(img_path):
                        # 直接加载为 PIL.Image 对象
                        pil_img = PILImage.open(img_path).convert('RGB')
                        pil_images.append(pil_img)
                    else:
                        print(f"[WARN] 图像文件不存在: {img_path}")
            
            inputs = self.processor(
                text=[text_prompt],
                images=pil_images if pil_images else None,
                padding=True,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"[ERROR] 图像处理失败: {e}")
            print("[INFO] 回退到纯文本处理...")
            # 回退到纯文本处理
            inputs = self.processor(
                text=[text_prompt],
                images=None,
                padding=True,
                return_tensors="pt"
            )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 使用流式生成
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            **self.gen_kwargs
        )
        
        # 开始生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            #print(new_text, end='', flush=True)
        
        # 添加到历史记录
        self.history.append([text, generated_text])
        
        return generated_text

    def remove_overlap(self, existing_text, new_text):
        '''
        移除新文本中与现有文本重叠的部分
        '''
        if not existing_text or not new_text:
            return new_text
        
        # 查找最长的重叠后缀
        max_overlap = min(len(existing_text), len(new_text))
        for i in range(max_overlap, 0, -1):
            if existing_text[-i:] == new_text[:i]:
                return new_text[i:]
        
        return new_text


# if __name__ =="__main__":

#     import argparse
#     parser = argparse.ArgumentParser(description='Args of Data Preprocess')

#     parser.add_argument('--model_dir', default='Qwen/Qwen2-VL-2B-Instruct', type=str)
#     parser.add_argument('--device', default='cuda:0', type=str)
#     args = parser.parse_args()

#     bot = HuatuoChatbot(args.model_dir, args.device)

#     # test
#     # print(bot.inference('what show in this picture?',['./output.png']))
#     # print(bot.inference('hi'))

#     while True:
#         images = input('images, split by ",": ')
#         images = [i.strip() for i in images.split(',') if len(i.strip()) > 1 ]
#         text = input('USER ("clear" to clear history, "q" to exit): ')
#         if text.lower() in ['q', 'quit']:
#             exit()

#         if text.lower() == 'clear':
#             bot.history = []
#             bot.images = []
#             continue

#         answer = bot.chat(images=images, text=text)

#         images = None # already in the history

#         print()
#         print(f'GPT: {answer}')
#         print()
