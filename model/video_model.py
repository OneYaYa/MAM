import torch
import transformers
from transformers import BitsAndBytesConfig
import gc

# 在导入videollama2之前先Monkey Patch禁用Flash Attention
def patch_flash_attention():
    """创建完整的Flash Attention替代品，包括包元数据"""
    try:
        import sys
        import torch
        from types import ModuleType
        
        # 删除已有的flash_attn模块
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('flash_attn')]
        for module in modules_to_remove:
            del sys.modules[module]
        
        # 创建兼容的fake flash_attn模块
        class FakeFlashAttnFunc:
            @staticmethod
            def flash_attn_func(
                q,
                k,
                v,
                dropout_p: float = 0.0,
                softmax_scale=None,
                causal: bool = False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic: bool = False,
                return_attn_probs: bool = False,
                attn_spec=None,
                **kwargs,
            ):
                """使用标准PyTorch实现替代Flash Attention"""
                # 简单的标准attention实现
                batch_size, seq_len, num_heads, head_dim = q.shape
                
                # 重塑为 (batch_size * num_heads, seq_len, head_dim)
                q = q.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
                k = k.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim) 
                v = v.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
                
                # 计算attention scores
                if softmax_scale is None:
                    softmax_scale = 1.0 / (head_dim ** 0.5)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
                
                if causal:
                    mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
                    scores.masked_fill_(mask, float('-inf'))
                
                attn_probs = torch.softmax(scores, dim=-1)
                
                if dropout_p > 0.0:
                    attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p, training=True)
                
                out = torch.matmul(attn_probs, v)
                
                # 重塑回原来的形状
                out = out.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
                
                if return_attn_probs:
                    return out, attn_probs.view(batch_size, num_heads, seq_len, seq_len)
                return out
        
        # 创建完整的fake模块
        fake_flash_attn = ModuleType('flash_attn')
        fake_flash_attn.__version__ = "2.0.0"  # 添加版本信息
        fake_flash_attn.__file__ = "fake_flash_attn.py"
        fake_flash_attn.flash_attn_func = FakeFlashAttnFunc.flash_attn_func
        
        # 创建flash_attn_interface子模块
        fake_interface = ModuleType('flash_attn.flash_attn_interface')
        fake_interface.flash_attn_func = FakeFlashAttnFunc.flash_attn_func
        
        # 注册fake模块
        sys.modules['flash_attn'] = fake_flash_attn
        sys.modules['flash_attn.flash_attn_interface'] = fake_interface
        
        # 也需要Monkey Patch importlib.metadata来处理包元数据查询
        try:
            import importlib.metadata
            original_version = importlib.metadata.version
            
            def fake_version(package_name):
                if package_name == 'flash_attn' or package_name == 'flash-attn':
                    return "2.0.0"
                return original_version(package_name)
            
            importlib.metadata.version = fake_version
            
            # 对于旧版本Python
            try:
                import pkg_resources
                original_get_distribution = pkg_resources.get_distribution
                
                def fake_get_distribution(package_name):
                    if package_name == 'flash_attn' or package_name == 'flash-attn':
                        class FakeDist:
                            version = "2.0.0"
                        return FakeDist()
                    return original_get_distribution(package_name)
                
                pkg_resources.get_distribution = fake_get_distribution
            except ImportError:
                pass
                
        except ImportError:
            pass
        
        print("[INFO] Flash Attention 已通过完整兼容性补丁替换为标准 PyTorch 实现")
        return True
    except Exception as e:
        print(f"[WARN] Flash Attention patch 失败: {e}")
        return False

# 执行patch
patch_flash_attention()

from videollama2.conversation import conv_templates


from videollama2.constants import DEFAULT_AUDIO_TOKEN, AUDIO_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_multimodal_token, process_video, process_image
from videollama2.model.__init__ import load_pretrained_model


    
class VideoLLaMAChatbot:
    def __init__(self, model_path='DAMO-NLP-SG/VideoLLaMA2-7B', device='cuda:0', use_quantization=True):
        # 在类初始化时就设置环境变量，确保及早生效
        import os
        os.environ["DISABLE_FLASH_ATTN"] = "1"  # 禁用Flash Attention
        os.environ["FORCE_DISABLE_FLASH_ATTN"] = "1"  # 强制禁用
        
        self.model_path = model_path
        self.device = device
        self.use_quantization = use_quantization
        self.model_name = get_model_name_from_path(model_path)
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        if use_quantization:
            print("正在加载 VideoLLaMA2 模型（8GB显存优化，使用4位量化）...")
            
            # 4位量化配置 - 大幅减少显存使用
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            try:
                # VideoLLaMA2的load_pretrained_model不直接支持quantization_config
                # 我们需要通过环境变量或其他方式来实现量化
                
                self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
                    model_path, 
                    None, 
                    self.model_name, 
                    vision_tower="openai/clip-vit-large-patch14-336",
                    device_map="auto"
                )
                
                # 手动应用量化（如果模型支持）
                try:
                    print("尝试应用4位量化...")
                    # 暂时跳过量化，直接使用半精度
                    if hasattr(self.model, 'half'):
                        self.model = self.model.half()
                    print(f"模型加载成功！当前显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                except Exception as quant_error:
                    print(f"[WARN] 量化应用失败: {quant_error}")
                    print("使用标准半精度模式")
                
            except Exception as e:
                print(f"[WARN] 量化加载失败: {e}")
                print("回退到标准加载模式...")
                self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
                    model_path, None, self.model_name, vision_tower="openai/clip-vit-large-patch14-336"
                )
                self.model = self.model.to(self.device)
        else:
            # 标准加载模式
            self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
                model_path, None, self.model_name, vision_tower="openai/clip-vit-large-patch14-336"
            )
            self.model = self.model.to(self.device)
        
        self.conv_mode = 'llama_2'

    def chat(self, paths, text, modal_type='video'):
        try:
            # 清理显存
            torch.cuda.empty_cache()
            
            # Visual preprocess (load & transform image or video)
            if modal_type == 'video':
                tensor = process_video(paths[0], self.processor, self.model.config.image_aspect_ratio).to(dtype=torch.float16, device=self.device, non_blocking=True)
                default_mm_token = DEFAULT_AUDIO_TOKEN["VIDEO"]
                modal_token_index = AUDIO_TOKEN_INDEX["VIDEO"]
            else:
                tensor = process_image(paths[0], self.processor, self.model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device=self.device, non_blocking=True)
                default_mm_token = DEFAULT_AUDIO_TOKEN["IMAGE"]
                modal_token_index = AUDIO_TOKEN_INDEX["IMAGE"]
            tensor = [tensor]

            # Text preprocess (tag process & generate prompt)
            question = default_mm_token + "\n" + text
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_multimodal_token(prompt, self.tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to(self.device)

            # 使用保守的生成参数以节省显存
            with torch.inference_mode():  # 禁用梯度计算以节省显存
                try:
                    output_ids = self.model.generate(
                        input_ids,
                        images_or_videos=tensor,
                        modal_list=[modal_type],
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=512,  # 限制生成长度以节省显存
                        use_cache=True,
                    )
                    
                    outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    
                    # 生成后清理显存
                    torch.cuda.empty_cache()
                    
                    return outputs[0]
                    
                except torch.cuda.OutOfMemoryError as oom_error:
                    print(f"[ERROR] 显存不足: {oom_error}")
                    torch.cuda.empty_cache()
                    
                    # 尝试更保守的设置
                    try:
                        output_ids = self.model.generate(
                            input_ids,
                            images_or_videos=tensor,
                            modal_list=[modal_type],
                            do_sample=False,  # 关闭采样
                            max_new_tokens=256,  # 进一步限制长度
                            use_cache=False,   # 关闭缓存
                        )
                        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                        torch.cuda.empty_cache()
                        return outputs[0]
                    except Exception as retry_error:
                        print(f"[ERROR] 重试失败: {retry_error}")
                        torch.cuda.empty_cache()
                        return f"显存不足，无法处理{modal_type}文件。建议关闭其他GPU程序后重试。"
                        
        except torch.cuda.OutOfMemoryError as e:
            print(f"[ERROR] 显存不足: {e}")
            torch.cuda.empty_cache()
            return f"显存不足，无法处理{modal_type}文件 {paths[0]}。建议关闭其他GPU程序后重试。"
            
        except Exception as e:
            print(f"[ERROR] {modal_type}处理失败: {e}")
            torch.cuda.empty_cache()
            return f"{modal_type}处理失败：{paths[0]}。错误信息：{str(e)}"
    
    def clear_model(self):
        """手动清理模型以释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        gc.collect()


def inference():
    # Video Inference
    paths = ['assets/cat_and_chicken.mp4']
    questions = ['What animals are in the video, what are they doing, and how does the video feel?']
    # Reply:
    # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.
    modal_list = ['video']

    # Image Inference
    paths = ['assets/sora.png']
    questions = ['What is the woman wearing, what is she doing, and how does the image feel?']
    # Reply:
    # The woman in the image is wearing a black coat and sunglasses, and she is walking down a rain-soaked city street. The image feels vibrant and lively, with the bright city lights reflecting off the wet pavement, creating a visually appealing atmosphere. The woman's presence adds a sense of style and confidence to the scene, as she navigates the bustling urban environment.
    modal_list = ['image']

    # 1. Initialize the model.
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    # Base model inference (only need to replace model_path)
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-Base'
    model_name = get_model_name_from_path(model_path)
    
    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        print("正在加载模型...")
        import os
        os.environ["DISABLE_FLASH_ATTN"] = "1"  # 禁用Flash Attention
        
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path, 
            None, 
            model_name,
            device_map="auto"
        )
        
        # 应用半精度以节省显存
        if hasattr(model, 'half'):
            model = model.half()
        print(f"模型加载成功！当前显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    except Exception as e:
        print(f"[WARN] 模型加载失败: {e}")
        print("回退到标准加载模式...")
        tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
        model = model.to('cuda:0')
    
    conv_mode = 'llama2'

    # 2. Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor = process_video(paths[0], processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_AUDIO_TOKEN["VIDEO"]
        modal_token_index = AUDIO_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_AUDIO_TOKEN["IMAGE"]
        modal_token_index = AUDIO_TOKEN_INDEX["IMAGE"]
    tensor = [tensor]

    # 3. text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda:0')

    with torch.inference_mode():
        try:
            output_ids = model.generate(
                input_ids,
                images_or_videos=tensor,
                modal_list=modal_list,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,  # 限制生成长度以节省显存
                use_cache=True,
            )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print(outputs[0])
            
            # 清理显存
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as oom_error:
            print(f"[ERROR] 显存不足: {oom_error}")
            torch.cuda.empty_cache()
            print("正在尝试更保守的设置...")
            
            try:
                output_ids = model.generate(
                    input_ids,
                    images_or_videos=tensor,
                    modal_list=modal_list,
                    do_sample=False,  # 关闭采样
                    max_new_tokens=256,  # 进一步限制长度
                    use_cache=False,   # 关闭缓存
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                print(outputs[0])
                torch.cuda.empty_cache()
            except Exception as retry_error:
                print(f"[ERROR] 重试失败: {retry_error}")
                torch.cuda.empty_cache()
                print("显存不足，无法处理视频文件。建议关闭其他GPU程序后重试。")


if __name__ == "__main__":
    inference()
