from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import gc  # 添加垃圾回收模块
import os  # 添加操作系统模块

# 兼容性修复：为新版本 transformers 添加 BeamSearchScorer 别名
try:
    from transformers import BeamSearchScorer
    print("[INFO] BeamSearchScorer 从主模块导入成功")
except ImportError:
    try:
        # transformers 4.57+ 版本的正确导入路径
        from transformers.generation.beam_search import BeamSearchScorer
        print("[INFO] BeamSearchScorer 从 beam_search 模块导入成功")
        # 将别名添加到 transformers 模块，供内部依赖使用
        import transformers
        transformers.BeamSearchScorer = BeamSearchScorer
    except ImportError:
        try:
            # 备用路径
            from transformers.generation.utils import BeamSearchScorer
            print("[INFO] BeamSearchScorer 从 generation.utils 模块导入成功")
            import transformers
            transformers.BeamSearchScorer = BeamSearchScorer
        except ImportError:
            print("[WARN] 无法导入 BeamSearchScorer，某些功能可能受限")
            # 创建一个占位符类以避免错误
            class BeamSearchScorer:
                pass
            import transformers
            transformers.BeamSearchScorer = BeamSearchScorer

print("[INFO] BeamSearchScorer 兼容性修复完成")

torch.manual_seed(1234)

class AudioChatbot():
    def __init__(self, cache_path='E:/hugging_face/huggingface', device="cpu"):
        self.cache_path = cache_path
        # 强制使用 CPU 避免显存问题
        self.device = "cpu"
        print(f"AudioChatbot 初始化完成，使用设备: {self.device}")
        
    def chat(self, audio: str, text: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-Audio-Chat",
                cache_dir=self.cache_path, 
                trust_remote_code=True
            )
            
            # 清理显存（如果有GPU的话）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-Audio-Chat",
                cache_dir=self.cache_path, 
                device_map="cpu",  # 强制使用 CPU
                torch_dtype=torch.float32,  # 使用 float32 确保兼容性
                low_cpu_mem_usage=True,  # 优化 CPU 内存使用
                trust_remote_code=True
            ).eval()
            

            
            query = tokenizer.from_list_format([
                {'audio': audio}, 
                {'text': text}
            ])
            
            # 使用 no_grad 减少内存占用
            with torch.no_grad():
                response, history = model.chat(
                    tokenizer, 
                    query=query, 
                    history=None,
                    max_new_tokens=512,  # 限制生成长度
                    do_sample=True,
                    temperature=0.7
                )
            
            # 清理内存
            del model
            del tokenizer
            gc.collect()
            

            return response
            
        except Exception as e:
            print(f"[ERROR] 处理失败: {e}")
            # 确保清理内存
            try:
                del model
                del tokenizer
            except:
                pass
            gc.collect()
            return f"处理失败: {str(e)}"
    
    def clear_memory(self):
        """手动清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("内存清理完成")
    
    def get_memory_info(self):
        """获取内存使用情况"""
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"系统内存使用率: {memory_info.percent}%")
        print(f"可用内存: {memory_info.available / (1024**3):.2f} GB")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU内存使用: {gpu_memory:.2f} GB")
        else:
            print("未使用GPU，运行在CPU模式")