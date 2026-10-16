"""
性能监控模块 - 记录Cases/s, Sec/case, Tokens/s, GPU利用率
"""

import subprocess
import time
import json
from typing import Dict, Optional, Tuple


class PerformanceMonitor:
    """性能指标监控器"""
    
    def __init__(self, case_index: int):
        self.case_index = case_index
        self.start_time = None
        self.end_time = None
        self.token_count = 0
        self.gpu_util_start = None
        self.gpu_util_end = None
        self.gpu_power_start = None
        self.gpu_power_end = None
        self.gpu_mem_start = None
        self.gpu_mem_end = None
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.gpu_util_start, self.gpu_power_start, self.gpu_mem_start = self._get_gpu_stats()
    
    def end(self):
        """结束监控"""
        self.end_time = time.time()
        self.gpu_util_end, self.gpu_power_end, self.gpu_mem_end = self._get_gpu_stats()
    
    def set_token_count(self, tokens: int):
        """记录生成的token数"""
        self.token_count = tokens
    
    def _get_gpu_stats(self) -> Tuple[float, float, float]:
        """获取GPU利用率、功耗、显存占用（百分比）"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,utilization.memory", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_util = float(parts[0].strip()) if len(parts) > 0 else 0.0
                # power.draw 格式是 "123.45 W"，需要提取数字
                power_str = parts[1].strip() if len(parts) > 1 else "0"
                power = float(power_str.replace("W", "").strip()) if power_str else 0.0
                gpu_mem = float(parts[2].strip()) if len(parts) > 2 else 0.0
                return gpu_util, power, gpu_mem
        except Exception as e:
            pass
        return 0.0, 0.0, 0.0
    
    def get_metrics(self) -> Dict:
        """获取所有性能指标"""
        if not self.start_time or not self.end_time:
            return None
        
        elapsed = self.end_time - self.start_time
        tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
        
        # GPU平均利用率
        avg_gpu_util = (self.gpu_util_start + self.gpu_util_end) / 2 if self.gpu_util_start and self.gpu_util_end else 0
        avg_gpu_power = (self.gpu_power_start + self.gpu_power_end) / 2 if self.gpu_power_start and self.gpu_power_end else 0
        avg_gpu_mem = (self.gpu_mem_start + self.gpu_mem_end) / 2 if self.gpu_mem_start and self.gpu_mem_end else 0
        
        return {
            "case_index": self.case_index,
            "elapsed_seconds": elapsed,
            "token_count": self.token_count,
            "tokens_per_second": tokens_per_sec,
            "gpu_util_percent_avg": avg_gpu_util,
            "gpu_power_watts_avg": avg_gpu_power,
            "gpu_mem_percent_avg": avg_gpu_mem,
            "gpu_util_start_percent": self.gpu_util_start,
            "gpu_util_end_percent": self.gpu_util_end,
        }


def estimate_tokens_from_json_output(json_output: str) -> int:
    """估算Qwen输出JSON中的token数
    
    粗略估计：中文字符通常是1token，英文单词平均1.3字符=0.77token
    这里统一按 字符数 / 1.5 估算
    """
    if not json_output:
        return 0
    
    # 统计字符数（包含所有字符）
    char_count = len(json_output)
    
    # 粗略估算token数（Qwen tokenizer中文通常1字符=1token，英文0.5token/字符）
    # 简化估算为 char_count / 1.5
    estimated_tokens = max(1, int(char_count / 1.5))
    
    return estimated_tokens


def get_gpu_info() -> Tuple[str, str]:
    """获取GPU名称和总显存"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            gpu_name = parts[0].strip() if len(parts) > 0 else "Unknown"
            gpu_memory = parts[1].strip() if len(parts) > 1 else "Unknown"
            return gpu_name, gpu_memory
    except Exception as e:
        pass
    return "Unknown GPU", "Unknown Memory"
